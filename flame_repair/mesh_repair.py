import trimesh
import numpy as np
from typing import Optional


def repair_mesh(mesh: trimesh.Trimesh, verbose: bool = True) -> trimesh.Trimesh:
    stats = {"original_verts": len(mesh.vertices), "original_faces": len(mesh.faces)}

    mesh = _remove_degenerate_faces(mesh)
    mesh = _remove_duplicate_faces(mesh)
    mesh = _merge_close_vertices(mesh)
    mesh = _fix_normals(mesh)
    mesh = _fill_holes(mesh)
    mesh = _remove_unreferenced_vertices(mesh)

    stats["repaired_verts"] = len(mesh.vertices)
    stats["repaired_faces"] = len(mesh.faces)

    if verbose:
        print(f"[Repair] Vertices: {stats['original_verts']} -> {stats['repaired_verts']}")
        print(f"[Repair] Faces: {stats['original_faces']} -> {stats['repaired_faces']}")
        print(f"[Repair] Watertight: {mesh.is_watertight}")

    return mesh


def _remove_degenerate_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    areas = mesh.area_faces
    valid = areas > 1e-10
    if not valid.all():
        removed = (~valid).sum()
        mesh.update_faces(valid)
        print(f"[Repair] Removed {removed} degenerate faces")
    return mesh


def _remove_duplicate_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    sorted_faces = np.sort(mesh.faces, axis=1)
    _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
    if len(unique_idx) < len(mesh.faces):
        removed = len(mesh.faces) - len(unique_idx)
        mask = np.zeros(len(mesh.faces), dtype=bool)
        mask[unique_idx] = True
        mesh.update_faces(mask)
        print(f"[Repair] Removed {removed} duplicate faces")
    return mesh


def _merge_close_vertices(mesh: trimesh.Trimesh, tol: float = 1e-8) -> trimesh.Trimesh:
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def _fix_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.fix_normals()
    return mesh


def _fill_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if mesh.is_watertight:
        return mesh
    try:
        trimesh.repair.fill_holes(mesh)
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
    except Exception as e:
        print(f"[Repair] Hole filling partial: {e}")
    return mesh


def _remove_unreferenced_vertices(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.remove_unreferenced_vertices()
    return mesh


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 3, lamb: float = 0.5) -> trimesh.Trimesh:
    try:
        trimesh.smoothing.filter_laplacian(mesh, lamb=lamb, iterations=iterations)
    except Exception:
        trimesh.smoothing.filter_humphrey(mesh, iterations=iterations)
    return mesh


def remesh_uniform(mesh: trimesh.Trimesh, target_faces: Optional[int] = None) -> trimesh.Trimesh:
    try:
        import open3d as o3d
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()

        if target_faces is None:
            target_faces = len(mesh.faces)

        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_non_manifold_edges()

        result = trimesh.Trimesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles),
            process=True,
        )
        print(f"[Repair] Remeshed: {len(result.faces)} faces")
        return result
    except ImportError:
        print("[Repair] open3d not available, skipping remesh")
        return mesh


def reconstruct_body(mesh: trimesh.Trimesh, poisson_depth: int = 8) -> trimesh.Trimesh:
    import open3d as o3d
    from scipy.spatial import cKDTree

    print(f"[Reconstruct] Poisson reconstruction (depth={poisson_depth})...")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()

    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=min(len(verts) * 3, 300000))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    recon_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, scale=1.1, linear_fit=False
    )

    densities_np = np.asarray(densities)
    density_threshold = np.percentile(densities_np, 5)
    vertices_to_remove = densities_np < density_threshold
    recon_mesh.remove_vertices_by_mask(vertices_to_remove)

    recon_mesh.remove_degenerate_triangles()
    recon_mesh.remove_duplicated_vertices()
    recon_mesh.remove_non_manifold_edges()

    recon_verts = np.asarray(recon_mesh.vertices, dtype=np.float64)
    recon_faces = np.asarray(recon_mesh.triangles)

    if len(recon_verts) == 0 or len(recon_faces) == 0:
        print("[Reconstruct] Reconstruction produced empty mesh, returning original")
        return mesh

    tree = cKDTree(verts)
    dists, _ = tree.query(recon_verts)
    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    keep_threshold = bbox_diag * 0.05

    face_verts_dists = dists[recon_faces]
    face_max_dist = face_verts_dists.max(axis=1)
    valid_faces = face_max_dist < keep_threshold

    if valid_faces.sum() < 100:
        keep_threshold = bbox_diag * 0.15
        face_max_dist = face_verts_dists.max(axis=1)
        valid_faces = face_max_dist < keep_threshold

    filtered_faces = recon_faces[valid_faces]
    result = trimesh.Trimesh(vertices=recon_verts, faces=filtered_faces, process=True)
    result = _keep_largest_component(result)
    result.fix_normals()

    print(f"[Reconstruct] Result: {len(result.vertices)} verts, {len(result.faces)} faces, watertight={result.is_watertight}")
    return result


def blend_reconstruction(
    original: trimesh.Trimesh,
    reconstructed: trimesh.Trimesh,
    distance_threshold: Optional[float] = None,
) -> trimesh.Trimesh:
    from scipy.spatial import cKDTree

    orig_verts = np.asarray(original.vertices, dtype=np.float64)
    recon_verts = np.asarray(reconstructed.vertices, dtype=np.float64)

    tree = cKDTree(orig_verts)
    dists, indices = tree.query(recon_verts)

    if distance_threshold is None:
        bbox_diag = float(np.linalg.norm(orig_verts.max(axis=0) - orig_verts.min(axis=0)))
        distance_threshold = bbox_diag * 0.02

    support = np.exp(-((dists / distance_threshold) ** 2)).reshape(-1, 1)
    matched_orig = orig_verts[indices]
    blended_verts = recon_verts * (1.0 - support) + matched_orig * support

    repaired_ratio = float(np.mean(dists > distance_threshold))
    print(f"[Blend] Original detail preserved where supported; FLAME/Poisson prior used on {repaired_ratio:.1%} of vertices")

    result = trimesh.Trimesh(vertices=blended_verts, faces=reconstructed.faces, process=False)
    result.fix_normals()
    return result


def _keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=3)
    components = list(components)
    if not components:
        return mesh
    largest = max(components, key=len)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[largest] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def auto_poisson_depth(num_faces: int) -> int:
    if num_faces < 20000:
        return 7
    elif num_faces < 150000:
        return 8
    else:
        return 9
