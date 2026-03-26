import trimesh
import numpy as np
from typing import Optional
from collections import Counter, defaultdict


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


def detect_holes(mesh: trimesh.Trimesh) -> list:
    """Find boundary loops (holes) in the mesh.
    Returns list of dicts: {'vertices': array of vertex indices, 'edge_count': int, 'centroid': np.array}
    """
    edges = mesh.edges_sorted
    edge_tuples = [tuple(e) for e in edges]
    edge_counts = Counter(edge_tuples)
    boundary_edges = np.array([e for e, c in edge_counts.items() if c == 1])

    if len(boundary_edges) == 0:
        print("[Repair] No holes detected - mesh is closed")
        return []

    adj = defaultdict(set)
    for e in boundary_edges:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])

    visited = set()
    loops = []
    for start in adj:
        if start in visited:
            continue
        loop = []
        current = start
        while current not in visited:
            visited.add(current)
            loop.append(current)
            neighbors = adj[current] - visited
            if not neighbors:
                break
            current = neighbors.pop()
        if len(loop) >= 3:
            verts = np.array(loop)
            centroid = mesh.vertices[verts].mean(axis=0)
            loops.append({
                'vertices': verts,
                'edge_count': len(verts),
                'centroid': centroid,
            })

    loops.sort(key=lambda h: h['edge_count'], reverse=True)
    print(f"[Repair] Detected {len(loops)} hole(s): {[h['edge_count'] for h in loops]} edges")
    return loops


def classify_head_holes(mesh: trimesh.Trimesh, holes: list, head_fraction: float = 0.15) -> tuple:
    """Split holes into head_holes and body_holes based on vertical position.
    Head region = top `head_fraction` of the bounding box height.
    """
    if not holes:
        return [], []

    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    extents = bbox_max - bbox_min
    up_axis = np.argmax(extents)
    head_threshold = bbox_max[up_axis] - extents[up_axis] * head_fraction

    head_holes = []
    body_holes = []
    for hole in holes:
        hole_verts = mesh.vertices[hole['vertices']]
        mean_up = hole_verts[:, up_axis].mean()
        if mean_up >= head_threshold:
            head_holes.append(hole)
        else:
            body_holes.append(hole)

    print(f"[Repair] Head holes: {len(head_holes)}, Body holes: {len(body_holes)}")
    return head_holes, body_holes


def fill_hole_fan(mesh: trimesh.Trimesh, hole: dict) -> trimesh.Trimesh:
    """Fill a hole using fan triangulation from the centroid."""
    verts = hole['vertices']
    centroid = hole['centroid']

    new_vert_idx = len(mesh.vertices)
    new_vertices = np.vstack([mesh.vertices, centroid.reshape(1, 3)])

    new_faces = []
    for i in range(len(verts)):
        v0 = verts[i]
        v1 = verts[(i + 1) % len(verts)]
        new_faces.append([v0, v1, new_vert_idx])

    all_faces = np.vstack([mesh.faces, np.array(new_faces)])
    result = trimesh.Trimesh(vertices=new_vertices, faces=all_faces, process=False)
    return result


def fill_body_holes(mesh: trimesh.Trimesh, holes: list) -> trimesh.Trimesh:
    """Fill all body holes using fan triangulation."""
    for i, hole in enumerate(holes):
        print(f"[Repair] Filling body hole {i+1}/{len(holes)} ({hole['edge_count']} edges)")
        mesh = fill_hole_fan(mesh, hole)
    return mesh


def fill_head_hole_with_flame(
    mesh: trimesh.Trimesh,
    hole: dict,
    flame_mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Fill a head hole using FLAME surface as reference.

    Strategy:
    1. Find FLAME vertices that fall inside the hole boundary projection
    2. Stitch FLAME patch vertices to the hole boundary via Delaunay
    3. Gives anatomically correct scalp curvature
    """
    from scipy.spatial import cKDTree, Delaunay

    boundary_verts = hole['vertices']
    boundary_pos = mesh.vertices[boundary_verts]

    flame_tree = cKDTree(flame_mesh.vertices)
    boundary_centroid = boundary_pos.mean(axis=0)
    boundary_radius = np.linalg.norm(boundary_pos - boundary_centroid, axis=1).max()

    candidates = flame_tree.query_ball_point(boundary_centroid, boundary_radius * 1.2)
    if len(candidates) < 3:
        print(f"[Repair] Not enough FLAME vertices near hole, falling back to fan fill")
        return fill_hole_fan(mesh, hole)

    flame_patch_verts = flame_mesh.vertices[candidates]

    # Keep FLAME vertices that are in the gap (far from existing mesh surface)
    mesh_tree = cKDTree(mesh.vertices)
    dists_to_mesh, _ = mesh_tree.query(flame_patch_verts)
    median_edge_len = np.median(mesh.edges_unique_length)
    inside_mask = dists_to_mesh > median_edge_len * 0.5
    interior_verts = flame_patch_verts[inside_mask]

    if len(interior_verts) < 1:
        print(f"[Repair] No interior FLAME vertices found, using fan fill")
        return fill_hole_fan(mesh, hole)

    # Add interior FLAME vertices to mesh
    new_start_idx = len(mesh.vertices)
    all_verts = np.vstack([mesh.vertices, interior_verts])

    # Combine boundary + new interior vertices for triangulation
    patch_indices = np.arange(new_start_idx, new_start_idx + len(interior_verts))
    all_patch_indices = np.concatenate([boundary_verts, patch_indices])
    all_patch_pos = all_verts[all_patch_indices]

    # Triangulate using Delaunay on 2D PCA projection
    centered = all_patch_pos - all_patch_pos.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_2d = centered @ Vt[:2].T

    tri = Delaunay(proj_2d)
    new_faces = all_patch_indices[tri.simplices]

    all_faces = np.vstack([mesh.faces, new_faces])
    result = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)

    print(f"[Repair] Filled head hole with {len(interior_verts)} FLAME vertices + {len(new_faces)} faces")
    return result


def fill_head_holes_with_flame(
    mesh: trimesh.Trimesh,
    holes: list,
    flame_model_path: str,
    device: str = "auto",
) -> trimesh.Trimesh:
    """Fit FLAME to head region, then fill head holes using FLAME surface."""
    from .flame_fitter import FLAMEFitter

    bbox_max = mesh.vertices.max(axis=0)
    bbox_min = mesh.vertices.min(axis=0)
    extents = bbox_max - bbox_min
    up_axis = np.argmax(extents)
    head_threshold = bbox_max[up_axis] - extents[up_axis] * 0.25

    head_mask = mesh.vertices[:, up_axis] > head_threshold
    head_face_mask = head_mask[mesh.faces].any(axis=1)
    head_mesh = mesh.submesh([head_face_mask], append=True)

    print(f"[Repair] Head region: {len(head_mesh.vertices)} vertices, {len(head_mesh.faces)} faces")

    fitter = FLAMEFitter(flame_model_path, device=device, n_shape=100, n_exp=50)
    result = fitter.fit(head_mesh, n_iters=300)
    flame_mesh = result["mesh"]

    print(f"[Repair] FLAME fitted: {len(flame_mesh.vertices)} vertices")

    for i, hole in enumerate(holes):
        print(f"[Repair] Filling head hole {i+1}/{len(holes)} ({hole['edge_count']} edges)")
        mesh = fill_head_hole_with_flame(mesh, hole, flame_mesh)

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
