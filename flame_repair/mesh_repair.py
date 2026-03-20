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
