import trimesh
from pathlib import Path
from .mesh_io import load_mesh, save_mesh
from .mesh_repair import (
    repair_mesh, smooth_mesh, remesh_uniform,
    detect_holes, classify_head_holes,
    fill_body_holes, fill_head_holes_with_flame,
)
from .flame_fitter import FLAMEFitter


class MeshRepairPipeline:

    def __init__(self, flame_model_path: str = None, device: str = "auto"):
        self.flame_model_path = flame_model_path
        self.device = device
        self.fitter = None

    def _ensure_fitter(self):
        if self.fitter is None:
            if self.flame_model_path is None:
                raise ValueError(
                    "FLAME model path required. Download from https://flame.is.tue.mpg.de/ "
                    "and provide path to generic_model.pkl"
                )
            self.fitter = FLAMEFitter(self.flame_model_path, device=self.device)

    def run(
        self,
        input_path: str,
        output_path: str,
        mode: str = "repair_only",
        smooth_iters: int = 2,
        target_faces: int = None,
        flame_iters: int = 500,
    ) -> trimesh.Trimesh:
        print(f"\n{'='*60}")
        print(f"  Mesh Repair Pipeline")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Mode:   {mode}")
        print(f"{'='*60}\n")

        mesh = load_mesh(input_path)

        if mode == "full_body_repair":
            return self._run_full_body_repair(mesh, output_path, smooth_iters)

        print("\n[Step 1] Basic Mesh Repair")
        mesh = repair_mesh(mesh)

        if smooth_iters > 0:
            print(f"\n[Step 2] Laplacian Smoothing (iterations={smooth_iters})")
            mesh = smooth_mesh(mesh, iterations=smooth_iters)

        if target_faces:
            print(f"\n[Step 3] Remeshing (target_faces={target_faces})")
            mesh = remesh_uniform(mesh, target_faces=target_faces)

        if mode == "flame_fit":
            print("\n[Step 4] FLAME Fitting & Reconstruction")
            self._ensure_fitter()
            result = self.fitter.fit(mesh, n_iters=flame_iters)
            mesh = result["mesh"]
            mesh = repair_mesh(mesh, verbose=False)

        elif mode == "flame_blend":
            print("\n[Step 4] FLAME Fitting & Blending")
            self._ensure_fitter()
            result = self.fitter.fit(mesh, n_iters=flame_iters)
            flame_mesh = result["mesh"]
            mesh = _blend_meshes(mesh, flame_mesh, alpha=0.3)

        save_mesh(mesh, output_path)
        print(f"\n[Done] Output saved to {output_path}")
        return mesh

    def _run_full_body_repair(
        self, mesh: trimesh.Trimesh, output_path: str, smooth_iters: int
    ) -> trimesh.Trimesh:
        print("\n[Step 1] Initial cleanup")
        mesh = repair_mesh(mesh)

        print("\n[Step 2] Hole detection")
        holes = detect_holes(mesh)

        if not holes:
            print("[Info] No holes found, mesh is already closed")
            save_mesh(mesh, output_path)
            return mesh

        print("\n[Step 3] Classify holes by location")
        head_holes, body_holes = classify_head_holes(mesh, holes)

        if head_holes and self.flame_model_path:
            print("\n[Step 4a] FLAME-guided head hole repair")
            mesh = fill_head_holes_with_flame(
                mesh, head_holes, self.flame_model_path, device=self.device
            )
        elif head_holes:
            print("\n[Step 4a] Geometric head hole fill (no FLAME model provided)")
            mesh = fill_body_holes(mesh, head_holes)

        if body_holes:
            print("\n[Step 4b] Geometric body hole repair")
            mesh = fill_body_holes(mesh, body_holes)

        print("\n[Step 5] Final repair pass")
        mesh = repair_mesh(mesh, verbose=True)

        if smooth_iters > 0:
            print(f"\n[Step 6] Smoothing (iterations={smooth_iters})")
            mesh = smooth_mesh(mesh, iterations=smooth_iters)

        if not mesh.is_watertight:
            print("[Warning] Mesh is still not fully watertight after repair")
            trimesh.repair.fill_holes(mesh)

        save_mesh(mesh, output_path)
        print(f"\n[Done] Output saved to {output_path}")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Watertight: {mesh.is_watertight}")
        return mesh


def _blend_meshes(original: trimesh.Trimesh, flame: trimesh.Trimesh, alpha: float = 0.3) -> trimesh.Trimesh:
    from scipy.spatial import cKDTree
    import numpy as np

    tree = cKDTree(flame.vertices)
    dists, indices = tree.query(original.vertices)
    median_dist = np.median(dists)

    weights = np.exp(-dists / (median_dist + 1e-8))
    weights = weights * alpha
    weights = weights.reshape(-1, 1)

    blended_verts = original.vertices * (1 - weights) + flame.vertices[indices] * weights
    result = trimesh.Trimesh(vertices=blended_verts, faces=original.faces, process=False)
    result.fix_normals()
    return result
