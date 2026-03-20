import trimesh
from pathlib import Path
from .mesh_io import load_mesh, save_mesh
from .mesh_repair import repair_mesh, smooth_mesh, remesh_uniform
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
        print("  Mesh Repair Pipeline")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Mode:   {mode}")
        print(f"{'='*60}\n")

        mesh = load_mesh(input_path)

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
            print("\n[Step 4] FLAME Fitting & Local Repair Blending")
            self._ensure_fitter()
            result = self.fitter.fit(mesh, n_iters=flame_iters)
            flame_mesh = result["mesh"]
            mesh = _blend_meshes(mesh, flame_mesh)

        save_mesh(mesh, output_path)
        print(f"\n[Done] Output saved to {output_path}")
        return mesh


def _blend_meshes(original: trimesh.Trimesh, flame: trimesh.Trimesh) -> trimesh.Trimesh:
    from scipy.spatial import cKDTree
    import numpy as np

    original_vertices = np.asarray(original.vertices)
    flame_vertices = np.asarray(flame.vertices)

    tree = cKDTree(original_vertices)
    dists, indices = tree.query(flame_vertices)
    matched_original = original_vertices[indices]

    median_dist = float(np.median(dists))
    mad_dist = float(np.median(np.abs(dists - median_dist)))
    support_scale = max(median_dist + 2.5 * mad_dist, 1e-6)

    support = np.exp(-((dists / support_scale) ** 2)).reshape(-1, 1)
    blended_verts = flame_vertices * (1.0 - support) + matched_original * support

    repaired_ratio = float(np.mean(dists > support_scale))
    print(
        f"[Blend] Preserving original detail where supported; "
        f"using FLAME prior on sparse/damaged regions ({repaired_ratio:.1%} repaired)"
    )

    result = trimesh.Trimesh(vertices=blended_verts, faces=flame.faces, process=False)
    result.fix_normals()
    return result
