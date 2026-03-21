import trimesh
import numpy as np
from pathlib import Path
from .mesh_io import load_mesh, save_mesh
from .mesh_repair import repair_mesh, smooth_mesh, remesh_uniform, reconstruct_body, blend_reconstruction, auto_poisson_depth
from .flame_fitter import FLAMEFitter
from .hole_analysis import estimate_damage


class MeshRepairPipeline:

    def __init__(self, flame_model_path: str = None, smplx_model_dir: str = None, device: str = "auto"):
        self.flame_model_path = flame_model_path
        self.smplx_model_dir = smplx_model_dir
        self.device = device
        self.fitter = None
        self.smplx_fitter = None

    def _ensure_flame_fitter(self):
        if self.fitter is None:
            if self.flame_model_path is None:
                raise ValueError(
                    "FLAME model path required. Download from https://flame.is.tue.mpg.de/ "
                    "and provide path to generic_model.pkl"
                )
            self.fitter = FLAMEFitter(self.flame_model_path, device=self.device)

    def _ensure_smplx_fitter(self):
        if self.smplx_fitter is None:
            if self.smplx_model_dir is None:
                raise ValueError(
                    "SMPL-X model directory required. Download from https://smpl-x.is.tue.mpg.de/ "
                    "and provide directory containing SMPLX_NEUTRAL.npz"
                )
            from .smplx_fitter import SMPLXFitter
            self.smplx_fitter = SMPLXFitter(self.smplx_model_dir, device=self.device)

    def run(
        self,
        input_path: str,
        output_path: str,
        mode: str = "repair_only",
        smooth_iters: int = 2,
        target_faces: int = None,
        flame_iters: int = 500,
        body_iters: int = 300,
        poisson_depth: int = None,
        save_intermediates: bool = True,
    ) -> trimesh.Trimesh:
        print(f"\n{'='*60}")
        print("  Mesh Repair Pipeline")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Mode:   {mode}")
        print(f"{'='*60}\n")

        out_dir = Path(output_path).parent
        out_stem = Path(output_path).stem

        mesh = load_mesh(input_path)

        print("\n[Step 1] Basic Mesh Repair")
        mesh = repair_mesh(mesh)

        if save_intermediates and mode.startswith("body_"):
            _save_intermediate(mesh, out_dir / f"{out_stem}_step1_repaired.obj")

        if smooth_iters > 0:
            print(f"\n[Step 2] Laplacian Smoothing (iterations={smooth_iters})")
            mesh = smooth_mesh(mesh, iterations=smooth_iters)

        if target_faces:
            print(f"\n[Step 3] Remeshing (target_faces={target_faces})")
            mesh = remesh_uniform(mesh, target_faces=target_faces)

        if mode == "flame_fit":
            print("\n[Step 4] FLAME Fitting & Reconstruction")
            self._ensure_flame_fitter()
            result = self.fitter.fit(mesh, n_iters=flame_iters)
            mesh = result["mesh"]
            mesh = repair_mesh(mesh, verbose=False)

        elif mode == "flame_blend":
            print("\n[Step 4] FLAME Fitting & Local Repair Blending")
            self._ensure_flame_fitter()
            result = self.fitter.fit(mesh, n_iters=flame_iters)
            flame_mesh = result["mesh"]
            mesh = _blend_meshes(mesh, flame_mesh)

        elif mode == "body_repair":
            print("\n[Step 4] Body Mesh Reconstruction (Poisson)")
            damage = estimate_damage(mesh)
            print(f"[Damage] score={damage['damage_score']:.3f}, holes={damage['hole_count']}, "
                  f"boundary_edges={damage['boundary_edge_count']}, watertight={damage['is_watertight']}")

            depth = poisson_depth or auto_poisson_depth(len(mesh.faces))
            reconstructed = _poisson_with_fallback(mesh, depth)

            if save_intermediates:
                _save_intermediate(reconstructed, out_dir / f"{out_stem}_step2_reconstructed.obj")

            mesh = blend_reconstruction(mesh, reconstructed)

        elif mode == "body_repair_blend":
            print("\n[Step 4] Body Mesh Reconstruction (Poisson) + SMPL-X Blending")
            damage = estimate_damage(mesh)
            print(f"[Damage] score={damage['damage_score']:.3f}, holes={damage['hole_count']}, "
                  f"boundary_edges={damage['boundary_edge_count']}, watertight={damage['is_watertight']}")

            depth = poisson_depth or auto_poisson_depth(len(mesh.faces))
            reconstructed = _poisson_with_fallback(mesh, depth)

            if save_intermediates:
                _save_intermediate(reconstructed, out_dir / f"{out_stem}_step2_reconstructed.obj")

            print("\n[Step 5] SMPL-X Fitting & Blending")
            self._ensure_smplx_fitter()
            smplx_result = self.smplx_fitter.fit(reconstructed, n_iters=body_iters)
            smplx_mesh = smplx_result["mesh"]

            if save_intermediates:
                _save_intermediate(smplx_mesh, out_dir / f"{out_stem}_step3_smplx.obj")

            mesh = _blend_meshes_body(mesh, reconstructed, smplx_mesh)

        save_mesh(mesh, output_path)
        print(f"\n[Done] Output saved to {output_path}")
        return mesh


def _blend_meshes(original: trimesh.Trimesh, flame: trimesh.Trimesh) -> trimesh.Trimesh:
    from scipy.spatial import cKDTree

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
    print(f"[Blend] Preserving original detail where supported; "
          f"using FLAME prior on sparse/damaged regions ({repaired_ratio:.1%} repaired)")

    result = trimesh.Trimesh(vertices=blended_verts, faces=flame.faces, process=False)
    result.fix_normals()
    return result


def _blend_meshes_body(
    original: trimesh.Trimesh,
    reconstructed: trimesh.Trimesh,
    smplx: trimesh.Trimesh,
) -> trimesh.Trimesh:
    from scipy.spatial import cKDTree

    orig_verts = np.asarray(original.vertices)
    smplx_verts = np.asarray(smplx.vertices)

    bbox_diag = float(np.linalg.norm(orig_verts.max(axis=0) - orig_verts.min(axis=0)))
    threshold = bbox_diag * 0.02

    tree_orig = cKDTree(orig_verts)
    dists_smplx, idx_smplx = tree_orig.query(smplx_verts)

    support = np.exp(-((dists_smplx / threshold) ** 2)).reshape(-1, 1)
    matched_orig = orig_verts[idx_smplx]
    blended_verts = smplx_verts * (1.0 - support) + matched_orig * support

    repaired_ratio = float(np.mean(dists_smplx > threshold))
    print(f"[Blend] SMPL-X blend: original detail on {1-repaired_ratio:.1%}, SMPL-X prior on {repaired_ratio:.1%}")

    result = trimesh.Trimesh(vertices=blended_verts, faces=smplx.faces, process=False)
    result.fix_normals()
    return result


def _poisson_with_fallback(mesh: trimesh.Trimesh, depth: int) -> trimesh.Trimesh:
    for d in [depth, max(depth - 1, 6), max(depth - 2, 6)]:
        try:
            result = reconstruct_body(mesh, poisson_depth=d)
            if len(result.vertices) > 100:
                return result
        except Exception as e:
            print(f"[Reconstruct] depth={d} failed: {e}, trying lower depth...")
    print("[Reconstruct] All Poisson depths failed, returning original mesh")
    return mesh


def _save_intermediate(mesh: trimesh.Trimesh, path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(path), file_type="obj")
        print(f"[IO] Intermediate saved: {path}")
    except Exception as e:
        print(f"[IO] Could not save intermediate {path}: {e}")
