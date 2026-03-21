#!/usr/bin/env python3
"""
Mesh Repair & Reconstruction Service
Usage:
    python run.py input.fbx output.fbx --mode repair_only
    python run.py input.fbx output.fbx --mode body_repair
    python run.py input.fbx output.fbx --mode body_repair_blend --smplx-model-dir models/smplx
    python run.py input.fbx output.fbx --mode flame_fit --flame-model models/generic_model.pkl
    python run.py input.fbx output.fbx --mode flame_blend --flame-model models/generic_model.pkl
"""
import argparse
from flame_repair.pipeline import MeshRepairPipeline


def main():
    parser = argparse.ArgumentParser(description="Mesh Repair & Reconstruction")
    parser.add_argument("input", help="Input mesh file (FBX/OBJ/PLY/STL)")
    parser.add_argument("output", help="Output mesh file path")
    parser.add_argument(
        "--mode",
        choices=["repair_only", "body_repair", "body_repair_blend", "flame_fit", "flame_blend"],
        default="body_repair",
        help=(
            "repair_only: basic geometric repair only; "
            "body_repair: Poisson reconstruction for whole-body completion; "
            "body_repair_blend: Poisson + SMPL-X prior blending (strongest); "
            "flame_fit: FLAME head fitting; "
            "flame_blend: blend original + FLAME"
        ),
    )
    parser.add_argument("--flame-model", default=None, help="Path to FLAME generic_model.pkl")
    parser.add_argument("--smplx-model-dir", default=None,
                        help="Directory containing SMPLX_NEUTRAL.npz (download from https://smpl-x.is.tue.mpg.de/)")
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps")
    parser.add_argument("--smooth-iters", type=int, default=2, help="Laplacian smoothing iterations")
    parser.add_argument("--target-faces", type=int, default=None, help="Target face count for remeshing")
    parser.add_argument("--flame-iters", type=int, default=500, help="FLAME fitting iterations")
    parser.add_argument("--body-iters", type=int, default=300, help="SMPL-X fitting iterations")
    parser.add_argument("--poisson-depth", type=int, default=None,
                        help="Poisson reconstruction depth (auto-selected if not set: 7/8/9 by mesh size)")
    parser.add_argument("--no-intermediates", action="store_true",
                        help="Do not save intermediate OBJ files")
    args = parser.parse_args()

    pipeline = MeshRepairPipeline(
        flame_model_path=args.flame_model,
        smplx_model_dir=args.smplx_model_dir,
        device=args.device,
    )
    pipeline.run(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        smooth_iters=args.smooth_iters,
        target_faces=args.target_faces,
        flame_iters=args.flame_iters,
        body_iters=args.body_iters,
        poisson_depth=args.poisson_depth,
        save_intermediates=not args.no_intermediates,
    )


if __name__ == "__main__":
    main()
