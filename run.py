#!/usr/bin/env python3
"""
Mesh Repair & FLAME Reconstruction Service
Usage:
    python run.py input.fbx output.fbx --mode repair_only
    python run.py input.fbx output.fbx --mode flame_fit --flame-model models/generic_model.pkl
    python run.py input.fbx output.fbx --mode flame_blend --flame-model models/generic_model.pkl
"""
import argparse
from flame_repair.pipeline import MeshRepairPipeline


def main():
    parser = argparse.ArgumentParser(description="Mesh Repair & FLAME Reconstruction")
    parser.add_argument("input", help="Input mesh file (FBX/OBJ/PLY/STL)")
    parser.add_argument("output", help="Output mesh file path")
    parser.add_argument("--mode", choices=["repair_only", "flame_fit", "flame_blend"], default="repair_only",
                        help="repair_only: basic repair; flame_fit: FLAME fitting; flame_blend: blend original+FLAME")
    parser.add_argument("--flame-model", default=None, help="Path to FLAME generic_model.pkl")
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps")
    parser.add_argument("--smooth-iters", type=int, default=2, help="Laplacian smoothing iterations")
    parser.add_argument("--target-faces", type=int, default=None, help="Target face count for remeshing")
    parser.add_argument("--flame-iters", type=int, default=500, help="FLAME fitting iterations")
    args = parser.parse_args()

    pipeline = MeshRepairPipeline(flame_model_path=args.flame_model, device=args.device)
    pipeline.run(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        smooth_iters=args.smooth_iters,
        target_faces=args.target_faces,
        flame_iters=args.flame_iters,
    )


if __name__ == "__main__":
    main()
