#!/bin/bash
# ============================================
#  Local M3 Mac Runner (MPS GPU acceleration)
# ============================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[Setup] Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install numpy scipy trimesh open3d pyrender tqdm chumpy
    echo "[Setup] Done."
else
    source "$VENV_DIR/bin/activate"
fi

echo ""
echo "============================================"
echo "  FLAME Mesh Repair Service (M3 Mac / MPS)"
echo "============================================"
echo ""
echo "PyTorch MPS available: $(python3 -c 'import torch; print(torch.backends.mps.is_available())')"
echo ""

INPUT="${1:-input/test.fbx}"
OUTPUT="${2:-output/repaired.fbx}"
MODE="${3:-repair_only}"
FLAME_MODEL="${4:-models/generic_model.pkl}"

if [[ "$MODE" == "repair_only" ]]; then
    python3 "$SCRIPT_DIR/run.py" "$INPUT" "$OUTPUT" --mode repair_only --device mps
else
    python3 "$SCRIPT_DIR/run.py" "$INPUT" "$OUTPUT" --mode "$MODE" --flame-model "$FLAME_MODEL" --device mps
fi
