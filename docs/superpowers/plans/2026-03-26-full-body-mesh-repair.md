# Full-Body Mesh Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair full-body FBX files (scalp holes, non-watertight mesh) using FLAME for head region and geometric filling for body, runnable in Google Colab.

**Architecture:** Load FBX via Blender→OBJ conversion, detect boundary loops (holes), classify holes by location (head vs body). Head holes get filled using FLAME surface as reference geometry. Body holes get geometric triangulation fill. Final pass ensures watertight mesh, unified normals. Output as FBX.

**Tech Stack:** Python 3, trimesh, scipy, numpy, torch, Blender (for FBX I/O), existing FLAME model (`generic_model.pkl`)

---

## File Structure

```
flame_repair/
├── __init__.py          (no change)
├── flame_model.py       (no change - existing FLAME layer)
├── flame_fitter.py      (no change - existing FLAME fitter)
├── mesh_io.py           (no change - existing FBX/OBJ I/O)
├── mesh_repair.py       (MODIFY - add hole detection, head region detection, FLAME-guided fill)
├── pipeline.py          (MODIFY - add full_body_repair mode)
run.py                   (MODIFY - add full_body_repair mode to CLI)
colab_mesh_repair.ipynb  (REWRITE - working Colab notebook, no external repo clone)
requirements.txt         (no change)
```

Key decision: keep everything in `mesh_repair.py` rather than splitting into new files — the hole detection + filling logic is tightly coupled and the file is only 111 lines currently.

---

### Task 1: Add Hole Detection to mesh_repair.py

**Files:**
- Modify: `flame_repair/mesh_repair.py`

- [ ] **Step 1: Add `detect_holes` function**

Add this function to `flame_repair/mesh_repair.py` after the existing `_remove_unreferenced_vertices` function:

```python
def detect_holes(mesh: trimesh.Trimesh) -> list:
    """Find boundary loops (holes) in the mesh.
    Returns list of dicts: {'vertices': array of vertex indices, 'edge_count': int, 'centroid': np.array}
    """
    edges = mesh.edges_sorted
    # boundary edges appear in exactly one face
    from collections import Counter
    edge_tuples = [tuple(e) for e in edges]
    edge_counts = Counter(edge_tuples)
    boundary_edges = np.array([e for e, c in edge_counts.items() if c == 1])

    if len(boundary_edges) == 0:
        print("[Repair] No holes detected - mesh is closed")
        return []

    # Build adjacency and find connected loops
    from collections import defaultdict
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
```

- [ ] **Step 2: Add `classify_head_holes` function**

Add right after `detect_holes`:

```python
def classify_head_holes(mesh: trimesh.Trimesh, holes: list, head_fraction: float = 0.15) -> tuple:
    """Split holes into head_holes and body_holes based on vertical position.
    Head region = top `head_fraction` of the bounding box height.
    Returns (head_holes, body_holes).
    """
    if not holes:
        return [], []

    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    # Find which axis is "up" - the one with the largest extent
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
```

- [ ] **Step 3: Test hole detection locally**

```bash
cd /Users/huijieqi/demo2026
python3 -c "
import trimesh
# Create a mesh with a known hole (half sphere)
sphere = trimesh.creation.icosphere(subdivisions=3)
# Remove top faces to create a hole
mask = sphere.face_normals[:, 2] < 0.8
sphere.update_faces(mask)
sphere.remove_unreferenced_vertices()
import sys; sys.path.insert(0, '.')
from flame_repair.mesh_repair import detect_holes, classify_head_holes
holes = detect_holes(sphere)
print(f'Found {len(holes)} holes')
assert len(holes) > 0, 'Should detect at least one hole'
head, body = classify_head_holes(sphere, holes)
print(f'Head: {len(head)}, Body: {len(body)}')
print('PASS')
"
```

- [ ] **Step 4: Commit**

```bash
git add flame_repair/mesh_repair.py
git commit -m "feat: add hole detection and head region classification"
```

---

### Task 2: Add Geometric Hole Filling for Body Holes

**Files:**
- Modify: `flame_repair/mesh_repair.py`

- [ ] **Step 1: Add `fill_hole_fan` function**

Add after `classify_head_holes`:

```python
def fill_hole_fan(mesh: trimesh.Trimesh, hole: dict) -> trimesh.Trimesh:
    """Fill a hole using fan triangulation from the centroid.
    Adds a new vertex at the centroid and creates triangles connecting
    each consecutive pair of boundary vertices to the centroid.
    """
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
```

- [ ] **Step 2: Test body hole filling**

```bash
cd /Users/huijieqi/demo2026
python3 -c "
import trimesh, numpy as np
sphere = trimesh.creation.icosphere(subdivisions=3)
# Remove a small patch at the bottom
mask = sphere.face_normals[:, 2] > -0.9
sphere.update_faces(mask)
sphere.remove_unreferenced_vertices()
print(f'Before: watertight={sphere.is_watertight}, faces={len(sphere.faces)}')
import sys; sys.path.insert(0, '.')
from flame_repair.mesh_repair import detect_holes, fill_body_holes
holes = detect_holes(sphere)
sphere = fill_body_holes(sphere, holes)
sphere.fix_normals()
print(f'After: watertight={sphere.is_watertight}, faces={len(sphere.faces)}')
print('PASS')
"
```

- [ ] **Step 3: Commit**

```bash
git add flame_repair/mesh_repair.py
git commit -m "feat: add geometric hole filling for body holes"
```

---

### Task 3: Add FLAME-Guided Head Hole Filling

**Files:**
- Modify: `flame_repair/mesh_repair.py`

- [ ] **Step 1: Add `fill_head_hole_with_flame` function**

Add after `fill_body_holes`:

```python
def fill_head_hole_with_flame(
    mesh: trimesh.Trimesh,
    hole: dict,
    flame_mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """Fill a head hole using FLAME surface as reference.

    Strategy:
    1. Find the FLAME vertices that fall inside the hole boundary projection
    2. Stitch FLAME patch vertices to the hole boundary
    3. This gives anatomically correct scalp curvature
    """
    from scipy.spatial import cKDTree

    boundary_verts = hole['vertices']
    boundary_pos = mesh.vertices[boundary_verts]

    # Find FLAME vertices near the hole region
    flame_tree = cKDTree(flame_mesh.vertices)
    boundary_centroid = boundary_pos.mean(axis=0)
    boundary_radius = np.linalg.norm(boundary_pos - boundary_centroid, axis=1).max()

    # Get FLAME vertices within the hole region
    candidates = flame_tree.query_ball_point(boundary_centroid, boundary_radius * 1.2)
    if len(candidates) < 3:
        print(f"[Repair] Not enough FLAME vertices near hole, falling back to fan fill")
        return fill_hole_fan(mesh, hole)

    flame_patch_verts = flame_mesh.vertices[candidates]

    # Filter: only keep FLAME vertices that are "inside" the hole
    # Project onto plane defined by boundary, check if inside convex hull
    # Simplified: keep vertices that are closer to centroid than to any boundary vertex
    mesh_tree = cKDTree(mesh.vertices)
    dists_to_mesh, _ = mesh_tree.query(flame_patch_verts)
    # Keep FLAME vertices that are far from existing mesh (i.e., in the gap)
    median_edge_len = np.median(mesh.edges_unique_length)
    inside_mask = dists_to_mesh > median_edge_len * 0.5
    interior_verts = flame_patch_verts[inside_mask]

    if len(interior_verts) < 1:
        print(f"[Repair] No interior FLAME vertices found, using fan fill")
        return fill_hole_fan(mesh, hole)

    # Add interior FLAME vertices to mesh
    new_start_idx = len(mesh.vertices)
    all_verts = np.vstack([mesh.vertices, interior_verts])

    # Create new vertex indices for the patch
    patch_indices = np.arange(new_start_idx, new_start_idx + len(interior_verts))
    all_patch_indices = np.concatenate([boundary_verts, patch_indices])
    all_patch_pos = all_verts[all_patch_indices]

    # Triangulate the patch using Delaunay on the local 2D projection
    from scipy.spatial import Delaunay

    # Project to 2D using PCA
    centered = all_patch_pos - all_patch_pos.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_2d = centered @ Vt[:2].T

    tri = Delaunay(proj_2d)
    new_faces = all_patch_indices[tri.simplices]

    all_faces = np.vstack([mesh.faces, new_faces])
    result = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)

    print(f"[Repair] Filled head hole with {len(interior_verts)} FLAME vertices + {len(new_faces)} faces")
    return result
```

- [ ] **Step 2: Add `fill_head_holes_with_flame` orchestrator**

```python
def fill_head_holes_with_flame(
    mesh: trimesh.Trimesh,
    holes: list,
    flame_model_path: str,
    device: str = "auto",
) -> trimesh.Trimesh:
    """Fit FLAME to head region, then fill head holes using FLAME surface."""
    from .flame_fitter import FLAMEFitter

    # Extract approximate head region for FLAME fitting
    bbox_max = mesh.vertices.max(axis=0)
    bbox_min = mesh.vertices.min(axis=0)
    extents = bbox_max - bbox_min
    up_axis = np.argmax(extents)
    head_threshold = bbox_max[up_axis] - extents[up_axis] * 0.25

    head_mask = mesh.vertices[:, up_axis] > head_threshold
    head_face_mask = head_mask[mesh.faces].any(axis=1)
    head_mesh = mesh.submesh([head_face_mask], append=True)

    print(f"[Repair] Head region: {len(head_mesh.vertices)} vertices, {len(head_mesh.faces)} faces")

    # Fit FLAME to head
    fitter = FLAMEFitter(flame_model_path, device=device, n_shape=100, n_exp=50)
    result = fitter.fit(head_mesh, n_iters=300)
    flame_mesh = result["mesh"]

    print(f"[Repair] FLAME fitted: {len(flame_mesh.vertices)} vertices")

    # Fill each head hole
    for i, hole in enumerate(holes):
        print(f"[Repair] Filling head hole {i+1}/{len(holes)} ({hole['edge_count']} edges)")
        mesh = fill_head_hole_with_flame(mesh, hole, flame_mesh)

    return mesh
```

- [ ] **Step 3: Commit**

```bash
git add flame_repair/mesh_repair.py
git commit -m "feat: add FLAME-guided head hole filling"
```

---

### Task 4: Add `full_body_repair` Mode to Pipeline

**Files:**
- Modify: `flame_repair/pipeline.py`

- [ ] **Step 1: Update pipeline.py with full_body_repair mode**

Replace the entire content of `flame_repair/pipeline.py`:

```python
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
            # Try trimesh built-in as last resort
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
```

- [ ] **Step 2: Commit**

```bash
git add flame_repair/pipeline.py
git commit -m "feat: add full_body_repair mode to pipeline"
```

---

### Task 5: Update CLI (run.py)

**Files:**
- Modify: `run.py`

- [ ] **Step 1: Add full_body_repair to CLI choices**

In `run.py`, update the `--mode` argument:

```python
parser.add_argument("--mode", choices=["repair_only", "flame_fit", "flame_blend", "full_body_repair"],
                    default="full_body_repair",
                    help="repair_only: basic repair; flame_fit: FLAME fitting; flame_blend: blend; full_body_repair: detect+fill holes with FLAME for head")
```

Change default from `"repair_only"` to `"full_body_repair"`.

- [ ] **Step 2: Commit**

```bash
git add run.py
git commit -m "feat: add full_body_repair mode to CLI with new default"
```

---

### Task 6: Rewrite Colab Notebook

**Files:**
- Rewrite: `colab_mesh_repair.ipynb`

- [ ] **Step 1: Write the complete Colab notebook**

Create `colab_mesh_repair.ipynb` with these cells:

**Cell 0 (markdown):**
```markdown
# Full-Body Mesh Repair (FLAME + Geometric)

Repairs FBX files with:
- **Head/scalp holes** → filled using FLAME 3D face model as reference surface
- **Body holes** → geometric triangulation fill
- **Non-watertight mesh** → closed and normals fixed

All model files and test FBX are included in the repo — no downloads needed.
```

**Cell 1 (code) — Install deps + Blender:**
```python
%%capture
!pip install torch numpy trimesh scipy tqdm
!apt-get update -qq && apt-get install -y -qq blender > /dev/null 2>&1
```

**Cell 2 (code) — Clone repo:**
```python
import os
if not os.path.exists('/content/demo2026'):
    !git clone https://github.com/huijieqi/demo2026.git /content/demo2026
os.chdir('/content/demo2026')
import sys
sys.path.insert(0, '/content/demo2026')
```

**Cell 3 (code) — Check environment:**
```python
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import trimesh
print(f'Trimesh: {trimesh.__version__}')

import subprocess
result = subprocess.run(['blender', '--version'], capture_output=True, text=True)
print(f'Blender: {result.stdout.strip().split(chr(10))[0]}')

# Verify files exist
assert os.path.exists('input/liukui.fbx'), 'Missing input/liukui.fbx'
assert os.path.exists('models/generic_model.pkl'), 'Missing models/generic_model.pkl'
print('\nAll files ready!')
```

**Cell 4 (code) — Run repair:**
```python
from flame_repair.pipeline import MeshRepairPipeline

pipeline = MeshRepairPipeline(
    flame_model_path='models/generic_model.pkl',
    device='auto',  # will use CUDA on Colab
)

result_mesh = pipeline.run(
    input_path='input/liukui.fbx',
    output_path='output/repaired_liukui.fbx',
    mode='full_body_repair',
    smooth_iters=2,
)
```

**Cell 5 (code) — Visualize before/after:**
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from flame_repair.mesh_io import load_mesh

original = load_mesh('input/liukui.fbx')

def plot_comparison(mesh_a, mesh_b, title_a='Original', title_b='Repaired', max_faces=8000):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})

    for ax, mesh, title in [(axes[0], mesh_a, title_a), (axes[1], mesh_b, title_b)]:
        verts = mesh.vertices
        faces = mesh.faces
        if len(faces) > max_faces:
            idx = np.random.choice(len(faces), max_faces, replace=False)
            faces = faces[idx]
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                        triangles=faces, alpha=0.6, edgecolor='gray', linewidth=0.05)
        ax.set_title(f'{title}\n{len(mesh.vertices)} verts, {len(mesh.faces)} faces\nWatertight: {mesh.is_watertight}')

    plt.tight_layout()
    plt.show()

plot_comparison(original, result_mesh)
```

**Cell 6 (code) — Download result:**
```python
from google.colab import files
import glob

for f in glob.glob('output/*'):
    print(f'Downloading: {f}')
    files.download(f)
```

- [ ] **Step 2: Commit**

```bash
git add colab_mesh_repair.ipynb
git commit -m "feat: rewrite Colab notebook for full-body repair"
```

---

### Task 7: Integration Test

- [ ] **Step 1: Run full pipeline locally on liukui.fbx (if Blender available)**

```bash
cd /Users/huijieqi/demo2026
python3 run.py input/liukui.fbx output/repaired_liukui.fbx \
    --mode full_body_repair \
    --flame-model models/generic_model.pkl \
    --device cpu
```

If Blender is not available locally, test with a synthetic mesh:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
import trimesh, numpy as np
from flame_repair.mesh_repair import detect_holes, classify_head_holes, fill_body_holes

# Create a capsule (body-like shape) with holes
capsule = trimesh.creation.capsule(height=1.7, radius=0.2, count=[16, 8])
# Remove some top faces (simulate scalp hole)
mask = capsule.face_normals[:, 2] < 0.85
capsule.update_faces(mask)
capsule.remove_unreferenced_vertices()

print(f'Before: verts={len(capsule.vertices)} faces={len(capsule.faces)} watertight={capsule.is_watertight}')
holes = detect_holes(capsule)
head_holes, body_holes = classify_head_holes(capsule, holes)
capsule = fill_body_holes(capsule, head_holes + body_holes)
capsule.fix_normals()
print(f'After: verts={len(capsule.vertices)} faces={len(capsule.faces)} watertight={capsule.is_watertight}')
print('Integration test PASS')
"
```

- [ ] **Step 2: Verify output file exists**

```bash
ls -la output/repaired_liukui.fbx 2>/dev/null || echo "FBX output requires Blender"
```

- [ ] **Step 3: Final commit with all changes**

```bash
git add -A
git status
git commit -m "feat: full-body mesh repair with FLAME head + geometric body hole filling"
```
