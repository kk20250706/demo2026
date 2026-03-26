import trimesh
import numpy as np
import subprocess
import tempfile
from pathlib import Path


def _convert_fbx_to_obj(fbx_path: Path) -> Path:
    obj_path = fbx_path.with_suffix(".obj")
    # Compatible with both old Blender (2.x/3.0-3.2) and new (3.3+)
    script = (
        "import bpy\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        f"bpy.ops.import_scene.fbx(filepath='{fbx_path}')\n"
        "try:\n"
        f"    bpy.ops.wm.obj_export(filepath='{obj_path}')\n"
        "except AttributeError:\n"
        f"    bpy.ops.export_scene.obj(filepath='{obj_path}')\n"
    )
    print(f"[IO] Converting FBX -> OBJ via Blender...")
    result = subprocess.run(
        ["blender", "--background", "--python-expr", script],
        capture_output=True, text=True, timeout=120,
    )
    if not obj_path.exists():
        print(f"[IO] Blender stdout: {result.stdout[-500:]}")
        print(f"[IO] Blender stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"Blender FBX conversion failed")
    print(f"[IO] Converted to {obj_path}")
    return obj_path


def _convert_obj_to_fbx(obj_path: Path, fbx_path: Path):
    # Compatible with both old Blender (2.x/3.0-3.2) and new (3.3+)
    script = (
        "import bpy\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        "try:\n"
        f"    bpy.ops.wm.obj_import(filepath='{obj_path}')\n"
        "except AttributeError:\n"
        f"    bpy.ops.import_scene.obj(filepath='{obj_path}')\n"
        f"bpy.ops.export_scene.fbx(filepath='{fbx_path}')\n"
    )
    print(f"[IO] Converting OBJ -> FBX via Blender...")
    result = subprocess.run(
        ["blender", "--background", "--python-expr", script],
        capture_output=True, text=True, timeout=120,
    )
    if not fbx_path.exists():
        print(f"[IO] Blender stdout: {result.stdout[-500:]}")
        print(f"[IO] Blender stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"Blender FBX export failed")
    print(f"[IO] Exported to {fbx_path}")


def load_mesh(filepath: str) -> trimesh.Trimesh:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    load_path = filepath
    if filepath.suffix.lower() == ".fbx":
        load_path = _convert_fbx_to_obj(filepath)

    scene_or_mesh = trimesh.load(str(load_path), force="mesh")
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No triangle mesh found in file")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    print(f"[IO] Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def save_mesh(mesh: trimesh.Trimesh, filepath: str, file_format: str = None):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    suffix = filepath.suffix.lower()
    if file_format:
        suffix = f".{file_format}"

    if suffix == ".fbx":
        obj_path = filepath.with_suffix(".obj")
        mesh.export(str(obj_path), file_type="obj")
        _convert_obj_to_fbx(obj_path, filepath)
        obj_path.unlink(missing_ok=True)
    elif suffix in (".obj", ".ply", ".stl", ".glb", ".gltf"):
        mesh.export(str(filepath), file_type=suffix.lstrip("."))
    else:
        mesh.export(str(filepath))

    print(f"[IO] Saved mesh to {filepath}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
