import trimesh
import numpy as np
from typing import List, Dict


def detect_holes(mesh: trimesh.Trimesh) -> List[List[int]]:
    edges = mesh.edges_unique
    edge_count = mesh.edges_unique_inverse
    face_edges = mesh.faces_unique_edges

    boundary_mask = np.zeros(len(edges), dtype=bool)
    for i, count in enumerate(np.bincount(mesh.edges_unique_inverse, minlength=len(edges))):
        if count == 1:
            boundary_mask[i] = True

    boundary_edge_indices = np.where(boundary_mask)[0]
    if len(boundary_edge_indices) == 0:
        return []

    boundary_edges = edges[boundary_edge_indices]

    adj = {}
    for e in boundary_edges:
        v0, v1 = int(e[0]), int(e[1])
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    visited = set()
    rings = []

    for start in adj:
        if start in visited:
            continue
        ring = []
        stack = [start]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            ring.append(v)
            for nb in adj.get(v, []):
                if nb not in visited:
                    stack.append(nb)
        if len(ring) >= 3:
            rings.append(ring)

    return rings


def estimate_damage(mesh: trimesh.Trimesh) -> Dict:
    holes = detect_holes(mesh)
    boundary_edge_count = sum(len(h) for h in holes)
    total_area = float(mesh.area) if mesh.area > 0 else 1.0

    largest_hole_area = 0.0
    if holes:
        for ring in holes:
            verts = mesh.vertices[ring]
            center = verts.mean(axis=0)
            dists = np.linalg.norm(verts - center, axis=1)
            approx_radius = float(np.mean(dists))
            approx_area = np.pi * approx_radius ** 2
            if approx_area > largest_hole_area:
                largest_hole_area = approx_area

    largest_hole_ratio = min(largest_hole_area / total_area, 1.0)

    components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1)
    num_components = len(list(components))

    boundary_ratio = min(boundary_edge_count / max(len(mesh.edges_unique), 1), 1.0)

    damage_score = float(np.clip(
        0.4 * largest_hole_ratio
        + 0.3 * boundary_ratio
        + 0.2 * min((num_components - 1) / 10.0, 1.0)
        + 0.1 * (0.0 if mesh.is_watertight else 1.0),
        0.0, 1.0
    ))

    return {
        "is_watertight": bool(mesh.is_watertight),
        "hole_count": len(holes),
        "boundary_edge_count": boundary_edge_count,
        "largest_hole_ratio": round(largest_hole_ratio, 4),
        "num_components": num_components,
        "damage_score": round(damage_score, 4),
    }
