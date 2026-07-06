#!/usr/bin/env python
"""
PROBE — edge-adjacency separation of face 97 (top flat) vs face 93 (hole ring).

On 96260B_front the blind-hole pass annexes both as floors. Toolpath says:
  97 = standalone flat (depth 4.45 mm, +Y plane)
  93 = blind-hole defining ring (depth 7.83 mm, +Y plane)

Hypothesis (must PASS before changing hole_detection):
  97 borders the opening via CONVEX fillet/top edges and does NOT wrap bore-wall
  cylinders; it sits opening-ward of the shallowest curved bore face (torus 96).
  93 sits below the entry-fillet tier, is CONCAVE-bound to entry fillets, and
  shares edges with recess-wall cylinders.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_plane_93_97_edges.py --part 96260B_front
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from feature_params import analyze_step, load_step_faces
from hole_detection import FaceGraph, _axial_position, _unit
from run_cascade import _load_edges

try:
    from brep_extents import collect_boundary_points
except ImportError:
    collect_boundary_points = None  # type: ignore[misc, assignment]

MM_PER_IN = 25.4
WATCH = (93, 97)
BORE_STACK = {93, 94, 95, 96, 97, 285, 286, 287}
CURVED_TYPES = {"cylinder", "cone", "torus", "bspline", "bezier"}
Y = np.array([0.0, 1.0, 0.0])

PARTS: dict[str, dict[str, str]] = {
    "96260B_front": {
        "step": "96260B_FRONT_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_front/graph.npz",
    },
}


class _Axis:
    def __init__(self, point: np.ndarray, direction: np.ndarray):
        self.point = point
        self.direction = direction


def top_of_part_y(step_path: str, n_faces: int) -> float:
    if collect_boundary_points is not None:
        occ = load_step_faces(step_path)
        all_y: list[float] = []
        for i in range(n_faces):
            pts = collect_boundary_points(occ[i])
            all_y.extend(pts[:, 1].tolist())
        return max(all_y)
    faces = analyze_step(step_path)
    return max(float(f.centroid[1]) for f in faces)


def radial_dist_from_axis(centroid, axis_pt, axis_dir) -> float:
    c = np.asarray(centroid, dtype=float) - np.asarray(axis_pt, dtype=float)
    d = np.asarray(axis_dir, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    proj = np.dot(c, d) * d
    return float(np.linalg.norm(c - proj))


def edge_dihedral_deg(cos_val: float | None) -> float | None:
    if cos_val is None:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, float(cos_val)))))


def classify_neighbor(by_index: dict, nb: int) -> str:
    f = by_index[nb]
    tags: list[str] = [f.surface_type]
    if nb in BORE_STACK:
        tags.append("bore")
    if f.surface_type in ("cylinder", "cone"):
        tags.append("wall")
    if f.surface_type == "torus":
        tags.append("fillet")
    if f.surface_type == "plane" and f.normal is not None and abs(float(f.normal[1])) > 0.9:
        tags.append("+Y")
    return "/".join(tags)


def edge_signature(face_idx: int, graph: FaceGraph, by_index: dict) -> dict[str, int | bool]:
    sig: dict[str, int | bool] = {
        "n_cyl_concave": 0,
        "n_cyl_convex": 0,
        "n_cyl_smooth": 0,
        "n_fillet_concave": 0,
        "n_fillet_convex": 0,
        "n_fillet_smooth": 0,
        "n_plane": 0,
        "has_bore_cylinder_neighbor": False,
        "has_convex_to_bore_fillet": False,
        "has_concave_to_bore_fillet": False,
    }
    for nb in graph.neighbors.get(face_idx, ()):
        st = by_index[nb].surface_type
        kind = graph.edge_kind(face_idx, nb) or "?"
        if st in ("cylinder", "cone"):
            sig[f"n_cyl_{kind}"] = sig.get(f"n_cyl_{kind}", 0) + 1
            if nb in BORE_STACK:
                sig["has_bore_cylinder_neighbor"] = True
        elif st == "torus":
            sig[f"n_fillet_{kind}"] = sig.get(f"n_fillet_{kind}", 0) + 1
            if nb in BORE_STACK:
                if kind == "convex":
                    sig["has_convex_to_bore_fillet"] = True
                if kind == "concave":
                    sig["has_concave_to_bore_fillet"] = True
        elif st == "plane":
            sig["n_plane"] = int(sig["n_plane"]) + 1
    return sig


def opening_ward_of_curved(
    face_idx: int,
    curved_ids: set[int],
    faces_list: list,
    occ_map: dict,
    axis: _Axis,
) -> tuple[bool, float, float]:
    plane_pos = _axial_position(face_idx, faces_list, occ_map, axis)
    curved_pos = [
        _axial_position(i, faces_list, occ_map, axis)
        for i in curved_ids
    ]
    max_curved = max(curved_pos)
    return plane_pos > max_curved, plane_pos, max_curved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="96260B_front")
    ap.add_argument("--step", default=None)
    ap.add_argument("--graph-npz", default=None)
    args = ap.parse_args()

    cfg = PARTS.get(args.part, {})
    step_path = args.step or cfg.get("step") or f"{args.part}.step"
    graph_npz = args.graph_npz or cfg.get("graph_npz")

    faces_list = analyze_step(step_path)
    by_index = {f.index: f for f in faces_list}
    n_faces = len(faces_list)
    occ = load_step_faces(step_path)
    occ_map = {i: occ[i] for i in range(n_faces)}

    edge_index, edge_attr = _load_edges(Path(graph_npz), Path(step_path))
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)

    top_y = top_of_part_y(step_path, n_faces)

    cyl95 = by_index.get(95)
    axis_dir = _unit(np.asarray(cyl95.axis if cyl95 and cyl95.axis is not None else Y))
    axis_pt = np.mean(
        [np.asarray(by_index[i].centroid, dtype=float) for i in (95, 286) if i in by_index],
        axis=0,
    )
    axis = _Axis(axis_pt, axis_dir)

    curved_ids = {i for i in BORE_STACK if by_index[i].surface_type in CURVED_TYPES}
    shallowest_curved = min(
        (i for i in curved_ids),
        key=lambda i: top_y - float(by_index[i].centroid[1]),
    )
    shallowest_depth = top_y - float(by_index[shallowest_curved].centroid[1])

    print(f"part={args.part}  bore axis from cylinders 95/286")
    print(f"top-of-part Y = {top_y:.3f} mm ({top_y / MM_PER_IN:.4f} in)")
    print(f"shallowest curved in stack: face {shallowest_curved} "
          f"({by_index[shallowest_curved].surface_type}) depth={shallowest_depth:.2f} mm\n")

    profiles: dict[int, dict] = {}
    for fi in WATCH:
        f = by_index[fi]
        depth = top_y - float(f.centroid[1])
        rad = radial_dist_from_axis(f.centroid, axis_pt, axis_dir)
        sig = edge_signature(fi, graph, by_index)
        open_ward, plane_ax, max_curved_ax = opening_ward_of_curved(
            fi, curved_ids, faces_list, occ_map, axis,
        )
        profiles[fi] = {
            "depth": depth,
            "radial": rad,
            "sig": sig,
            "open_ward": open_ward,
            "plane_ax": plane_ax,
            "max_curved_ax": max_curved_ax,
        }

        print(f"=== face {fi} ({f.surface_type}, area={f.area:.1f} mm²) ===")
        print(f"  depth_below_top = {depth:.2f} mm ({depth / MM_PER_IN:.4f} in)")
        print(f"  radial_from_axis = {rad:.2f} mm")
        print(f"  axial_pos = {plane_ax:+.3f}  max_curved_axial = {max_curved_ax:+.3f}  "
              f"opening_ward_of_curved = {open_ward}")
        print(f"  edge signature: {sig}")
        print("  neighbors:")
        for nb in sorted(graph.neighbors.get(fi, ())):
            kind = graph.edge_kind(fi, nb)
            cos_v = graph.edge_cos(fi, nb)
            ang = edge_dihedral_deg(cos_v)
            ang_s = f"{ang:.1f}°" if ang is not None else "?"
            print(f"    {nb:3d}  {classify_neighbor(by_index, nb):16s}  "
                  f"{kind:8s}  cos={cos_v:+.3f}  ang={ang_s}")
        print()

    p93, p97 = profiles[93], profiles[97]

    checks = {
        "97_shallower_than_shallowest_curved": p97["depth"] < shallowest_depth,
        "93_at_or_below_entry_tier": p93["depth"] >= shallowest_depth,
        "97_opening_ward_of_curved_axial": p97["open_ward"],
        "93_not_opening_ward_of_curved": not p93["open_ward"],
        "97_no_cylinder_neighbors": p97["sig"]["n_cyl_concave"] + p97["sig"]["n_cyl_convex"] + p97["sig"]["n_cyl_smooth"] == 0,
        "93_has_cylinder_neighbors": (p93["sig"]["n_cyl_concave"] + p93["sig"]["n_cyl_convex"] + p93["sig"]["n_cyl_smooth"]) > 0,
        "97_convex_to_bore_fillet": p97["sig"]["has_convex_to_bore_fillet"],
        "93_concave_to_bore_fillet": p93["sig"]["has_concave_to_bore_fillet"],
        "97_not_only_concave_fillet_binding": p97["sig"]["has_convex_to_bore_fillet"],
    }

    print("--- separation checks ---")
    all_pass = True
    for name, ok in checks.items():
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{mark}] {name}")

    print("\n--- verdict ---")
    if all_pass:
        print("PASS: edge-adjacency + depth-relative-to-curved separates 97 (top flat) "
              "from 93 (hole ring). Safe to implement hole_detection gate.")
        return 0
    print("FAIL: probes do not cleanly separate 93 vs 97 — do NOT wire hole_detection fix yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
