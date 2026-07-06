#!/usr/bin/env python
"""
PROBE A — identify Toolpath's single flat (FACES(1) -> Face #1) among the 9
faces currently in the cascade's flat bucket on 96260B_front. Wires nothing.

Toolpath fingerprint for Face #1 (from panel):
    Depth below top of part = 0.1752 in   (= 4.450 mm)
    Feature depth           = 0 in
    3D Surface = No, Fillet radius = inf   (planar)
    tool dia ~5.9131 in                    (large-diameter face)

Goal: print, for each of the 9 flat-bucket faces, its depth-below-top (mm),
area, surface_type, and centroid, sorted by how close depth is to 4.450 mm.
The face that matches ~4.45 mm AND is large-area is Toolpath's flat.
The other 8 are then, by structure, NOT flats -> tells us the axis that
separates them (depth-below-top and/or pocket adjacency), WITHOUT hardcoding.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_identify_toolpath_flat.py --part 96260B_front
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from feature_params import analyze_step, load_step_faces
from run_cascade import _load_edges, run_cascade

try:
    from brep_extents import collect_boundary_points
except ImportError:
    collect_boundary_points = None  # type: ignore[misc, assignment]

MM_PER_IN = 25.4
TOOLPATH_FACE1_DEPTH_IN = 0.1752
TOOLPATH_FACE1_DEPTH_MM = TOOLPATH_FACE1_DEPTH_IN * MM_PER_IN  # 4.450 mm

# Opening axis = Y (project fact). "Top of part" is the max-Y extent of the
# stock along the opening axis; depth-below-top = (top_Y - face_centroid_Y).
Y = np.array([0.0, 1.0, 0.0])

PARTS: dict[str, dict[str, str]] = {
    "96260B_front": {
        "step": "96260B_FRONT_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_front/graph.npz",
    },
    "96260B_plate": {
        "step": "96260B_REAR_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_plate/graph.npz",
    },
}


def get_flat_bucket_face_indices(
    part: str, *, step_path: str, graph_npz: str,
) -> list[int]:
    """Return face indices claimed by the cascade flats pass for `part`."""
    edge_index, edge_attr = _load_edges(Path(graph_npz), Path(step_path))
    _, _, _, flat_result, _, _ = run_cascade(step_path, edge_index, edge_attr)
    return sorted(int(i) for i in flat_result.claimed_faces)


def top_of_part_y(step_path: str, faces) -> float:
    """Max Y over all face boundaries (opening datum). Falls back to centroid max."""
    if collect_boundary_points is not None:
        occ_faces = load_step_faces(step_path)
        all_y: list[float] = []
        for i in range(len(occ_faces)):
            pts = collect_boundary_points(occ_faces[i])
            all_y.extend(pts[:, 1].tolist())
        return max(all_y)
    return max(float(f.centroid[1]) for f in faces)


def depth_below_top_mm(face, top_y):
    return top_y - float(face.centroid[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="96260B_front")
    ap.add_argument("--step", default=None,
                    help="Path to the STEP file (defaults from --part).")
    ap.add_argument("--graph-npz", default=None,
                    help="Cached face graph (defaults from --part).")
    args = ap.parse_args()

    cfg = PARTS.get(args.part, {})
    step_path = args.step or cfg.get("step") or f"{args.part}.step"
    graph_npz = args.graph_npz or cfg.get("graph_npz")

    all_faces = analyze_step(step_path)
    by_index = {f.index: f for f in all_faces}

    # NOTE: top-of-part should be measured over ALL faces, not just the bucket.
    top_y = top_of_part_y(step_path, all_faces)
    print(f"part={args.part}")
    print(f"top-of-part Y (boundary max) = {top_y:.3f} mm ({top_y / MM_PER_IN:.4f} in)")
    print(f"Toolpath Face#1 target depth = {TOOLPATH_FACE1_DEPTH_MM:.3f} mm "
          f"({TOOLPATH_FACE1_DEPTH_IN} in)\n")

    idx = get_flat_bucket_face_indices(
        args.part, step_path=step_path, graph_npz=graph_npz,
    )
    print(f"flat-bucket size = {len(idx)}\n")

    rows = []
    for i in idx:
        f = by_index.get(i)
        if f is None:
            print(f"  face {i}: NOT in analyze_step output (?)")
            continue
        d = depth_below_top_mm(f, top_y)
        rows.append((
            i, d, abs(d - TOOLPATH_FACE1_DEPTH_MM),
            getattr(f, "area", float("nan")),
            str(getattr(f, "surface_type", "?")),
            tuple(round(float(c), 1) for c in f.centroid),
        ))

    rows.sort(key=lambda r: r[2])  # closest depth-match first
    print(f"{'face':>5} {'depth_mm':>9} {'|d-4.45|':>9} "
          f"{'area_mm2':>10} {'surf':>10}  centroid")
    print("-" * 78)
    for i, d, dd, area, surf, cen in rows:
        flag = "  <== depth+size match?" if dd < 0.6 and area > 1000 else ""
        print(f"{i:>5} {d:>9.3f} {dd:>9.3f} {area:>10.1f} {surf:>10}  {cen}{flag}")

    print("\n--- read ---")
    print("Expect exactly ONE row with depth ~4.45 mm AND large area = Toolpath's flat.")
    print("The 7 small ~104 mm2 step-planes should sit at a DIFFERENT depth")
    print("(pocket-floor depth), which is the structural axis that separates them.")
    print("Do NOT hardcode 4.45 or 2463 as a gate -- report the separation, then")
    print("we choose a relative rule (e.g. depth-cluster + not-pocket-adjacent).")


if __name__ == "__main__":
    main()
