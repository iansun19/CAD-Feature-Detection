#!/usr/bin/env python
"""
PROBE ONLY — validates ONE assumption, wires nothing.

Hypothesis: the ~14 WALL faces leaking into the cascade's "flat" bucket on
96260B_front separate cleanly from the 1 true flat by a single structural axis:
the angle between the face normal and the Y opening axis.

  - true flat  -> normal ~parallel to Y (angle near 0 deg or 180 deg)
  - wall       -> normal ~perpendicular to Y (angle near 90 deg)

If the bucket splits cleanly by this angle, the wall/flat distinction is
STRUCTURAL and recoverable for free (no Toolpath GT needed later).
If it does NOT split cleanly, the distinction is label-only -> you'd need
Toolpath GT to recover it, and we learned that cheaply.

Run with the mlcad env python, e.g.:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_flat_wall_normal.py \
      --part 96260B_front

This script deliberately does NOT import cascade internals it doesn't need.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from feature_params import analyze_step
from run_cascade import _load_edges, run_cascade

# Opening axis per project notes: Y. OCC works in mm.
Y_AXIS = np.array([0.0, 1.0, 0.0])

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


def angle_to_y_deg(normal):
    n = np.asarray(normal, dtype=float)
    ln = np.linalg.norm(n)
    if ln == 0:
        return float("nan")
    n = n / ln
    # fold to [0, 90]: a wall is ~90 whether normal points +/-, a flat is ~0 or ~180
    c = abs(float(np.dot(n, Y_AXIS)))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))  # 0 = parallel to Y (flat), 90 = perp (wall)


def get_flat_bucket_face_indices(part: str, *, step_path: str, graph_npz: str) -> list[int]:
    """Return face indices claimed by the cascade flats pass for `part`."""
    edge_index, edge_attr = _load_edges(Path(graph_npz), Path(step_path))
    _, _, _, flat_result, _, _ = run_cascade(step_path, edge_index, edge_attr)
    return sorted(int(i) for i in flat_result.claimed_faces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="96260B_front")
    ap.add_argument("--step", default=None,
                    help="Path to the STEP file (defaults from --part).")
    ap.add_argument("--graph-npz", default=None,
                    help="Cached face graph (defaults from --part).")
    ap.add_argument("--split-deg", type=float, default=45.0,
                    help="Angle threshold that would classify flat(<) vs wall(>).")
    args = ap.parse_args()

    cfg = PARTS.get(args.part, {})
    step_path = args.step or cfg.get("step") or f"{args.part}.step"
    graph_npz = args.graph_npz or cfg.get("graph_npz")

    faces = analyze_step(step_path)
    by_index = {f.index: f for f in faces}

    flat_idx = get_flat_bucket_face_indices(
        args.part, step_path=step_path, graph_npz=graph_npz,
    )
    print(f"part={args.part}  flat-bucket size = {len(flat_idx)} "
          f"(expect ~15: 1 true flat + ~14 walls)\n")

    rows = []
    for i in flat_idx:
        f = by_index.get(i)
        if f is None:
            print(f"  face {i}: NOT in analyze_step output (?)")
            continue
        ang = angle_to_y_deg(f.normal)
        rows.append((i, ang, getattr(f, "surface_type", "?"),
                     getattr(f, "area", float("nan"))))

    rows.sort(key=lambda r: r[1])  # by angle to Y
    print(f"{'face':>6} {'ang_to_Y':>9} {'surf':>10} {'area_mm2':>12}  guess")
    print("-" * 52)
    n_flat = n_wall = 0
    for i, ang, surf, area in rows:
        guess = "FLAT" if ang < args.split_deg else "wall"
        n_flat += guess == "FLAT"
        n_wall += guess == "wall"
        print(f"{i:>6} {ang:>9.2f} {str(surf):>10} {area:>12.2f}  {guess}")

    # Separation quality: gap between the flat-side cluster and wall-side cluster.
    angs = sorted(r[1] for r in rows)
    biggest_gap, gap_at = 0.0, None
    for a, b in zip(angs, angs[1:]):
        if b - a > biggest_gap:
            biggest_gap, gap_at = b - a, (a, b)

    print("\n--- verdict ---")
    print(f"guessed FLAT={n_flat}  wall={n_wall}  (want FLAT=1, wall~14)")
    if gap_at:
        print(f"largest angular gap in bucket: {biggest_gap:.1f} deg "
              f"between {gap_at[0]:.1f} and {gap_at[1]:.1f}")
    if n_flat == 1 and biggest_gap > 20:
        print("CLEAN: normal-vs-Y separates the 1 flat from the walls. "
              "Structural signature exists -> recoverable for free.")
    else:
        print("NOT CLEAN: angle alone does not isolate exactly 1 flat. "
              "Split may be label-only, or needs a second axis "
              "(e.g. depth-below-top == 0, or planar+area). "
              "Do NOT hardcode a threshold to force N=1.")


if __name__ == "__main__":
    main()
