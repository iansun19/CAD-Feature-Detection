#!/usr/bin/env python
"""
PROBE — why does the blind-hole recognizer annex face 97 (Toolpath's flat)?

Established:
  hole feature 0: kind=blind_hole, dia=101.75 mm,
  faces = [93, 94, 95, 96 (cylinders), 97, 285, 286, 287 (planes)]
  Face 97 = Toolpath's standalone flat (depth 4.45 mm, +Y). It should NOT be
  in the hole. 285/286/287 are presumably legit bore caps/floor.

Question: what structural axis separates 97 (wrongly annexed) from 285/286/287
(legit)? Two hypotheses:
  H1 depth: 97 sits at top-of-part tier (~4.45 mm), true caps sit deep.
  H2 radius: 97 lies radially OUTSIDE the bore radius (it's a mouth ring),
             true caps lie within the bore footprint.

This dumps, for each of the 4 planes + the 4 cylinders, the geometry needed
to decide: depth-below-top, centroid, area, and radial distance of the plane
centroid from the bore axis. Wires nothing.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_hole_annex_97.py --part 96260B_front
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from feature_params import analyze_step, load_step_faces

try:
    from brep_extents import collect_boundary_points
except ImportError:
    collect_boundary_points = None  # type: ignore[misc, assignment]

MM_PER_IN = 25.4
CYL_FACES = [93, 94, 95, 96]
PLANE_FACES = [97, 285, 286, 287]
Y = np.array([0.0, 1.0, 0.0])  # opening axis

PARTS: dict[str, dict[str, str]] = {
    "96260B_front": {
        "step": "96260B_FRONT_XR004_PCD PLATE.stp copy",
    },
    "96260B_plate": {
        "step": "96260B_REAR_XR004_PCD PLATE.stp copy",
    },
}


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


def radial_dist_from_axis(centroid, axis_pt, axis_dir):
    """Perpendicular distance of centroid from the line (axis_pt, axis_dir)."""
    c = np.asarray(centroid, dtype=float) - np.asarray(axis_pt, dtype=float)
    d = np.asarray(axis_dir, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    proj = np.dot(c, d) * d
    return float(np.linalg.norm(c - proj))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="96260B_front")
    ap.add_argument("--step", default=None)
    args = ap.parse_args()

    cfg = PARTS.get(args.part, {})
    step_path = args.step or cfg.get("step") or f"{args.part}.step"

    faces = analyze_step(step_path)
    by_index = {f.index: f for f in faces}

    top_y = top_of_part_y(step_path, faces)

    # Bore axis: prefer a true cylinder in the stack; fall back to mean centroids.
    cyls = [by_index[i] for i in CYL_FACES if i in by_index]
    true_cyls = [c for c in cyls if c.surface_type == "cylinder" and c.radius is not None]
    axis_src = true_cyls[0] if true_cyls else (cyls[0] if cyls else None)
    if axis_src is not None:
        axis_dir = np.asarray(axis_src.axis if axis_src.axis is not None else Y, dtype=float)
        axis_pt = np.mean([np.asarray(c.centroid, dtype=float) for c in cyls], axis=0)
        bore_radii = [
            float(c.radius) if c.radius is not None
            else float(c.torus_major_r) if c.torus_major_r is not None
            else float("nan")
            for c in cyls
        ]
        cyl_radii = [float(c.radius) for c in true_cyls]
        bore_r = float(np.mean(cyl_radii)) if cyl_radii else float("nan")
        print(f"hole stack {CYL_FACES}  surf="
              f"{[by_index[i].surface_type for i in CYL_FACES if i in by_index]}")
        print(f"  radii(mm)={[round(r, 2) if r == r else None for r in bore_radii]}  "
              f"(cylinder R={bore_r:.2f} mm, dia~{2 * bore_r:.1f} mm)")
    else:
        axis_dir = Y
        axis_pt = np.array([0.0, 0.0, 0.0])
        bore_r = float("nan")
        print("WARN: no bore stack faces found; using Y through origin as axis")

    print(f"top-of-part Y (boundary max) = {top_y:.2f} mm ({top_y / MM_PER_IN:.4f} in)\n")
    print(f"{'face':>5} {'surf':>8} {'depth_mm':>9} {'area_mm2':>10} "
          f"{'radial_mm':>10}  centroid")
    print("-" * 78)
    for i in CYL_FACES + PLANE_FACES:
        f = by_index.get(i)
        if f is None:
            print(f"{i:>5}  (missing from analyze_step output)")
            continue
        depth = top_y - float(f.centroid[1])
        area = getattr(f, "area", float("nan"))
        rad = radial_dist_from_axis(f.centroid, axis_pt, axis_dir)
        tag = "  <== 97 (Toolpath flat)" if i == 97 else ""
        if i in PLANE_FACES and bore_r == bore_r:
            if rad > bore_r + 1.0:
                tag += "  radial> bore R"
            elif rad <= bore_r:
                tag += "  radial<= bore R"
        print(f"{i:>5} {str(f.surface_type):>8} {depth:>9.2f} {area:>10.1f} {rad:>10.2f}  "
              f"{tuple(round(float(c), 1) for c in f.centroid)}{tag}")

    print("\n--- read ---")
    print("Compare face 97 to 285/286/287:")
    print("  H1 TRUE if 97's depth is much SMALLER (top tier) than the caps' depth.")
    print("     -> hole pass must not annex planes at top-of-part tier (they're flats).")
    print("  H2 TRUE if 97's radial_mm is > bore R (~50.9) while caps are <= R.")
    print("     -> hole pass must only claim caps within the bore footprint.")
    print("Pick the axis that cleanly separates 97 from the 3 legit caps; do NOT")
    print("hardcode 101.75 or 4.45 -- express the gate relative to the bore's own")
    print("radius / the part's own top tier.")


if __name__ == "__main__":
    main()
