"""DIAGNOSE-ONLY: per-face inventory of the 10 Part3 countersink groups.

No edits to cascade code. Dumps surface type, area, axis, half-angle (cones),
radius, centroid, adjacency, axial ordering, and confirms each 4-face group is
one coaxial unit. Also runs the hole pass to show current ownership + params.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STEP = "part3.step"
GROUPS = [
    [0, 1, 44, 45], [10, 11, 34, 35],
    [2, 3, 42, 43], [12, 13, 32, 33],
    [4, 5, 40, 41], [14, 15, 30, 31],
    [6, 7, 38, 39], [16, 17, 28, 29],
    [8, 9, 36, 37], [18, 19, 26, 27],
]
TARGET = sorted({f for g in GROUPS for f in g})


def cone_half_angle_deg(occ_face):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cone

    surf = BRepAdaptor_Surface(occ_face, True)
    if surf.GetType() != GeomAbs_Cone:
        return None
    return float(np.degrees(surf.Cone().SemiAngle()))


def occ_axis_radius(occ_face):
    from hole_detection import _axis_from_occ_face

    r = _axis_from_occ_face(occ_face)
    if r is None:
        return None
    ax, rad, kind = r
    return ax.point, ax.direction, rad, kind


def main() -> int:
    from feature_params import analyze_step, load_step_faces
    from step_ingest import ingest_step_to_pyg

    faces = analyze_step(STEP)
    occ = load_step_faces(STEP)
    _x, edge_index, edge_attr, _stats = ingest_step_to_pyg(STEP)

    # adjacency
    adj: dict[int, set[int]] = {}
    ei = np.asarray(edge_index)
    for a, b in zip(ei[0], ei[1]):
        adj.setdefault(int(a), set()).add(int(b))
        adj.setdefault(int(b), set()).add(int(a))

    print(f"n_faces={len(faces)}  target={len(TARGET)} faces in {len(GROUPS)} groups\n")
    print(f"{'idx':>4} {'stype':>9} {'area':>9} {'radius':>8} {'half°':>7} "
          f"{'axisdir':>22} {'centroid':>26}  neighbors")
    for i in TARGET:
        g = faces[i]
        occf = occ[i]
        ha = cone_half_angle_deg(occf)
        info = occ_axis_radius(occf)
        rad = info[2] if info else None
        axd = info[1] if info else np.asarray(g.axis) if g.axis is not None else None
        axs = "[%+.2f %+.2f %+.2f]" % tuple(axd) if axd is not None else "-"
        cen = "[%+7.2f %+7.2f %+7.2f]" % tuple(g.centroid)
        nb = sorted(n for n in adj.get(i, set()))
        print(f"{i:>4} {g.surface_type:>9} {g.area:>9.2f} "
              f"{(rad if rad is not None else -1):>8.3f} "
              f"{(ha if ha is not None else float('nan')):>7.2f} {axs:>22} {cen}  {nb}")

    print("\n--- per-group coaxiality + axial ordering ---")
    for gi, grp in enumerate(GROUPS):
        print(f"\ngroup {gi}: faces {grp}")
        # collect axes
        axes = []
        for i in grp:
            info = occ_axis_radius(occ[i])
            if info:
                pt, d, rad, kind = info
                axes.append((i, kind, pt, d, rad))
        # reference axis = first
        i0, k0, pt0, d0, r0 = axes[0]
        for (i, kind, pt, d, rad) in axes:
            # perp dist of this axis point to reference line
            v = pt - pt0
            perp = np.linalg.norm(v - np.dot(v, d0) * d0)
            paral = abs(float(np.dot(d, d0)))
            # axial position of centroid
            apos = float(np.dot(faces[i].centroid - pt0, d0))
            ha = cone_half_angle_deg(occ[i])
            print(f"   f{i:>2} {kind:>8} r={rad:>6.3f} half={ha if ha else 0:>6.2f} "
                  f"axis‖={paral:.4f} perp={perp:.4f}mm  axialpos={apos:+.3f}")
        # intra-group adjacency edges
        edges = []
        gs = set(grp)
        for i in grp:
            for n in adj.get(i, set()):
                if n in gs and n > i:
                    edges.append((i, n))
        print(f"   intra-group edges: {edges}")
        ext = {}
        for i in grp:
            outside = sorted(n for n in adj.get(i, set()) if n not in gs)
            if outside:
                ext[i] = outside
        print(f"   external neighbors: {ext}")

    # ---- run the hole pass on the same pool to show ownership & params ----
    print("\n--- hole pass (full-face candidate pool) ---")
    from hole_detection import detect_holes

    res = detect_holes(faces, edge_index, edge_attr, occ_faces=occ,
                       candidate_faces=set(range(len(faces))))
    for f in res.features:
        fs = sorted(int(x) for x in f.face_indices)
        if set(fs) & set(TARGET):
            print(f"  feat kind={f.kind} is_cs={f.is_countersink} "
                  f"is_cb={f.is_counterbore} r={f.radius:.3f} "
                  f"cones={sorted(f.cone_face_indices)} cyls={sorted(f.cylinder_face_indices)} "
                  f"faces={fs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
