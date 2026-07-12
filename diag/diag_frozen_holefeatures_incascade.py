"""DIAGNOSE-ONLY: what hole_result.features contains in the FULL cascade.

The counterbore precedent consumes hole_result.features post-cascade. A
post-hole countersink detector would see the SAME set. On fish_mold the coned
faces are claimed by pockets BEFORE the hole pass, so they should not appear as
hole features here — unlike the standalone detect_holes() on the full pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURES = {
    "96260B_front": ("96260B_FRONT_XR004_PCD PLATE.stp copy",
                     "eval/regression/graphs/96260B_front.graph.npz", "front"),
    "96260B_rear": ("96260B_REAR_XR004_PCD PLATE.stp copy",
                    "eval/regression/graphs/96260B_rear.graph.npz", "back"),
    "fish_mold": ("fish mold.stp", "eval/regression/graphs/fish_mold.graph.npz", "back"),
    "part1": ("part1.step", "eval/regression/graphs/part1.graph.npz", None),
    "part2": ("part2.step", "eval/regression/graphs/part2.graph.npz", None),
    "part3": ("part3.step", None, None),
}


def cone_refradius(occ_face):
    from hole_detection import _axis_from_occ_face
    r = _axis_from_occ_face(occ_face)
    return r[1] if r else None


def main() -> int:
    from run_cascade import run_cascade, _resolve_pocket_setup_for_cascade
    from pocket_detection import PocketDetectionConfig
    from hole_detection import HoleDetectionConfig
    from feature_params import load_step_faces

    for name, (step, npz, side) in FIXTURES.items():
        if not Path(step).is_file():
            print(f"[{name}] STEP MISSING — skip"); continue
        if npz and Path(npz).is_file():
            d = np.load(npz); ei, ea = d["edge_index"], d["edge_attr"]
        else:
            from step_ingest import ingest_step_to_pyg
            _x, ei, ea, _ = ingest_step_to_pyg(step)
        setup = _resolve_pocket_setup_for_cascade(Path(step), machining_side=side)
        pk_cfg = PocketDetectionConfig(setup=setup)
        out = run_cascade(Path(step), ei, ea,
                          pocket_config=pk_cfg,
                          hole_config=HoleDetectionConfig(max_hole_diameter_mm=150.0))
        faces = out[0]; hl = out[2]
        occ = load_step_faces(step)
        qual = []
        coned = []
        for f in hl.features:
            if not f.cone_face_indices:
                continue
            coned.append(f)
            if f.kind == "through_hole" and f.is_countersink and not f.is_counterbore:
                refs = [cone_refradius(occ[c]) for c in f.cone_face_indices]
                refs = [r for r in refs if r is not None and r > 0]
                widen = bool(refs) and max(refs) > f.radius
                if widen:
                    qual.append(sorted(int(i) for i in f.face_indices))
        print(f"[{name}] hole_features={len(hl.features)} coned_holes(in-cascade)={len(coned)}")
        for f in coned:
            refs = [cone_refradius(occ[c]) for c in f.cone_face_indices]
            print(f"    coned: kind={f.kind} cs={f.is_countersink} cb={f.is_counterbore} "
                  f"bore_r={f.radius:.3f} coneRefR={[round(r,3) if r else r for r in refs]} "
                  f"faces={sorted(int(i) for i in f.face_indices)}")
        print(f"    >>> COUNTERSINK QUALIFYING: {qual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
