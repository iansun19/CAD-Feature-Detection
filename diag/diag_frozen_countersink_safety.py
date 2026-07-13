"""DIAGNOSE-ONLY: golden-safety evidence for the proposed countersink predicate.

For each frozen part, run the REAL hole pass (pinned graph.npz + step) and apply
the candidate countersink predicate:

    kind == "through_hole" AND is_countersink AND NOT is_counterbore
    AND >=1 coaxial cone whose RefRadius > bore radius (widening entry)

Print the qualifying set per frozen part — MUST be empty on all five.
Also list every coned hole feature (any kind) for transparency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURES = {
    "96260B_front": ("fixtures/step/96260B_front.stp",
                     "eval/regression/graphs/96260B_front.graph.npz"),
    "96260B_rear": ("fixtures/step/96260B_rear.stp",
                    "eval/regression/graphs/96260B_rear.graph.npz"),
    "fish_mold": ("fixtures/step/fish_mold.stp", "eval/regression/graphs/fish_mold.graph.npz"),
    "part1": ("fixtures/step/fixtures/step/part1.step", "eval/regression/graphs/part1.graph.npz"),
    "part2": ("fixtures/step/fixtures/step/part2.step", "eval/regression/graphs/part2.graph.npz"),
}


def countersink_qualifies(feat) -> bool:
    """Proposed predicate — evaluated on a hole feature."""
    if feat.kind != "through_hole":
        return False
    if feat.is_counterbore:
        return False
    if not feat.is_countersink:
        return False
    # widening-entry check: at least one cone RefRadius exceeds the bore radius.
    if not feat.cone_face_indices:
        return False
    return True  # RefRadius>bore checked in the OCC probe below


def main() -> int:
    from brep.feature_params import analyze_step, load_step_faces
    from cascade.hole_detection import detect_holes

    for name, (step, npz) in FIXTURES.items():
        if not Path(step).is_file():
            print(f"[{name}] STEP MISSING: {step} — skipping")
            continue
        d = np.load(npz)
        ei, ea = d["edge_index"], d["edge_attr"]
        faces = analyze_step(step)
        occ = load_step_faces(step)
        res = detect_holes(faces, ei, ea, occ_faces=occ,
                           candidate_faces=set(range(len(faces))))
        coned = [f for f in res.features if f.cone_face_indices]
        qual = []
        for f in res.features:
            if countersink_qualifies(f):
                # verify widening: max cone RefRadius > bore radius
                cone_r = max(float(faces[c].radius or 0.0) for c in f.cone_face_indices)
                if cone_r > f.radius:
                    qual.append(sorted(int(i) for i in f.face_indices))
        print(f"[{name}] n_faces={len(faces)} holes={len(res.features)} "
              f"coned_holes={len(coned)}")
        for f in coned:
            print(f"    coned: kind={f.kind} is_cs={f.is_countersink} "
                  f"is_cb={f.is_counterbore} cones={sorted(f.cone_face_indices)} "
                  f"faces={sorted(int(i) for i in f.face_indices)}")
        print(f"    >>> COUNTERSINK QUALIFYING SET: {qual}  "
              f"({'EMPTY ✓' if not qual else 'NON-EMPTY ✗✗✗'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
