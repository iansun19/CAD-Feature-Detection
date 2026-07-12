"""Phase 1 probe: for each corpus part, report pocket-seed presence and the
broad-face axis the no-seed path WOULD derive. Read-only; changes nothing."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_params import analyze_step
from pocket_detection import (
    FLOOR_TYPES, SPHERE_TYPES, _broad_face_axis, _snap_to_cardinal,
    PocketDetectionConfig,
)

CORPUS = {
    "96260B_front": "96260B_FRONT_XR004_PCD PLATE.stp copy",
    "96260B_rear": "96260B_REAR_XR004_PCD PLATE.stp copy",
    "fish_mold": "fish mold.stp",
    "part1": "part1.step",
    "part2": "part2.step",
    "part3": "part3.step",
    "part4": "part4.step",
    "nist_ctc_01": "nist_ctc_01.step",
}

cfg = PocketDetectionConfig()
print(f"{'part':<16} {'seeds?':<8} {'#seed':<6} {'broad_area X/Y/Z (mm^2)':<40} {'broad_axis':<14} {'snapped':<14}")
print("-" * 110)

for name, step in CORPUS.items():
    p = Path(step)
    if not p.is_file():
        print(f"{name:<16} STEP NOT FOUND: {step}")
        continue
    faces = analyze_step(str(p))
    floor_ids = [f.index for f in faces if f.surface_type in FLOOR_TYPES]
    sphere_ids = [f.index for f in faces if f.surface_type in SPHERE_TYPES]
    n_seed = len(floor_ids) + len(sphere_ids)
    seeded = n_seed > 0

    # broad-face area buckets (mirror _broad_face_axis internals for reporting)
    area = np.zeros(3)
    for f in faces:
        if f.surface_type != "plane":
            continue
        n = np.asarray(f.normal, dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-12)
        area[int(np.argmax(np.abs(n)))] += float(f.area)
    broad = _broad_face_axis(faces)
    if broad is None:
        broad_s = "None(no planes)"
        snapped_s = "None"
    else:
        broad_s = str(broad.tolist())
        snapped_s = str(_snap_to_cardinal(broad, cfg.opening_axis_snap_tol_deg).tolist())
    area_s = f"{area[0]:.1f}/{area[1]:.1f}/{area[2]:.1f}"
    print(f"{name:<16} {str(seeded):<8} {n_seed:<6} {area_s:<40} {broad_s:<14} {snapped_s:<14}")
