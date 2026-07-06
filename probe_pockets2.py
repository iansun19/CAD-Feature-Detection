"""probe_pockets2.py — verify Q1 with REAL OCC axis lines (location+direction).

axis_key() in probe_pockets.py anchors on the face centroid, which is off-axis for
partial-arc faces and can over-split a genuine coaxial stack. Here we read the true
axis line from the OCC face and group with hole_detection's own colinearity test.
"""
import numpy as np

from feature_params import analyze_step, load_step_faces
from hole_detection import (
    Candidate,
    HoleDetectionConfig,
    _axis_from_occ_face,
    _coaxial,
    _UnionFind,
)

STEP_PATH = "96260B_REAR_XR004_PCD PLATE.stp copy"
MM_PER_IN = 25.4

faces = analyze_step(STEP_PATH)
occ = load_step_faces(STEP_PATH)


def dia_in(f):
    return (2 * f.radius) / MM_PER_IN if f.radius else None


def near(x, target, tol=0.05):
    return x is not None and abs(x - target) < tol


targets = {"0.800": 0.800, "0.500": 0.500, "3.453": 3.453}

# Build Candidates (with true axis lines) for every cyl/cone face in the 3 families.
cands = []
fam_of = {}
for f in faces:
    if f.surface_type not in ("cylinder", "cone") or not f.radius:
        continue
    di = dia_in(f)
    for name, t in targets.items():
        if near(di, t):
            info = _axis_from_occ_face(occ[f.index])
            if info is None:
                continue
            axis, occ_r, kind = info
            cands.append(Candidate(f.index, kind, axis, occ_r or f.radius, True))
            fam_of[f.index] = name

cfg = HoleDetectionConfig()

# Coaxial union-find over the REAL axis lines (no graph-connectivity requirement:
# this measures pure geometric coaxiality, which is what "shared pocket axis" means).
uf = _UnionFind(range(len(cands)))
for i in range(len(cands)):
    for j in range(i + 1, len(cands)):
        if _coaxial(cands[i], cands[j], cfg):
            uf.union(i, j)

groups = uf.groups()
print(f"{len(cands)} family faces -> {len(groups)} distinct coaxial axes "
      f"(real OCC axis lines, angular<= {cfg.axis_angular_tol_deg} deg)")
print()
for gi, grp in enumerate(sorted(groups, key=lambda g: -len(g))):
    idxs = sorted(cands[i].face_index for i in grp)
    comp = {}
    for i in grp:
        comp[fam_of[cands[i].face_index]] = comp.get(fam_of[cands[i].face_index], 0) + 1
    comp_s = ", ".join(f"{k}×{v}" for k, v in sorted(comp.items()))
    d = cands[grp[0]].axis.direction
    print(f"  axis {gi}: {len(idxs)} faces  [{comp_s}]  dir=[{d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}]  faces={idxs}")
