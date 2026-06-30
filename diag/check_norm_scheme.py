"""
check_norm_scheme.py
 - What normalization does H5 V_1 use? (print area/centroid ranges)
 - For equal-count models, does a clean 1:1 bijection exist between H5 normalized
   centroids and STEP min-max-normalized centroids? (definitive correspondence test)

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/check_norm_scheme.py
"""
import json
import os
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus)
from OCC.Extend.TopologyUtils import TopologyExplorer
try:
    from OCC.Core.BRepGProp import brepgprop
    def sprops(f):
        p = GProp_GProps(); brepgprop.SurfaceProperties(f, p); return p
except Exception:
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
    def sprops(f):
        p = GProp_GProps(); brepgprop_SurfaceProperties(f, p); return p

OCC2CODE = {GeomAbs_Plane: 1, GeomAbs_Cylinder: 2, GeomAbs_Torus: 3,
            GeomAbs_Sphere: 4, GeomAbs_Cone: 5}


def step_feats(path):
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        return None
    r.TransferRoots()
    shape = r.OneShape()
    rows, types = [], []
    for fc in TopologyExplorer(shape).faces():
        s = BRepAdaptor_Surface(fc, True)
        types.append(OCC2CODE.get(s.GetType(), 11))
        p = sprops(fc); c = p.CentreOfMass()
        rows.append([p.Mass(), c.X(), c.Y(), c.Z()])
    return np.array(rows), np.array(types)


def mm(a):
    a = np.asarray(a, float)
    return (a - a.min(0)) / (a.max(0) - a.min(0) + 1e-9)


def main():
    with open("diag/step_candidates.json") as f:
        cands = json.load(f)

    print("H5 V_1 ranges (area, cx, cy, cz) — tells us the H5 normalization scheme:")
    for c in cands[:4]:
        v1 = np.array(c["v1"])
        lo = v1[:, :4].min(0); hi = v1[:, :4].max(0)
        print(f"  {c['model_id']:>7}: min={np.round(lo,3).tolist()}  max={np.round(hi,3).tolist()}")

    print("\nbijection test (equal-count models): H5 norm-centroid vs STEP minmax-centroid")
    for c in cands:
        mid = c["model_id"]
        v1 = np.array(c["v1"])
        path = os.path.join("MFCAD++_dataset", "step", c["split"], f"{mid}.step")
        sf = step_feats(path)
        if sf is None:
            continue
        srow, stype = sf
        if len(srow) != len(v1):
            continue   # only equal-count models for the clean bijection test
        h5c = mm(v1[:, 1:4])              # H5 centroids re-min-maxed to [0,1]
        sc = mm(srow[:, 1:3 + 1])         # STEP centroids min-maxed
        # greedy nearest bijection
        D = np.linalg.norm(h5c[:, None, :] - sc[None, :, :], axis=2)
        used = set(); resid = []; bij = True
        for i in np.argsort(D.min(1)):
            order = np.argsort(D[i])
            j = next((k for k in order if k not in used), None)
            if j is None:
                bij = False; break
            used.add(int(j)); resid.append(D[i, j])
        resid = np.array(resid)
        print(f"  {mid:>7} (n={len(v1)}): median resid={np.median(resid):.3f} "
              f"max={resid.max():.3f}  frac<0.02={np.mean(resid<0.02):.2f}  "
              f"{'CLEAN' if resid.max()<0.05 else 'NOISY/NO-CORRESPONDENCE'}")


if __name__ == "__main__":
    main()
