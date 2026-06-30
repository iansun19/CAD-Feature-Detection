"""
verify_id_map.py — independent correspondence check on the first models of batch0
(not selected for class-8). For each H5 model compare to STEP file of the SAME id:
  face count, surface-type histogram, and a normalized-centroid bijection residual.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/verify_id_map.py
"""
import os
import numpy as np
import h5py

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
H5 = "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
STEP_DIR = "MFCAD++_dataset/step/train"


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


def hist(ts):
    u, ct = np.unique(ts, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, ct)}


def bij_resid(h5c, sc):
    D = np.linalg.norm(h5c[:, None, :] - sc[None, :, :], axis=2)
    used = set(); resid = []
    for i in np.argsort(D.min(1)):
        order = np.argsort(D[i])
        j = next((k for k in order if k not in used), None)
        if j is None:
            break
        used.add(int(j)); resid.append(D[i, j])
    return np.array(resid)


def main():
    f = h5py.File(H5, "r")
    bk = list(f.keys())[0]
    b = f[bk]
    cm = [x.decode() if isinstance(x, bytes) else str(x) for x in b["CAD_model"][()]]
    idx = np.asarray(b["idx"]); v1all = np.asarray(b["V_1"]); base = int(idx[0, 0])

    print(f"{'id':>7} {'nH5':>4} {'nSTEP':>5} {'cnt':>4}  {'H5 types':<16} {'STEP types':<16} "
          f"{'medResid':>8} {'frac<.02':>8}")
    n_clean = 0; n_tot = 0
    for k in range(12):
        mid = cm[k]
        s = int(idx[k, 0]) - base
        e = int(idx[k + 1, 0]) - base if k + 1 < len(idx) else v1all.shape[0]
        v1 = v1all[s:e]
        h5_types = np.round(v1[:, 4] * 11).astype(int)
        path = os.path.join(STEP_DIR, f"{mid}.step")
        if not os.path.isfile(path):
            print(f"{mid:>7}  STEP missing"); continue
        sf = step_feats(path)
        if sf is None:
            print(f"{mid:>7}  STEP read fail"); continue
        srow, stype = sf
        n_tot += 1
        cnt_ok = len(srow) == len(v1)
        resid = np.array([np.nan]); frac = np.nan
        if cnt_ok:
            resid = bij_resid(mm(v1[:, 1:4]), mm(srow[:, 1:3 + 1]))
            frac = float(np.mean(resid < 0.02))
            if np.nanmax(resid) < 0.05:
                n_clean += 1
        print(f"{mid:>7} {len(v1):>4} {len(srow):>5} {str(cnt_ok):>4}  "
              f"{str(hist(h5_types)):<16} {str(hist(stype)):<16} "
              f"{np.median(resid):>8.3f} {frac:>8.2f}")
    print(f"\nclean correspondences: {n_clean}/{n_tot}")


if __name__ == "__main__":
    main()
