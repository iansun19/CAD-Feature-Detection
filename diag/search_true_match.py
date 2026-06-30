"""
search_true_match.py — does each H5 model have an EXACT twin among the local .step
files (possibly under a different id)? If yes -> correspondence exists but ids are
remapped/shuffled. If no -> the local STEP export is a genuinely different B-rep
decomposition and per-face normals can't be transferred by matching at all.

Signature per part: (face_count, type-histogram, sorted min-max-normalized areas).
We scan a pool of STEP files, then for a few H5 models find the closest STEP part.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/search_true_match.py
"""
import os
import glob
import random
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
POOL = 700


def sig_from_types_areas(types, areas):
    types = np.asarray(types); areas = np.asarray(areas, float)
    th = np.array([np.sum(types == t) for t in range(1, 12)])
    a = np.sort((areas - areas.min()) / (areas.max() - areas.min() + 1e-9))
    q = np.interp(np.linspace(0, 1, 24), np.linspace(0, 1, len(a)), a)
    return len(types), th, q


def step_sig(path):
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        return None
    r.TransferRoots()
    shape = r.OneShape()
    types, areas = [], []
    for fc in TopologyExplorer(shape).faces():
        s = BRepAdaptor_Surface(fc, True)
        types.append(OCC2CODE.get(s.GetType(), 11))
        areas.append(sprops(fc).Mass())
    if not types:
        return None
    return sig_from_types_areas(types, areas)


def dist(sa, sb):
    na, tha, qa = sa; nb, thb, qb = sb
    return abs(na - nb) * 0.0 + np.abs(tha - thb).sum() + 5.0 * np.linalg.norm(qa - qb)


def main():
    random.seed(0)
    files = glob.glob(os.path.join(STEP_DIR, "*.step"))
    random.shuffle(files)
    files = files[:POOL]

    # ensure target ids are in the pool
    targets = ["5804", "27519", "20965", "46319", "50424"]
    for t in targets:
        p = os.path.join(STEP_DIR, f"{t}.step")
        if p not in files and os.path.isfile(p):
            files.append(p)

    pool = {}
    for i, p in enumerate(files):
        s = step_sig(p)
        if s is not None:
            pool[os.path.basename(p)[:-5]] = s
    print(f"indexed {len(pool)} STEP parts")

    # H5 signatures
    f = h5py.File(H5, "r")
    b = f[list(f.keys())[0]]
    cm = [x.decode() if isinstance(x, bytes) else str(x) for x in b["CAD_model"][()]]
    idx = np.asarray(b["idx"]); v1all = np.asarray(b["V_1"]); base = int(idx[0, 0])

    print(f"\n{'H5 id':>7} {'nH5':>4}  best-STEP-twin   bestDist  same-id-dist  exact?")
    for k in range(8):
        mid = cm[k]
        s = int(idx[k, 0]) - base
        e = int(idx[k + 1, 0]) - base if k + 1 < len(idx) else v1all.shape[0]
        v1 = v1all[s:e]
        h5_types = np.round(v1[:, 4] * 11).astype(int)
        hsig = sig_from_types_areas(h5_types, v1[:, 0])
        best, bd = None, 1e9
        for sid, ssig in pool.items():
            if ssig[0] != hsig[0]:
                continue                      # hard filter: same face count
            d = dist(hsig, ssig)
            if d < bd:
                bd, best = d, sid
        same = dist(hsig, pool[mid]) if mid in pool else float("nan")
        exact = best is not None and bd < 0.05
        print(f"{mid:>7} {len(v1):>4}  {str(best):>12}   {bd:8.3f}   {same:10.3f}   "
              f"{'YES' if exact else 'no'}")


if __name__ == "__main__":
    main()
