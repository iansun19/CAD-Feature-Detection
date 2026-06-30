"""
check_correspondence.py — is the H5 model <-> STEP file (by id) mapping even correct?

For each candidate id we build a normalization-free-ish signature from BOTH sides:
  - surface-type histogram (generator scheme: plane=1,cyl=2,torus=3,sphere=4,cone=5,...)
  - sorted per-model-min-max-normalized face areas, resampled to fixed quantiles
Then cross-compare all H5 signatures vs all STEP signatures (NxN). If id->file mapping
is correct, the diagonal should be the unique minimum per row.

Also prints, per id, the STEP vs H5 type histogram so we can see whether the count
mismatch is a split/merge of specific surface types.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/check_correspondence.py
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

# OCC type -> generator face_type code
OCC2CODE = {GeomAbs_Plane: 1, GeomAbs_Cylinder: 2, GeomAbs_Torus: 3,
            GeomAbs_Sphere: 4, GeomAbs_Cone: 5}


def step_faces(path):
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        return None
    r.TransferRoots()
    shape = r.OneShape()
    types, areas = [], []
    for fc in TopologyExplorer(shape).faces():
        s = BRepAdaptor_Surface(fc, True)
        types.append(OCC2CODE.get(s.GetType(), 11))
        p = sprops(fc); areas.append(p.Mass())
    return np.array(types), np.array(areas)


def signature(types, areas):
    areas = np.asarray(areas, float)
    a = (areas - areas.min()) / (areas.max() - areas.min() + 1e-9)
    a = np.sort(a)
    q = np.interp(np.linspace(0, 1, 20), np.linspace(0, 1, len(a)), a)
    th = np.array([np.mean(types == t) for t in range(1, 12)])
    return np.concatenate([q, th])


def main():
    with open("diag/step_candidates.json") as f:
        cands = json.load(f)
    ids = [c["model_id"] for c in cands]

    h5_sig, step_sig = {}, {}
    print(f"{'id':>7} {'nH5':>4} {'nSTEP':>5}   H5 typehist        STEP typehist  (code:count)")
    for c in cands:
        mid = c["model_id"]
        v1 = np.array(c["v1"])
        h5_types = np.round(v1[:, 4] * 11).astype(int)         # generator 1-11 scheme
        h5_areas = v1[:, 0]
        path = os.path.join("MFCAD++_dataset", "step", c["split"], f"{mid}.step")
        sf = step_faces(path)
        if sf is None:
            print(f"{mid}: read fail"); continue
        st, sa = sf
        h5_sig[mid] = signature(h5_types, h5_areas)
        step_sig[mid] = signature(st, sa)

        def hist(ts):
            u, ct = np.unique(ts, return_counts=True)
            return {int(k): int(v) for k, v in zip(u, ct)}
        print(f"{mid:>7} {len(h5_types):>4} {len(st):>5}   {str(hist(h5_types)):20s} {hist(st)}")

    # cross signature distance matrix (rows=H5, cols=STEP)
    print("\ncross-match (rows=H5 id, value=STEP id with smallest signature distance):")
    n = len(ids)
    D = np.zeros((n, n))
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            D[i, j] = np.linalg.norm(h5_sig[a] - step_sig[b])
    correct = 0
    for i, a in enumerate(ids):
        j = int(np.argmin(D[i]))
        ok = ids[j] == a
        correct += ok
        print(f"  H5 {a:>7} -> best STEP {ids[j]:>7}  (self-dist={D[i,i]:.3f}, "
              f"best-dist={D[i,j]:.3f})  {'OK' if ok else 'MISMATCH'}")
    print(f"\nid-mapping self-match: {correct}/{n} "
          f"({'mapping looks CORRECT' if correct == n else 'mapping SUSPECT'})")


if __name__ == "__main__":
    main()
