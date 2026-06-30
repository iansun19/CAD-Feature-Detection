"""
global_twin_search.py — decide between two hypotheses for the H5<->STEP mismatch:
  (A) SHUFFLED IDS, same export: every H5 model has an EXACT twin somewhere in the
      local .step set (count+typehist+sorted-areas match), just under a different id
      -> mapping is recoverable by global matching.
  (B) DIFFERENT EXPORT: H5 models have NO exact twin among local .step files
      -> per-face normals cannot be transferred at all; STEP approach is a dead end.

Indexes ALL train .step signatures once (cached to npz), then tests a random sample
of H5 models for an exact twin.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/global_twin_search.py
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
CACHE = "diag/step_train_sigs.npz"
NQ = 24


def sig(types, areas):
    types = np.asarray(types); areas = np.asarray(areas, float)
    th = np.array([np.sum(types == t) for t in range(1, 12)], float)
    a = np.sort((areas - areas.min()) / (areas.max() - areas.min() + 1e-9))
    q = np.interp(np.linspace(0, 1, NQ), np.linspace(0, 1, len(a)), a)
    return np.concatenate([[len(types)], th, q])  # 1 + 11 + NQ


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
    return sig(types, areas)


def build_index():
    if os.path.isfile(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return list(d["ids"]), d["sigs"]
    files = sorted(glob.glob(os.path.join(STEP_DIR, "*.step")))
    ids, sigs = [], []
    for i, p in enumerate(files):
        s = step_sig(p)
        if s is not None:
            ids.append(os.path.basename(p)[:-5]); sigs.append(s)
        if i % 2000 == 0:
            print(f"  indexed {i}/{len(files)}", flush=True)
    sigs = np.array(sigs)
    np.savez(CACHE, ids=np.array(ids), sigs=sigs)
    print(f"  cached {len(ids)} sigs -> {CACHE}", flush=True)
    return ids, sigs


def main():
    ids, sigs = build_index()
    ids = list(ids)
    counts = sigs[:, 0].astype(int)
    th = sigs[:, 1:12]
    q = sigs[:, 12:]
    id_pos = {sid: i for i, sid in enumerate(ids)}

    f = h5py.File(H5, "r")
    bk = list(f.keys())
    random.seed(1)
    samples = []
    for _ in range(200):
        b = f[random.choice(bk)]
        cm = [x.decode() if isinstance(x, bytes) else str(x) for x in b["CAD_model"][()]]
        idx = np.asarray(b["idx"]); v1all = np.asarray(b["V_1"]); base = int(idx[0, 0])
        k = random.randrange(len(cm))
        s = int(idx[k, 0]) - base
        e = int(idx[k + 1, 0]) - base if k + 1 < len(idx) else v1all.shape[0]
        samples.append((cm[k], v1all[s:e]))

    n_exact = 0; n_same_id = 0; n_self_present = 0
    dists = []
    for mid, v1 in samples:
        h5_types = np.round(v1[:, 4] * 11).astype(int)
        hs = sig(h5_types, v1[:, 0])
        m = counts == int(hs[0])
        if not np.any(m):
            dists.append(np.inf); continue
        d = np.abs(th[m] - hs[1:12]).sum(1) + 5.0 * np.linalg.norm(q[m] - hs[12:], axis=1)
        bd = float(d.min())
        best_id = np.array(ids)[m][int(np.argmin(d))]
        dists.append(bd)
        if bd < 0.05:
            n_exact += 1
            if best_id == mid:
                n_same_id += 1
        if mid in id_pos:
            n_self_present += 1
    dists = np.array(dists)
    print(f"\nsampled {len(samples)} H5 train models")
    print(f"  have an EXACT twin in local .step set (dist<0.05): {n_exact} "
          f"({100*n_exact/len(samples):.0f}%)")
    print(f"    of those, twin is the SAME id: {n_same_id}")
    print(f"  same-id .step file exists for: {n_self_present}/{len(samples)}")
    finite = dists[np.isfinite(dists)]
    print(f"  best-twin distance: median={np.median(finite):.2f} "
          f"min={finite.min():.2f}  (exact<0.05)")
    print("\nINTERPRETATION:")
    print("  ~100% exact + same-id     -> ids correct, count diffs were my matching bug")
    print("  ~100% exact + different id-> SHUFFLED ids, recoverable by global twin match")
    print("  ~0% exact                 -> DIFFERENT export, normals NOT transferable")


if __name__ == "__main__":
    main()
