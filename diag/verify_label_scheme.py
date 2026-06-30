"""
verify_label_scheme.py  (Step C)

(1) SEMANTIC/GEOMETRIC: regenerate parts from local STEP and, per feature class in the
    confused step-family, measure the dihedral on concave same-label adjacent face
    pairs. Documented meaning must hold:
      class 8  (rectangular through step) -> ~90 deg
      class 10 (slanted through step)     -> systematically != 90 deg
      class 9  (2-sided through step)     -> ~90 deg (two rectangular walls)
      class 22 (rectangular blind step)   -> ~90 deg
    If 8 and 10 separate cleanly, the label integers match the documented scheme AND
    the dihedral feature distinguishes exactly the classes the model confuses.

(2) CROSS-CHECK vs released H5: for simple single-feature local-STEP parts
    (label set == {24=stock, k}), confirm the released H5 part of the SAME id also
    contains class k. Reports agreement rate.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/verify_label_scheme.py
"""
import os
import sys
import glob
import random
import numpy as np
import h5py

sys.path.insert(0, "diag")
from regen_dihedral_check import read_part, convexity_sign  # faithful regen helpers
from OCC.Extend.TopologyUtils import TopologyExplorer

STEP_DIR = "MFCAD++_dataset/step/train"
H5 = "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
NAMES = {0: "Chamfer", 1: "Through hole", 2: "Tri passage", 3: "Rect passage",
         4: "6-sided passage", 5: "Tri through slot", 6: "Rect through slot",
         7: "Circ through slot", 8: "Rect through step", 9: "2-sided through step",
         10: "Slanted through step", 11: "O-ring", 12: "Blind hole", 13: "Tri pocket",
         14: "Rect pocket", 15: "6-sided pocket", 16: "Circ end pocket",
         17: "Rect blind slot", 18: "V circ end blind slot", 19: "H circ end blind slot",
         20: "Tri blind step", 21: "Circ blind step", 22: "Rect blind step",
         23: "Round", 24: "Stock"}


def concave_same_label_angles(shape, faces, fidx, labels, normals, target):
    topo = TopologyExplorer(shape)
    out = []
    for edge in topo.edges():
        ef = list(topo.faces_from_edge(edge))
        if len(ef) != 2:
            continue
        i, j = fidx[ef[0]], fidx[ef[1]]
        if labels[i] != target or labels[j] != target:
            continue
        if convexity_sign(edge, ef) >= 0:        # concave only
            continue
        ni, nj = normals[i], normals[j]
        if ni is None or nj is None:
            continue
        cos = np.clip(ni @ nj / (np.linalg.norm(ni) * np.linalg.norm(nj)), -1, 1)
        out.append(float(np.degrees(np.arccos(cos))))
    return out


def scan_class(files, target, max_parts=20, scan_cap=600):
    ang = []; used = 0; scanned = 0
    tstr = str(target)
    for path in files:
        if used >= max_parts or scanned >= scan_cap:
            break
        scanned += 1
        try:
            shape, faces, fidx, labels, normals = read_part(path)
        except Exception:
            continue
        if tstr not in labels:
            continue
        a = concave_same_label_angles(shape, faces, fidx, labels, normals, tstr)
        if a:
            ang.extend(a); used += 1
    return np.array(ang), used, scanned


def main():
    random.seed(7)
    files = glob.glob(os.path.join(STEP_DIR, "*.step"))
    random.shuffle(files)

    print("=== (1) per-class concave same-label dihedral (regenerated from local STEP) ===", flush=True)
    for cls in [8, 10, 9, 22]:
        a, used, scanned = scan_class(files, cls, max_parts=20, scan_cap=600)
        if a.size:
            within = 100 * np.mean((a >= 80) & (a <= 95))
            print(f"  class {cls:>2} {NAMES[cls]:<22} n={a.size:4d} parts={used:2d}  "
                  f"mean={a.mean():5.1f} median={np.median(a):5.1f} std={a.std():4.1f} "
                  f"80-95deg={within:5.1f}%")
        else:
            print(f"  class {cls:>2} {NAMES[cls]:<22} no concave same-label pairs found "
                  f"(scanned {scanned})")

    print("\n=== (2) cross-check simple single-feature parts vs released H5 ===")
    # build H5 id -> (batch, model_idx)
    f = h5py.File(H5, "r")
    index = {}
    for bk in f.keys():
        cm = f[bk]["CAD_model"][()]
        for i, raw in enumerate(cm):
            pid = raw.decode() if isinstance(raw, bytes) else str(raw)
            index[pid] = (bk, i)

    def h5_label_set(pid):
        bk, mi = index[pid]
        b = f[bk]; idx = np.asarray(b["idx"]); lab = np.asarray(b["labels"]).reshape(-1)
        base = int(idx[0, 0]); s = int(idx[mi, 0]) - base
        e = int(idx[mi + 1, 0]) - base if mi + 1 < len(idx) else lab.shape[0]
        return set(int(x) for x in lab[s:e])

    # MFCAD++ parts are multi-feature, so instead compare the STEP label SET to the
    # released-H5 label set for the SAME id (Jaccard) vs random-id baseline. If the
    # integer->class scheme matches, same-id Jaccard should be far above random.
    all_ids = list(index.keys())
    same_j = []; rand_j = []; seen = 0
    for path in files:
        if len(same_j) >= 60 or seen >= 800:
            break
        seen += 1
        mid = os.path.basename(path)[:-5]
        if mid not in index:
            continue
        try:
            shape, faces, fidx, labels, normals = read_part(path)
        except Exception:
            continue
        sset = set(int(x) for x in labels if x != "")
        hset = h5_label_set(mid)
        if not (sset | hset):
            continue
        same_j.append(len(sset & hset) / len(sset | hset))
        ro = random.choice(all_ids)
        rset = h5_label_set(ro)
        rand_j.append(len(sset & rset) / len(sset | rset))
    same_j = np.array(same_j); rand_j = np.array(rand_j)
    print(f"  parts compared: {same_j.size}")
    print(f"  STEP-vs-H5 label-set Jaccard  SAME id : median={np.median(same_j):.2f} "
          f"mean={same_j.mean():.2f}")
    print(f"  STEP-vs-H5 label-set Jaccard  RANDOM  : median={np.median(rand_j):.2f} "
          f"mean={rand_j.mean():.2f}")
    print(f"  => {'SCHEME ALIGNS (same-id >> random)' if np.median(same_j) > 2*np.median(rand_j)+1e-6 else 'inconclusive'}")


if __name__ == "__main__":
    main()
