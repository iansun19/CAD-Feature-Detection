"""
final_report.py — validation report over the FULL regenerated dataset.
Per split: model/face counts, per-class label coverage, dihedral (A_1_values)
distribution across all edges, and the concave class-8 90-degree anchor.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/final_report.py
"""
import os
import sys
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brep.taxonomy import NUM_CLASSES

DIR = "MFCAD++_dataset/hierarchical_graphs_regen"
SPLITS = [("train", "training_MFCAD++.h5"), ("val", "val_MFCAD++.h5"),
          ("test", "test_MFCAD++.h5")]


def canon(u, v):
    return (u, v) if u < v else (v, u)


def main():
    for split, fn in SPLITS:
        f = h5py.File(f"{DIR}/{fn}", "r")
        groups = list(f.keys())
        nmodels = nfaces = 0
        labhist = np.zeros(NUM_CLASSES, np.int64)
        ang_hist = np.zeros(7, np.int64)            # 7 bins below
        ang_n = 0; nan_like = 0
        c8 = []
        bins = [0, 30, 60, 80, 95, 120, 150, 181]
        for bk in groups:
            b = f[bk]
            idx = b["idx"][()]; lab = b["labels"][()].astype(int)
            nmodels += len(idx); nfaces += b["V_1"].shape[0]
            for c in lab:
                if 0 <= c < NUM_CLASSES:
                    labhist[c] += 1
            AV = b["A_1_values"][()]
            deg = np.degrees(AV)
            ang_n += deg.size
            h, _ = np.histogram(deg, bins=bins); ang_hist += h
            # concave class-8 anchor: edges in E_2 between two class-8 faces
            A1 = b["A_1_idx"][()]
            E2 = set(map(tuple, (canon(*r) for r in b["E_2_idx"][()].tolist())))
            # need per-model base offsets to map labels; idx col0 cumulative (base 0)
            base = int(idx[0, 0])
            starts = idx[:, 0] - base
            # global label array for the batch is lab (already concatenated)
            for (i, j), a in zip(A1.tolist(), deg.tolist()):
                if lab[i] == 8 and lab[j] == 8 and canon(i, j) in E2:
                    c8.append(a)
        c8 = np.array(c8)
        print(f"\n===== {split} =====")
        print(f"models={nmodels} faces={nfaces} groups={len(groups)}")
        miss = [c for c in range(NUM_CLASSES) if labhist[c] == 0]
        print(f"label classes present: {NUM_CLASSES - len(miss)}/{NUM_CLASSES}" +
              (f"  MISSING {miss}" if miss else "  (all present)"))
        print(f"dihedral over all {ang_n} edges:")
        names = ["0-30", "30-60", "60-80", "80-95", "95-120", "120-150", "150-180"]
        for k in range(7):
            print(f"   {names[k]:>8} deg: {ang_hist[k]:8d} ({100*ang_hist[k]/ang_n:5.1f}%)")
        if c8.size:
            print(f"concave class-8 anchor: n={c8.size} median={np.median(c8):.1f} "
                  f"std={c8.std():.2f} in80-95={100*np.mean((c8>=80)&(c8<=95)):.1f}%")


if __name__ == "__main__":
    main()
