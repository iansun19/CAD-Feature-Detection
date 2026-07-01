"""
validate_regen_h5.py — verify the regenerated H5 is loader-compatible and correct,
without needing torch. Replicates dataset.py's B-rep reads.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/validate_regen_h5.py [path]
"""
import sys
import numpy as np
import h5py

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "MFCAD++_dataset/hierarchical_graphs_regen_12/val_MFCAD++.h5"


def canon(u, v):
    return (int(u), int(v)) if u < v else (int(v), int(u))


def edge_set(idx, s, e):
    m = ((idx[:, 0] >= s) & (idx[:, 0] < e) & (idx[:, 1] >= s) & (idx[:, 1] < e))
    return idx[m] - s


def main():
    f = h5py.File(PATH, "r")
    # build index like dataset._build_index
    index = {}
    for bk in f.keys():
        for i, raw in enumerate(f[bk]["CAD_model"][()]):
            pid = raw.decode() if isinstance(raw, bytes) else str(raw)
            index[pid] = (bk, i)
    print(f"file={PATH}")
    print(f"models indexed={len(index)} groups={len(list(f.keys()))}")

    issues = 0
    v1min = np.array([9, 9, 9, 9.]); v1max = -v1min
    lab_seen = set()
    class3_ok = []; class3_all = []
    checked = 0
    for pid, (bk, mi) in index.items():
        b = f[bk]
        idx = b["idx"][()]; V1 = b["V_1"][()]; lab = b["labels"][()].astype(int)
        base = int(idx[0, 0]); s = int(idx[mi, 0]) - base
        e = int(idx[mi + 1, 0]) - base if mi + 1 < len(idx) else V1.shape[0]
        n = e - s
        v1 = V1[s:e]; y = lab[s:e]
        # checks
        if v1.shape[1] != 9:
            issues += 1
        if n != len(y):
            print(f"  {pid}: node/label mismatch"); issues += 1
        if (y < 0).any() or (y > 11).any():
            print(f"  {pid}: label out of range"); issues += 1
        lab_seen |= set(y.tolist())
        v1min = np.minimum(v1min, v1[:, :4].min(0)); v1max = np.maximum(v1max, v1[:, :4].max(0))

        a1 = edge_set(b["A_1_idx"][()], s, e)
        # endpoints valid
        if a1.size and (a1.min() < 0 or a1.max() >= n):
            print(f"  {pid}: A_1 endpoint OOB"); issues += 1
        # symmetric (both directions)
        sset = set(map(tuple, a1.tolist()))
        if not all((j, i) in sset for (i, j) in sset):
            print(f"  {pid}: A_1 not symmetric"); issues += 1
        # buckets partition A_1 (canonical)
        e1 = {canon(*r) for r in edge_set(b["E_1_idx"][()], s, e)}
        e2 = {canon(*r) for r in edge_set(b["E_2_idx"][()], s, e)}
        e3 = {canon(*r) for r in edge_set(b["E_3_idx"][()], s, e)}
        acanon = {canon(*r) for r in a1.tolist()}
        # every non-selfloop A_1 canon edge should be in exactly one bucket
        miss = [ed for ed in acanon if ed[0] != ed[1]
                and (ed in e1) + (ed in e2) + (ed in e3) != 1]
        if miss:
            print(f"  {pid}: {len(miss)} A_1 edges not in exactly one bucket"); issues += 1

        # geometric: concave (E_2) edges between two through_step (class 3) faces ~ 90 deg
        if 3 in y:
            A1 = b["A_1_idx"][()]; AV = b["A_1_values"][()]
            m = ((A1[:, 0] >= s) & (A1[:, 0] < e) & (A1[:, 1] >= s) & (A1[:, 1] < e))
            rows = A1[m] - s; vals = AV[m]
            for (i, j), ang in zip(rows.tolist(), vals.tolist()):
                if y[i] == 3 and y[j] == 3 and canon(i, j) in e2:
                    deg = np.degrees(ang)
                    class3_all.append(deg)
                    if 80 <= deg <= 95:
                        class3_ok.append(deg)
        checked += 1

    print(f"\nchecked {checked} models; structural issues={issues}")
    print(f"V_1[:, :4] global min={np.round(v1min,3)} max={np.round(v1max,3)} "
          f"(expect ~0 and ~1)")
    print(f"label values seen: {sorted(lab_seen)}")
    if class3_all:
        a = np.array(class3_all)
        print(f"concave through_step (class 3) pairs (A_1_values): n={a.size} "
              f"median={np.median(a):.1f} std={a.std():.1f} "
              f"in80-95={100*len(class3_ok)/a.size:.0f}%")
    print("\nVERDICT:", "PASS" if issues == 0 else f"{issues} ISSUES")


if __name__ == "__main__":
    main()
