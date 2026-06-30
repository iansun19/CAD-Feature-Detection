"""
probe_v2_encoding.py — characterize the REAL V_2 normal encoding empirically.

Prints:
  * all HDF5 .attrs at file / batch-group / dataset level (overlooked metadata?)
  * for several PLANAR faces (surface_type==plane) with many facets: the raw V_2
    rows, their per-facet spread, and what 2v-1 would give (to show it's wrong).
  * per-column min/max/quantiles of V_2[:, :3] within a batch, and how often the
    exact extremes (0.0 / 1.0) occur (per-batch min-max leaves exactly one facet at
    each extreme per column?).
  * the set of distinct col-0 values that equal exactly 0.5 etc. — looking for a
    "component==0 -> 0.5" fixed point that pins f(0).
  * facets whose raw vector hits an extreme (0 or 1) in some column — these have a
    known extreme true component and calibrate the scaling.

Usage:
    python diag/probe_v2_encoding.py [h5_path] [batch_key]
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
BK = sys.argv[2] if len(sys.argv) > 2 else "0"
np.set_printoptions(precision=5, suppress=True, linewidth=140)

with h5py.File(path, "r") as f:
    print("=== file-level .attrs ===")
    print(dict(f.attrs) or "  (none)")
    b = f[BK]
    print(f"\n=== group '{BK}' .attrs ===")
    print(dict(b.attrs) or "  (none)")
    print("\n=== per-dataset .attrs ===")
    for k in sorted(b.keys()):
        a = dict(b[k].attrs)
        if a:
            print(f"  {k}: {a}")
    print("  (no per-dataset attrs printed above means none present)")

    v1 = np.asarray(b["V_1"])
    v2 = np.asarray(b["V_2"])
    a3 = np.asarray(b["A_3_idx"])
    face_col = a3[:, 1].astype(np.int64)
    stype = np.clip(np.round(v1[:, 4] * 11).astype(int) - 1, 0, 5)
    counts = np.bincount(face_col, minlength=v1.shape[0])

    n3 = v2[:, :3].astype(np.float64)
    print("\n=== V_2[:, :3] per-column stats (this batch) ===")
    for c in range(3):
        col = n3[:, c]
        print(f"  col{c}: min={col.min():.4f} max={col.max():.4f} "
              f"mean={col.mean():.4f} median={np.median(col):.4f} "
              f"q05={np.percentile(col,5):.4f} q95={np.percentile(col,95):.4f}")
        print(f"        #==0.0: {(col==0).sum()}  #==1.0: {(col==1).sum()}  "
              f"#==0.5: {(col==0.5).sum()}")

    print("\n=== a few PLANAR faces: raw V_2 rows + 2v-1 decode ===")
    planar = [fc for fc in range(v1.shape[0]) if stype[fc] == 0 and counts[fc] >= 4]
    planar.sort(key=lambda fc: counts[fc], reverse=True)
    for fc in planar[:5]:
        m = face_col == fc
        rows = v2[m]
        raw = rows[:, :3].astype(np.float64)
        dec = 2 * raw - 1
        decn = np.linalg.norm(dec, axis=1)
        print(f"\n  face {fc}: planar, {m.sum()} facets, "
              f"V_1=[area={v1[fc,0]:.3f} c=({v1[fc,1]:.3f},{v1[fc,2]:.3f},{v1[fc,3]:.3f})]")
        print(f"    raw[:4]   =\n{raw[:4]}")
        print(f"    col std over facets = {raw.std(axis=0)}  (≈0 => constant normal)")
        print(f"    mean raw  = {raw.mean(axis=0)}")
        print(f"    2*meanraw-1 = {2*raw.mean(axis=0)-1}  ||.||={np.linalg.norm(2*raw.mean(axis=0)-1):.4f}")
        print(f"    col3(d) over facets: min={rows[:,3].min():.3f} max={rows[:,3].max():.3f}")

    print("\n=== facets at a column extreme (raw==0 or ==1) — calibration anchors ===")
    for c in range(3):
        for tgt, lab in [(0.0, "min"), (1.0, "max")]:
            idxs = np.where(n3[:, c] == tgt)[0]
            if idxs.size:
                r = v2[idxs[0], :3]
                print(f"  col{c}=={lab}(={tgt}): {idxs.size} facets; "
                      f"example raw vec = {r}")
