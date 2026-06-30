"""
diagnose_step_normals.py — why is the mean-normal dihedral proxy a NO-GO?

Two competing failure modes:
  (A) UPSTREAM: mean-normal pooling washes out per-face orientation, so even a
      true 90-degree rectangular-step corner doesn't read as ~90 degrees.
  (B) HYPOTHESIS WRONG: classes 8 and 10 genuinely don't differ in wall angle here.

Discriminating evidence gathered:
  1. Pooling fidelity on PLANAR faces: a plane has a single constant normal, so its
     facets' normals should have ~0 spread and the mean normal should be a clean unit
     axis vector. If planar faces show large facet-normal spread, decode/pooling is
     broken (mode A).
  2. Axis-alignment: MFCAD++ stock is largely axis-aligned; faithful planar normals
     should cluster near {-1,0,1} components. Smeared components => mode A.
  3. CONCAVE same-label class-8 edges (the actual wall<->floor inner corner of a
     RECTANGULAR step, which is 90 degrees by construction): histogram of the proxy
     angle. Tight peak at 90 => pooling fine, hypothesis suspect (mode B). Broad smear
     => pooling broken (mode A).
  4. A few concrete class-8 instances printed end to end (face normals + angle).

Usage:
    python diag/diagnose_step_normals.py [h5_path] [n_batches]
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
np.set_printoptions(precision=3, suppress=True, linewidth=120)


def canon_set(idx):
    s = set()
    for u, v in idx:
        u, v = int(u), int(v)
        if u != v:
            s.add((u, v) if u < v else (v, u))
    return s


def decode_units(v2):
    n = 2.0 * v2[:, :3].astype(np.float64) - 1.0
    n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-6, None)
    return n


# ---- accumulators ----
plane_facet_spread = []        # per planar face: mean component std of facet normals
plane_meannorm_comp = []       # decoded mean-normal components for planar faces
concave8_angles = []           # angle on concave same-label class-8 edges
v2_col_ranges = []             # per-batch (min,max) of raw V_2 normal cols
printed = 0

with h5py.File(path, "r") as f:
    keys = list(f.keys())[:N_BATCHES]
    for k in keys:
        b = f[k]
        v1 = np.asarray(b["V_1"])
        v2 = np.asarray(b["V_2"])
        a3 = np.asarray(b["A_3_idx"])
        labels = np.asarray(b["labels"]).reshape(-1)
        num_faces = v1.shape[0]
        stype = np.clip(np.round(v1[:, 4] * 11).astype(int) - 1, 0, 5)

        v2_col_ranges.append((v2[:, :3].min(axis=0), v2[:, :3].max(axis=0)))

        units = decode_units(v2)
        face_col = a3[:, 1].astype(np.int64)
        raw_norm = v2[:, :3].astype(np.float64)

        # per-face pooled mean normal (decoded) + facet spread on raw encoding
        mean_n = np.zeros((num_faces, 3))
        counts = np.zeros(num_faces, dtype=np.int64)
        np.add.at(mean_n, face_col, units)
        np.add.at(counts, face_col, 1)
        safe = np.clip(counts, 1, None)
        mean_n = mean_n / safe[:, None]
        mean_n /= np.clip(np.linalg.norm(mean_n, axis=1, keepdims=True), 1e-6, None)

        # planar-face fidelity
        for fc in range(num_faces):
            if stype[fc] != 0 or counts[fc] < 2:
                continue
            m = face_col == fc
            spread = raw_norm[m].std(axis=0).mean()   # spread on the RAW encoding
            plane_facet_spread.append(spread)
            plane_meannorm_comp.append(mean_n[fc])

        # concave same-label class-8 edges
        ca1 = canon_set(np.asarray(b["A_1_idx"]))
        ce2 = canon_set(np.asarray(b["E_2_idx"]))   # concave
        for (u, v) in ca1:
            if labels[u] == 8 and labels[v] == 8 and (u, v) in ce2 \
                    and counts[u] and counts[v]:
                cos = float(np.clip(mean_n[u] @ mean_n[v], -1, 1))
                ang = np.degrees(np.arccos(cos))
                concave8_angles.append(ang)
                if printed < 6:
                    print(f"[class-8 concave edge] faces {u}<->{v}  "
                          f"types=({stype[u]},{stype[v]})  facets=({counts[u]},{counts[v]})")
                    print(f"    mean_n[{u}] = {mean_n[u]}   mean_n[{v}] = {mean_n[v]}")
                    print(f"    angle between mean normals = {ang:.1f} deg\n")
                    printed += 1

print("=" * 72)
print("1. RAW V_2 normal-column ranges per batch (first 5 batches):")
for (mn, mx) in v2_col_ranges[:5]:
    print(f"   min={mn}  max={mx}")
allmn = np.array([r[0] for r in v2_col_ranges])
allmx = np.array([r[1] for r in v2_col_ranges])
print(f"   across {len(v2_col_ranges)} batches: global min={allmn.min(axis=0)} "
      f"max={allmx.max(axis=0)}")
print("   (if every batch is exactly [0,1] per column => per-batch min-max encoded;")
print("    decode 2v-1 is then only exact when the true normals hit the col extremes)")

ps = np.asarray(plane_facet_spread)
print("\n2. PLANAR-face facet-normal spread on RAW encoding (should be ~0 for a plane):")
print(f"   n_planar_faces={ps.size}  mean={ps.mean():.4f}  median={np.median(ps):.4f}  "
      f"p90={np.percentile(ps,90):.4f}  max={ps.max():.4f}")
print("   (large spread => facets of a single flat face disagree => pooling/decoded")
print("    normals unreliable = MODE A upstream issue)")

pc = np.asarray(plane_meannorm_comp)
if pc.size:
    near_axis = np.mean(np.max(np.abs(pc), axis=1) > 0.95)
    print("\n3. PLANAR-face decoded mean-normal axis-alignment:")
    print(f"   fraction with a dominant component |comp|>0.95 = {near_axis:.3f}")
    print(f"   |component| histogram (share in bins 0-.3 .3-.7 .7-.95 .95-1):")
    a = np.abs(pc).reshape(-1)
    bins = [np.mean((a >= lo) & (a < hi)) for lo, hi in
            [(0, .3), (.3, .7), (.7, .95), (.95, 1.01)]]
    print(f"     {bins}")

ca = np.asarray(concave8_angles)
print("\n4. CONCAVE same-label class-8 edges (true RECTANGULAR 90-degree corners):")
if ca.size:
    print(f"   n={ca.size}  mean={ca.mean():.1f}  median={np.median(ca):.1f}  "
          f"std={ca.std():.1f}")
    hist, edges = np.histogram(ca, bins=[0, 30, 60, 75, 85, 95, 105, 120, 180])
    for i in range(len(hist)):
        print(f"     {edges[i]:5.0f}-{edges[i+1]:5.0f} deg : {hist[i]:5d} "
              f"({100*hist[i]/ca.size:4.1f}%)")
    print("   If these 90-degree corners do NOT pile up near 90 => pooling is washing")
    print("   out orientation (MODE A). If they DO pile near 90 but 8 vs 10 still")
    print("   overlap => the wall-angle hypothesis is weak on this data (MODE B).")
