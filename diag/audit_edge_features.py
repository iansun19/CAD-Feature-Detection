"""
audit_edge_features.py — Step 1 audit for adding a real dihedral-angle edge feature.

Answers, strictly from the data (NOT the docs):

  1. EDGE CONSTRUCTION: what does the graph edge set (A_1) look like? Is it directed
     (both (u,v) and (v,u) present) or effectively undirected? Are A_1 / E_* / A_3
     "_values" arrays constant 1.0 (pure adjacency indicators) or do they carry signal?
  2. CONVEXITY DERIVATION: A_1 edges are bucketed by membership in E_1 (convex) /
     E_2 (concave) / E_3 (smooth). How many A_1 edges fall in none of them (the
     "default convex" branch)? E_* is a 3-way sign-of-dihedral bucket, so the new
     continuous angle partially subsumes it — quantify the overlap.
  3. CONNECTIVITY: are the wall<->floor faces of the step features actually connected
     by an A_1 edge? Confirmed by counting same-label adjacent face pairs for the
     collapsed step-family classes (3 through_step, 8 blind_step).
  4. GO/NO-GO: for same-label step edges, compute the angle between the two faces'
     pooled mean normals (decoded exactly as pool_facet_features_to_faces does) and
     report mean/median/std per class. Under 12-class collapse, old rectangular
     through step (8) vs slanted through step (10) separation is NO LONGER testable
     — both map to new class 3.

Local .venv has only h5py + numpy (no torch), so pooling logic is replicated inline
rather than imported from dataset.py.

Usage:
    python diag/audit_edge_features.py [h5_path] [n_batches]
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "MFCAD++_dataset/hierarchical_graphs_regen_12/training_MFCAD++.h5"
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 150
np.set_printoptions(precision=4, suppress=True, linewidth=120)

# 12-class collapsed ids. Old 8/9/10 (rect/2-sided/slanted through step) merged -> 3.
LABELS = {
    3: "through_step (was old 8/9/10)",
    8: "blind_step (was old 22)",
}
STEP_CLASSES = [3, 8]


def canon_set(idx):
    """Undirected, self-loop-free canonical edge set from an [E,2] idx array."""
    s = set()
    for u, v in idx:
        u, v = int(u), int(v)
        if u == v:
            continue
        s.add((u, v) if u < v else (v, u))
    return s


def pool_mean_normals(v2, a3, num_faces):
    """Decode [0,1]->[-1,1], L2-normalize per facet, mean per face, renormalize.

    Identical math to dataset.pool_facet_features_to_faces (normal block only).
    Returns (mean_normals[num_faces,3], facet_counts[num_faces]).
    """
    facet = a3[:, 0].astype(np.int64)
    face = a3[:, 1].astype(np.int64)
    n = 2.0 * v2[facet, :3].astype(np.float64) - 1.0
    n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-6, None)
    sum_n = np.zeros((num_faces, 3), dtype=np.float64)
    counts = np.zeros(num_faces, dtype=np.int64)
    np.add.at(sum_n, face, n)
    np.add.at(counts, face, 1)
    safe = np.clip(counts, 1, None).astype(np.float64)
    mean = sum_n / safe[:, None]
    mean /= np.clip(np.linalg.norm(mean, axis=1, keepdims=True), 1e-6, None)
    mean[counts == 0] = 0.0
    return mean, counts


def summ(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return "n=0"
    return (f"n={a.size:6d}  mean={a.mean():7.2f}  median={np.median(a):7.2f}  "
            f"std={a.std():6.2f}  p10={np.percentile(a,10):6.2f}  "
            f"p90={np.percentile(a,90):6.2f}")


# accumulators
val_const = {k: set() for k in ("A_1_values", "E_1_values", "E_2_values",
                                "E_3_values", "A_3_values")}
dir_sym_num = dir_sym_den = 0          # symmetry of A_1
a1_total = a1_in_e1 = a1_in_e2 = a1_in_e3 = a1_in_none = 0
self_loops = 0
samelabel_adj = {c: 0 for c in STEP_CLASSES}          # connectivity counts
angles_by_class = {c: [] for c in STEP_CLASSES}        # same-label edge angles
conv_by_class = {c: {"convex": 0, "concave": 0, "smooth": 0, "none": 0}
                 for c in STEP_CLASSES}
a1_idx_oob = False
n_batches_done = 0

with h5py.File(path, "r") as f:
    keys = list(f.keys())[:N_BATCHES]
    for k in keys:
        b = f[k]
        v1 = np.asarray(b["V_1"])
        v2 = np.asarray(b["V_2"])
        a1 = np.asarray(b["A_1_idx"])
        a3 = np.asarray(b["A_3_idx"])
        labels = np.asarray(b["labels"]).reshape(-1)
        num_faces = v1.shape[0]

        # --- _values constants ---
        for vk in val_const:
            if vk in b:
                vals = np.asarray(b[vk][()]).reshape(-1)
                if vals.size:
                    val_const[vk].add(round(float(vals.min()), 6))
                    val_const[vk].add(round(float(vals.max()), 6))

        # --- A_1 0-based-within-batch sanity ---
        if a1.size and (a1.max() >= num_faces or a1.min() < 0):
            a1_idx_oob = True

        # --- directedness of A_1 (raw, before canonicalizing) ---
        raw = set()
        for u, v in a1:
            u, v = int(u), int(v)
            if u == v:
                self_loops += 1
                continue
            raw.add((u, v))
        for (u, v) in raw:
            dir_sym_den += 1
            if (v, u) in raw:
                dir_sym_num += 1

        # --- A_1 vs E_* overlap (canonical, undirected) ---
        ca1 = canon_set(a1)
        ce1 = canon_set(np.asarray(b["E_1_idx"]))
        ce2 = canon_set(np.asarray(b["E_2_idx"]))
        ce3 = canon_set(np.asarray(b["E_3_idx"]))
        a1_total += len(ca1)
        a1_in_e1 += len(ca1 & ce1)
        a1_in_e2 += len(ca1 & ce2)
        a1_in_e3 += len(ca1 & ce3)
        a1_in_none += len(ca1 - ce1 - ce2 - ce3)

        # --- mean normals for this batch ---
        mean_n, counts = pool_mean_normals(v2, a3, num_faces)

        # --- same-label step edges: connectivity + angle + convexity ---
        for (u, v) in ca1:
            lu, lv = int(labels[u]), int(labels[v])
            if lu != lv or lu not in samelabel_adj:
                continue
            c = lu
            samelabel_adj[c] += 1
            # convexity bucket (same precedence as dataset.py)
            if (u, v) in ce1:
                conv_by_class[c]["convex"] += 1
            elif (u, v) in ce2:
                conv_by_class[c]["concave"] += 1
            elif (u, v) in ce3:
                conv_by_class[c]["smooth"] += 1
            else:
                conv_by_class[c]["none"] += 1
            # angle between mean normals (skip zero-facet faces)
            if counts[u] == 0 or counts[v] == 0:
                continue
            cos = float(np.clip(mean_n[u] @ mean_n[v], -1.0, 1.0))
            angles_by_class[c].append(np.degrees(np.arccos(cos)))
        n_batches_done += 1

# ----------------------------------------------------------------------------
print(f"file: {path}")
print(f"batches scanned: {n_batches_done}\n")

print("=== 1a. edge '_values' arrays: constant? (set of observed min/max) ===")
for vk, vals in val_const.items():
    sv = sorted(vals)
    const = "CONSTANT" if len(sv) == 1 else "varies"
    print(f"  {vk:12s} observed values across batches = {sv}   -> {const}")

print("\n=== 1b. A_1 self-loops & directedness ===")
print(f"  self-loops removed: {self_loops}")
print(f"  A_1 0-based-within-batch (no OOB index): {not a1_idx_oob}")
if dir_sym_den:
    frac = dir_sym_num / dir_sym_den
    print(f"  directed pairs whose reverse also present: {dir_sym_num}/{dir_sym_den} "
          f"= {frac:.3f}")
    print("  -> A_1 is " + ("effectively UNDIRECTED (both directions stored)"
                            if frac > 0.99 else
                            "DIRECTED / asymmetric (sign convention matters)"))

print("\n=== 2. A_1 vs E_1/E_2/E_3 (convexity) overlap, canonical undirected ===")
print(f"  total A_1 edges      : {a1_total}")
if a1_total:
    print(f"  in E_1 (convex)      : {a1_in_e1:8d}  ({100*a1_in_e1/a1_total:5.1f}%)")
    print(f"  in E_2 (concave)     : {a1_in_e2:8d}  ({100*a1_in_e2/a1_total:5.1f}%)")
    print(f"  in E_3 (smooth)      : {a1_in_e3:8d}  ({100*a1_in_e3/a1_total:5.1f}%)")
    print(f"  in NONE (default cvx): {a1_in_none:8d}  ({100*a1_in_none/a1_total:5.1f}%)")
print("  NOTE: E_1/E_2/E_3 = convex/concave/smooth = a 3-way SIGN-of-dihedral bucket.")
print("        A continuous angle feature partially subsumes this (flag, don't drop).")

print("\n=== 3. connectivity: same-label adjacent face pairs (step family) ===")
for c in STEP_CLASSES:
    print(f"  class {c:2d} {LABELS[c]:22s}: {samelabel_adj[c]:6d} same-label A_1 edges")
print("  (>0 confirms the wall<->floor pair IS a graph edge under A_1)")

print("\n=== 4. same-label step edge angles (mean normals) ===")
for c in STEP_CLASSES:
    print(f"  class {c:2d} {LABELS[c]:22s}: {summ(angles_by_class[c])}")

print("\n  GRANULARITY LOST: old class 8 (rect through step) vs 10 (slanted) both "
      "collapsed to new class 3 — the former 8-vs-10 GO/NO-GO separation test is "
      "no longer applicable.")

print("\n=== 5. convexity-bucket distribution on same-label step edges (redundancy) ===")
for c in STEP_CLASSES:
    d = conv_by_class[c]
    tot = sum(d.values()) or 1
    print(f"  class {c:2d}: convex={d['convex']:5d} concave={d['concave']:5d} "
          f"smooth={d['smooth']:5d} none={d['none']:5d}  "
          f"(conv {100*d['convex']/tot:.0f}% / conc {100*d['concave']/tot:.0f}%)")
