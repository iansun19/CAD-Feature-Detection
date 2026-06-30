"""
recover_per_model_minmax.py — can we invert the (now-confirmed) V_2 encoding?

CONFIRMED encoding (CAPP-SRC/hierarchical-brep-graphs):
  mesh_operations.get_normal(): n = cross(v1,v2)/||.||           # UNIT normal
  generate_batches.normalize_data(): per-COLUMN min-max, PER MODEL:
        raw_c = (n_c - min_c) / (max_c - min_c + 1e-6)
  then per-model arrays are appended into batches.

So n_c = min_c + b_c*raw_c with b_c=(max_c-min_c). min_c/max_c are NOT stored and
differ per model. Per-column scaling with different b_c per axis does NOT preserve
angles, so faithful dihedral angles require recovering (min_c,max_c) per model.

Recovery: unit constraint Σ_c (min_c + b_c raw_c)^2 = 1 is a DIAGONAL quadric (no
cross terms) the model's raw points must lie on. Fit it as the smallest right
singular vector of [r0^2,r1^2,r2^2,r0,r1,r2,1] (per model), then back out min_c,b_c.
A model is RECOVERABLE only if its facet normals are diverse enough (curved faces);
axis-aligned-only stock is degenerate.

Reports: recovery success rate, decode fidelity ||n||~1 on recovered models, and the
concave class-8 (Rect through step) wall-floor angle (must cluster near 90).

Usage: python diag/recover_per_model_minmax.py [h5_path] [n_batches]
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else \
    "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
N_BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 80
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def recover(raw):
    """raw [F,3] -> (mn[3], b[3]) with n_c=mn_c+b_c*raw_c unit, or None if degenerate."""
    F = raw.shape[0]
    if F < 12 or np.unique(np.round(raw, 4), axis=0).shape[0] < 8:
        return None
    r0, r1, r2 = raw[:, 0], raw[:, 1], raw[:, 2]
    M = np.stack([r0 * r0, r1 * r1, r2 * r2, r0, r1, r2, np.ones(F)], axis=1)
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    w = Vt[-1]                                  # [A0,A1,A2,B0,B1,B2,C]
    A = w[0:3]; B = w[3:6]; C = w[6]
    # A_c = λ b_c^2 ; B_c = λ 2 min_c b_c ; C = λ(Σmin_c^2 - 1)
    t = B / (2 * A)                             # = min_c / b_c
    lam = float(np.sum(t * t * A) - C)          # so that b_c^2 = A_c/lam
    if lam == 0:
        return None
    b2 = A / lam
    if np.any(b2 <= 1e-8):
        return None
    b = np.sqrt(b2)
    mn = t * b
    # fix sign: b_c=max-min>0 by construction; SVD sign is arbitrary -> force b>0
    flip = b < 0
    b[flip] = -b[flip]; mn[flip] = -mn[flip]
    return mn, b


def canon_set(idx):
    s = set()
    for u, v in idx:
        u, v = int(u), int(v)
        if u != v:
            s.add((u, v) if u < v else (v, u))
    return s


solved = degen = bad = 0
resid_all = []
concave8 = []
with h5py.File(path, "r") as f:
    for k in list(f.keys())[:N_BATCHES]:
        b = f[k]
        v1 = np.asarray(b["V_1"]); v2 = np.asarray(b["V_2"])
        a3 = np.asarray(b["A_3_idx"]); idx = np.asarray(b["idx"])
        labels = np.asarray(b["labels"]).reshape(-1)
        nf = v1.shape[0]; base = int(idx[0, 0])
        face_col = a3[:, 1].astype(np.int64); facet_col = a3[:, 0].astype(np.int64)
        ca1 = canon_set(np.asarray(b["A_1_idx"])); ce2 = canon_set(np.asarray(b["E_2_idx"]))
        fmean = np.full((nf, 3), np.nan); fcnt = np.zeros(nf, np.int64)
        for mi in range(len(idx)):
            fs = int(idx[mi, 0]) - base
            fe = int(idx[mi + 1, 0]) - base if mi + 1 < len(idx) else nf
            fm = (face_col >= fs) & (face_col < fe)
            if not fm.any():
                continue
            faces = face_col[fm]; raw = v2[facet_col[fm], :3].astype(np.float64)
            sol = recover(raw)
            if sol is None:
                degen += 1
                continue
            mn, bb = sol
            n = mn[None, :] + raw * bb[None, :]
            nlen = np.linalg.norm(n, axis=1)
            resid = np.abs(nlen - 1.0)
            if np.median(resid) > 0.05:
                bad += 1
                continue
            solved += 1
            resid_all.append(resid)
            unit = n / np.clip(nlen[:, None], 1e-9, None)
            for fc in np.unique(faces):
                sel = faces == fc
                m = unit[sel].mean(axis=0)
                fmean[fc] = m / max(np.linalg.norm(m), 1e-9); fcnt[fc] = sel.sum()
        for (u, v) in ca1:
            if labels[u] == 8 and labels[v] == 8 and (u, v) in ce2 and fcnt[u] and fcnt[v]:
                cos = float(np.clip(fmean[u] @ fmean[v], -1, 1))
                concave8.append(np.degrees(np.arccos(cos)))

tot = solved + degen + bad
print(f"file: {path}\nmodels: total={tot}  solved={solved} "
      f"({100*solved/max(tot,1):.1f}%)  degenerate={degen}  bad-fit={bad}\n")
if resid_all:
    ra = np.concatenate(resid_all)
    print(f"decode fidelity on solved models: |‖n‖-1| median={np.median(ra):.4f} "
          f"p90={np.percentile(ra,90):.4f} frac<0.02={np.mean(ra<0.02):.3f}")
ca = np.asarray(concave8)
print(f"\nconcave class-8 wall-floor angle (want tight ~90): n={ca.size}")
if ca.size:
    print(f"  mean={ca.mean():.1f} median={np.median(ca):.1f} std={ca.std():.1f}")
    hist, edges = np.histogram(ca, bins=[0, 30, 60, 80, 85, 95, 100, 120, 180])
    for i in range(len(hist)):
        print(f"   {edges[i]:5.0f}-{edges[i+1]:5.0f}: {hist[i]:5d} ({100*hist[i]/ca.size:4.1f}%)")
