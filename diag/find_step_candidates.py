"""
find_step_candidates.py — pick parts with concave same-label class-8 face pairs.

Class 8 = "Rectangular through step". Its wall-floor inner corner is a CONCAVE edge
(E_2) joining two faces that are BOTH labelled 8. That corner is 90 deg by
construction -> our validation anchor.

For each chosen model we record, in a JSON the STEP probe can consume:
  - model_id (CAD_model name -> step/<split>/<id>.step) and split
  - the concave class-8 face pairs (H5 batch-LOCAL, 0-based face indices)
  - per-face V_1 features [area, cx, cy, cz, surface_type_id] for STEP<->H5 matching
  - whether the STEP file exists on disk

Also reports whether V_1[:, :4] looks per-model min-max normalized (so the STEP probe
knows to normalize STEP centroids/areas the same way before matching).

Run in the .venv (h5py + numpy only):
    python diag/find_step_candidates.py
"""
import json
import os
import h5py
import numpy as np

H5 = "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
SPLIT = "train"
STEP_DIR = f"MFCAD++_dataset/step/{SPLIT}"
OUT = "diag/step_candidates.json"
N_WANT = 8
MIN_PAIRS = 3          # want parts with multiple concave class-8 pairs


def canon(u, v):
    return (int(u), int(v)) if u < v else (int(v), int(u))


def edge_set_local(idx, start, end):
    m = ((idx[:, 0] >= start) & (idx[:, 0] < end) &
         (idx[:, 1] >= start) & (idx[:, 1] < end))
    return {canon(*(row - start)) for row in idx[m]}


cands = []
norm_report = []
with h5py.File(H5, "r") as f:
    for bk in f.keys():
        b = f[bk]
        idx = np.asarray(b["idx"])
        v1_all = np.asarray(b["V_1"])
        labels_all = np.asarray(b["labels"]).reshape(-1)
        a1 = np.asarray(b["A_1_idx"])
        e2 = np.asarray(b["E_2_idx"])
        names = b["CAD_model"][()]
        base = int(idx[0, 0])
        nrows = v1_all.shape[0]
        for mi in range(len(idx)):
            s = int(idx[mi, 0]) - base
            e = int(idx[mi + 1, 0]) - base if mi + 1 < len(idx) else nrows
            lab = labels_all[s:e]
            if not np.any(lab == 8):
                continue
            a1_set = edge_set_local(a1, s, e)
            e2_set = edge_set_local(e2, s, e)
            pairs = [[u, v] for (u, v) in (a1_set & e2_set)
                     if lab[u] == 8 and lab[v] == 8]
            if len(pairs) < MIN_PAIRS:
                continue
            mid = names[mi].decode() if isinstance(names[mi], bytes) else str(names[mi])
            step_path = os.path.join(STEP_DIR, f"{mid}.step")
            v1 = v1_all[s:e]
            stype = np.clip(np.round(v1[:, 4] * 11).astype(int) - 1, 0, 5)
            if len(norm_report) < 5:
                norm_report.append((mid, v1[:, :4].min(0).tolist(), v1[:, :4].max(0).tolist()))
            cands.append({
                "model_id": mid,
                "split": SPLIT,
                "batch": bk,
                "model_idx": mi,
                "num_faces": int(e - s),
                "step_exists": os.path.isfile(step_path),
                "concave_class8_pairs": pairs,
                "v1": v1[:, :5].astype(float).tolist(),   # [area,cx,cy,cz,typecode]
                "surface_type_id": stype.astype(int).tolist(),
            })
        if len(cands) >= N_WANT * 4:
            break

cands.sort(key=lambda c: len(c["concave_class8_pairs"]), reverse=True)
chosen = cands[:N_WANT]

print(f"found {len(cands)} candidate models with >= {MIN_PAIRS} concave class-8 pairs; "
      f"keeping {len(chosen)}\n")
print("V_1[:, :4] per-model ranges (expect [0,1] if min-max normalized):")
for mid, mn, mx in norm_report:
    print(f"  {mid}: min={np.round(mn,3)}  max={np.round(mx,3)}")
print()
for c in chosen:
    print(f"  model {c['model_id']:>7} ({c['split']}): faces={c['num_faces']:3d}  "
          f"concave-class8-pairs={len(c['concave_class8_pairs'])}  "
          f"step_exists={c['step_exists']}")

with open(OUT, "w") as fo:
    json.dump(chosen, fo)
print(f"\nwrote {OUT}")
missing = [c["model_id"] for c in chosen if not c["step_exists"]]
if missing:
    print(f"WARNING: STEP files missing for: {missing}")
