"""One-off: compute the 12 canonical template faces (TRAIN split only).

For each class: mean 14-dim node-feature vector (centroid) over ALL training
faces of that class, then the training face nearest (L2) to that centroid.
Prints (part_id, face_index, class_id) tuples to hard-code into llm_baseline.py.
"""
import os
import numpy as np

from legacy.llm_baseline import load_cfg, open_regen_split
from legacy.dataset import _brep_bounds, build_node_features_regen

cfg = load_cfg()
C = cfg["num_classes"]
nst = cfg["num_surface_types"]

h5, index = open_regen_split(cfg, "train.txt")
with open(os.path.join(cfg["data_root"], "train.txt")) as f:
    train_ids = [ln.strip() for ln in f if ln.strip() and ln.strip() in index]

# collect every training face's 14-dim feature + (pid, face_idx, label)
feats_by_cls = {c: [] for c in range(C)}
meta_by_cls = {c: [] for c in range(C)}
for pid in train_ids:
    bk, mi = index[pid]
    batch = h5[bk]
    s, e = _brep_bounds(batch["idx"][()], mi, batch["V_1"].shape[0])
    v1 = np.asarray(batch["V_1"][s:e], dtype=np.float32)
    x = build_node_features_regen(v1, nst)
    labels = np.asarray(batch["labels"][s:e], dtype=np.int64)
    for fi in range(len(labels)):
        c = int(labels[fi])
        feats_by_cls[c].append(x[fi])
        meta_by_cls[c].append((pid, fi))

print("class  n_faces   nearest (pid, face_idx)   dist")
selected = []
for c in range(C):
    if not feats_by_cls[c]:
        print(f"{c:>5}  MISSING")
        selected.append(None)
        continue
    X = np.stack(feats_by_cls[c])
    centroid = X.mean(axis=0)
    d = np.linalg.norm(X - centroid, axis=1)
    j = int(np.argmin(d))
    pid, fi = meta_by_cls[c][j]
    selected.append((pid, fi, c))
    print(f"{c:>5}  {len(X):>7}   ({pid}, {fi})   {d[j]:.4f}")

print("\nTEMPLATE_FACES = [")
for t in selected:
    print(f"    {t},")
print("]")
h5.close()
