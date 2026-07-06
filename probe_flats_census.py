"""probe_flats_census.py — census planar faces in the flats-pass residual.

Run: /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_flats_census.py
"""
from collections import defaultdict

import numpy as np

from pathlib import Path

from feature_params import analyze_step
from run_cascade import _load_edges, run_cascade

STEP_PATH = "96260B_REAR_XR004_PCD PLATE.stp copy"
GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"
MM_PER_IN = 25.4

faces = analyze_step(STEP_PATH)
by_index = {f.index: f for f in faces}

edge_index, edge_attr = _load_edges(Path(GRAPH_NPZ), Path(STEP_PATH))
_, pk, hl, _, _, _ = run_cascade(STEP_PATH, edge_index, edge_attr)
residual = set(int(i) for i in hl.remaining_faces)
print(f"residual feeding flats: {len(residual)} faces")

planes = [by_index[i] for i in residual if by_index[i].surface_type == "plane"]
print(f"planar faces in residual: {len(planes)}")

hist = defaultdict(int)
for i in residual:
    hist[by_index[i].surface_type] += 1
print("residual surface-type histogram:", dict(hist))


def plane_key(f, ang_round=2, dist_tol=0.5):
    n = np.array(f.normal, float)
    n /= np.linalg.norm(n) + 1e-12
    if n[np.argmax(np.abs(n))] < 0:
        n = -n
    d = float(np.dot(np.array(f.centroid, float), n))
    return (tuple(np.round(n, ang_round)), round(d / dist_tol) * dist_tol)


groups = defaultdict(list)
for f in planes:
    groups[plane_key(f)].append(f.index)

print(f"\ndistinct plane equations: {len(groups)}  (Toolpath 'faces' target = 8)")
for i, (k, idxs) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
    n, d = k
    total_area = sum(by_index[j].area for j in idxs)
    print(
        f"  group {i}: {len(idxs)} face(s)  normal~{n}  offset~{d:+.2f}mm  "
        f"area={total_area:.1f}  faces={sorted(idxs)}"
    )

print("\ntolerance sensitivity (distinct plane count):")
for dt in (0.1, 0.25, 0.5, 1.0, 2.0):
    for at in (2, 3):
        g = defaultdict(list)
        for f in planes:
            g[plane_key(f, ang_round=at, dist_tol=dt)].append(f.index)
        print(f"  dist_tol={dt:>4} ang_round={at}: {len(g)} groups")

# leak check: pocket/hole claimed planes that leaked into residual
claimed_planes = [
    i
    for i in (pk.claimed_faces | hl.claimed_faces)
    if by_index[i].surface_type == "plane"
]
leaked_claimed = sorted(i for i in claimed_planes if i in residual)
if leaked_claimed:
    print(f"\nWARNING: {len(leaked_claimed)} claimed planes leaked into residual: {leaked_claimed}")
else:
    print("\nclaimed-plane leak check: OK (no pocket/hole floor planes in residual)")
