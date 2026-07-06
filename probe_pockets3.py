"""probe_pockets3.py — what do the 42 B-spline floors actually touch in the graph?"""
from collections import Counter, defaultdict

import numpy as np

from feature_params import analyze_step

STEP_PATH = "96260B_REAR_XR004_PCD PLATE.stp copy"
GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"

faces = analyze_step(STEP_PATH)
st_of = {f.index: f.surface_type for f in faces}

data = np.load(GRAPH_NPZ)
ei = data["edge_index"]
adj = defaultdict(set)
for a, b in zip(ei[0], ei[1]):
    adj[int(a)].add(int(b))
    adj[int(b)].add(int(a))

bspline_idx = [f.index for f in faces if f.surface_type in ("bspline", "bezier")]

nbr_types = Counter()
deg = []
for bi in bspline_idx:
    deg.append(len(adj[bi]))
    for nb in adj[bi]:
        nbr_types[st_of.get(nb, "?")] += 1

print(f"{len(bspline_idx)} bspline faces; degree min/mean/max = "
      f"{min(deg)}/{np.mean(deg):.1f}/{max(deg)}")
print(f"neighbor surface-type histogram: {dict(nbr_types)}")

# how many bsplines are adjacent to another bspline (floors chaining together)?
bset = set(bspline_idx)
n_touch_bspline = sum(1 for bi in bspline_idx if adj[bi] & bset)
print(f"bsplines adjacent to >=1 other bspline: {n_touch_bspline}/{len(bspline_idx)}")

# connected components among bsplines only (do the 42 floors form ~N pocket floors?)
from hole_detection import _UnionFind
uf = _UnionFind(bspline_idx)
for bi in bspline_idx:
    for nb in adj[bi] & bset:
        uf.union(bi, nb)
groups = uf.groups()
print(f"bspline-only connected components: {len(groups)} "
      f"(sizes: {sorted((len(g) for g in groups), reverse=True)})")
