"""probe_pockets.py — one-off diagnostic for the pocket pass.

Q1: do the ⌀0.800 / ⌀0.500 / ⌀3.453 families share axes -> how many pocket axes?
Q2: do the B-spline faces attach (in the graph) to those pocket axes?
Q3: confirm the ⌀0.000 cone faces are accounted for (claimed vs remaining) by the
    hole pass (nothing silently dropped).
"""
from collections import defaultdict

import numpy as np

from feature_params import analyze_step
from hole_detection import detect_holes_from_step

STEP_PATH = "96260B_REAR_XR004_PCD PLATE.stp copy"
GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"
MM_PER_IN = 25.4

faces = analyze_step(STEP_PATH)
by_index = {f.index: f for f in faces}


def axis_key(f, lin_tol=1.0):
    """Sign-normalized axis direction + radial anchor of the centroid."""
    d = np.array(f.axis, float)
    d /= (np.linalg.norm(d) + 1e-12)
    if d[np.argmax(np.abs(d))] < 0:
        d = -d
    c = np.array(f.centroid, float)
    radial = c - np.dot(c, d) * d
    return (tuple(np.round(d, 2)), tuple(np.round(radial / lin_tol) * lin_tol))


def dia_in(f):
    return (2 * f.radius) / MM_PER_IN if f.radius else None


def near(x, target, tol=0.05):
    return x is not None and abs(x - target) < tol


cyl = [f for f in faces if f.surface_type in ("cylinder", "cone") and f.radius]

fam = {
    "0.800 (⌀20.32)": [f for f in cyl if near(dia_in(f), 0.800)],
    "0.500 (⌀12.70)": [f for f in cyl if near(dia_in(f), 0.500)],
    "3.453 (⌀87.716)": [f for f in cyl if near(dia_in(f), 3.453)],
}


def group_by_axis(fs):
    g = {}
    for f in fs:
        g.setdefault(axis_key(f), []).append(f.index)
    return g


print("=== per-family axis grouping ===")
fam_axes = {}
for name, fs in fam.items():
    g = group_by_axis(fs)
    fam_axes[name] = g
    print(f"{name}: {len(fs)} faces -> {len(g)} distinct axes")

print("\n=== shared axes across all three families ===")
all_faces = [f for fs in fam.values() for f in fs]
merged = group_by_axis(all_faces)
print(f"combined -> {len(merged)} distinct axes "
      f"(expect ~7 if each pocket = wall+fillet+spotface on one axis)")

print("\n=== composition of each shared axis ===")
for i, (k, idxs) in enumerate(sorted(merged.items(), key=lambda kv: -len(kv[1]))):
    comp = []
    for name, fs in fam.items():
        n = sum(1 for f in fs if f.index in idxs)
        if n:
            comp.append(f"{name.split()[0]}×{n}")
    print(f"  axis {i}: {len(idxs)} faces  [{', '.join(comp)}]  faces={sorted(idxs)}")

# ---- Q2: do the B-spline faces attach (in the graph) to these axes? ----
print("\n=== B-spline adjacency to pocket axes ===")
data = np.load(GRAPH_NPZ)
ei = data["edge_index"]
adj = defaultdict(set)
for a, b in zip(ei[0], ei[1]):
    adj[int(a)].add(int(b))
    adj[int(b)].add(int(a))

bspline_idx = [f.index for f in faces if f.surface_type in ("bspline", "bezier")]
print(f"{len(bspline_idx)} bspline/bezier faces total")

axis_facesets = {i: set(idxs) for i, (k, idxs) in enumerate(merged.items())}
attached = 0
attach_count_per_axis = defaultdict(int)
for bi in bspline_idx:
    hits = [i for i, fs in axis_facesets.items() if adj[bi] & fs]
    if hits:
        attached += 1
        for h in hits:
            attach_count_per_axis[h] += 1
print(f"bsplines adjacent to at least one pocket-axis cluster: "
      f"{attached}/{len(bspline_idx)}")
print(f"bsplines per axis: {dict(attach_count_per_axis)}")

# ---- Q3: confirm the ⌀0.000 cones' fate via the actual hole pass ----
print("\n=== ⌀0.000 cone faces ===")
zero_cones = [
    f.index for f in faces
    if f.surface_type == "cone" and (not f.radius or abs(f.radius) < 1e-6)
]
print(f"⌀0.000 cones: {zero_cones}")

edge_attr = data["edge_attr"]
res = detect_holes_from_step(STEP_PATH, ei, edge_attr)
in_claimed = [i for i in zero_cones if i in res.claimed_faces]
in_remaining = [i for i in zero_cones if i in res.remaining_faces]
missing = [
    i for i in zero_cones
    if i not in res.claimed_faces and i not in res.remaining_faces
]
print(f"zero-cones claimed  : {in_claimed}")
print(f"zero-cones remaining: {in_remaining}")
print(f"zero-cones MISSING  : {missing}  (must be empty)")
