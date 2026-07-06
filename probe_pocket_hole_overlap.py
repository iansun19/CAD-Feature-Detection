"""probe_pocket_hole_overlap.py — STEP 0 verification probe for the cascade REORDER.

Why this exists
---------------
We are about to reorder the cascade so POCKETS run BEFORE HOLES (the more-specific
recognizer claims first; holes take the residual). Before writing the pocket pass,
we must prove that running pocket (X,Z) proximity clustering on the FULL candidate
pool — *before* any hole claiming — does NOT sweep the two genuine central holes
into a pocket cluster. If a central hole leaked into a pocket cluster, option-1
ordering would have its own ownership leak and we would fix the candidate filter
first.

The two genuine central holes (Toolpath ground truth, confirmed by hole_detection)
  * blind   ⌀101.752 mm (4.006 in): wall faces 110, 335 (two half-arcs -> one hole)
  * through ⌀81.280  mm (3.200 in): wall faces 329, 347 (two half-arcs -> one hole)
Both open along +Y and sit spatially CENTRAL, inboard of the 7 pocket lobes.

What the probe does
-------------------
1. Discovers the central-hole faces empirically (by diameter family), not hardcoded.
2. Builds the pocket candidate pool (wall families ⌀0.800/0.500/3.453 in +
   bspline/bezier floors + spheres) and detects the opening axis generically.
3. Runs (X,Z) single-linkage proximity clustering to K=7 pockets.
4. For every central-hole wall face, checks BOTH:
     (a) membership — is the face a member of any of the 7 clusters?
     (b) spatial footprint — is the face inside any cluster's (X,Z) footprint
         (distance to that cluster's centroid <= its max intra-cluster spread)?
5. Asserts the central-hole faces fall in NONE of the 7 pocket clusters, and prints
   the nearest-cluster assignment (with distance vs footprint) for each.

Run: /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_pocket_hole_overlap.py
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from feature_params import analyze_step

STEP_PATH = "96260B_REAR_XR004_PCD PLATE.stp copy"
MM_PER_IN = 25.4

# Wall-family nominal diameters (inches) that make up a pocket wall set.
WALL_DIA_IN = (0.800, 0.500, 3.453)
# Central-hole nominal diameters (inches) we must keep OUT of the pocket clusters.
CENTRAL_HOLE_DIA_IN = (4.006, 3.200)
DIA_TOL_IN = 0.02


def dia_in(f):
    return (2.0 * f.radius) / MM_PER_IN if f.radius else None


def near(x, target, tol=DIA_TOL_IN):
    return x is not None and abs(x - target) < tol


# ---------------------------------------------------------------------------
# Inlined single-linkage union-find to a target K (numpy only, no sklearn)
# ---------------------------------------------------------------------------
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def ncomp(self):
        return len({self.find(k) for k in range(len(self.p))})

    def labels(self):
        roots, out = {}, []
        for k in range(len(self.p)):
            r = self.find(k)
            roots.setdefault(r, len(roots))
            out.append(roots[r])
        return np.asarray(out)


def single_linkage_to_k(points_2d: np.ndarray, k: int) -> np.ndarray:
    """Merge closest pairs until exactly `k` connected components remain."""
    n = len(points_2d)
    diff = points_2d[:, None, :] - points_2d[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    iu, ju = np.triu_indices(n, k=1)
    order = np.argsort(dist[iu, ju])
    uf = _UF(n)
    for o in order:
        if uf.ncomp() <= k:
            break
        uf.union(int(iu[o]), int(ju[o]))
    return uf.labels()


# ---------------------------------------------------------------------------
# Build the face pool + opening axis
# ---------------------------------------------------------------------------
faces = analyze_step(STEP_PATH)
by_index = {f.index: f for f in faces}


def is_wall(f):
    if f.surface_type not in ("cylinder", "cone") or not f.radius:
        return False
    d = dia_in(f)
    return any(near(d, t) for t in WALL_DIA_IN)


pool_idx = [
    f.index for f in faces
    if is_wall(f) or f.surface_type in ("bspline", "bezier", "sphere")
]

# Opening axis = the shared wall-axis direction (detected, not assumed to be Y).
wall_axes = np.array([
    np.asarray(by_index[i].axis, float) / (np.linalg.norm(by_index[i].axis) + 1e-12)
    for i in pool_idx
    if is_wall(by_index[i]) and by_index[i].axis is not None
])
# Sign-fold (+d == -d) then take the dominant direction.
folded = wall_axes * np.sign(wall_axes[:, np.argmax(np.abs(wall_axes).sum(axis=0))])[:, None]
opening_axis = folded.mean(axis=0)
opening_axis /= np.linalg.norm(opening_axis) + 1e-12
drop = int(np.argmax(np.abs(opening_axis)))  # coordinate to project out
keep = [c for c in range(3) if c != drop]

pool_hist = Counter(
    "wall" if is_wall(by_index[i]) else by_index[i].surface_type for i in pool_idx
)
print(f"opening axis (detected) = {np.round(opening_axis, 3)}  -> project out coord {drop} "
      f"('{'XYZ'[drop]}'), cluster in ('{'XYZ'[keep[0]]}','{'XYZ'[keep[1]]}')")
print(f"pocket candidate pool: {len(pool_idx)} faces  {dict(pool_hist)}")


def plane_xz(idx_list):
    return np.array([[by_index[i].centroid[keep[0]], by_index[i].centroid[keep[1]]]
                     for i in idx_list], float)


# ---------------------------------------------------------------------------
# Identify the central-hole faces empirically
# ---------------------------------------------------------------------------
central = {}  # dia_in -> [face indices]
for f in faces:
    if f.surface_type == "cylinder" and f.radius:
        d = dia_in(f)
        for t in CENTRAL_HOLE_DIA_IN:
            if near(d, t):
                central.setdefault(t, []).append(f.index)

print("\ncentral-hole wall faces (empirically identified):")
for t in CENTRAL_HOLE_DIA_IN:
    ids = sorted(central.get(t, []))
    kind = "blind" if abs(t - 4.006) < 1e-3 else "through"
    print(f"  ⌀{t:.3f} in ({t * MM_PER_IN:.3f} mm, {kind}): faces {ids}")

# ---------------------------------------------------------------------------
# Cluster the pool into K=7 and characterize each cluster's (X,Z) footprint
# ---------------------------------------------------------------------------
K = 7
P = plane_xz(pool_idx)
labels = single_linkage_to_k(P, K)
n_clusters = len(np.unique(labels))
print(f"\n(X,Z) single-linkage proximity clustering -> {n_clusters} clusters (target K={K})")

clusters = {}  # label -> dict(members, centroid, spread)
for lab in np.unique(labels):
    members = [pool_idx[j] for j in range(len(pool_idx)) if labels[j] == lab]
    pts = P[labels == lab]
    c = pts.mean(axis=0)
    spread = float(np.sqrt(((pts - c) ** 2).sum(axis=1)).max())
    clusters[int(lab)] = {"members": set(members), "centroid": c, "spread": spread,
                          "n": len(members)}

for lab in sorted(clusters):
    cl = clusters[lab]
    print(f"  cluster {lab}: {cl['n']:2d} faces  centroid=({cl['centroid'][0]:+7.2f},"
          f"{cl['centroid'][1]:+7.2f})  footprint_radius={cl['spread']:.2f} mm")

# ---------------------------------------------------------------------------
# Overlap test: are any central-hole faces in / inside a pocket cluster?
# ---------------------------------------------------------------------------
print("\ncentral-hole -> pocket-cluster overlap test:")
leaked = []
central_faces = sorted({i for ids in central.values() for i in ids})
for fid in central_faces:
    p = np.array([by_index[fid].centroid[keep[0]], by_index[fid].centroid[keep[1]]], float)
    # (a) membership
    member_of = [lab for lab, cl in clusters.items() if fid in cl["members"]]
    # (b) spatial footprint — nearest cluster + whether inside its footprint radius
    dists = {lab: float(np.linalg.norm(p - cl["centroid"])) for lab, cl in clusters.items()}
    nearest = min(dists, key=dists.get)
    d_near = dists[nearest]
    inside = d_near <= clusters[nearest]["spread"]
    dia = dia_in(by_index[fid])
    status = "LEAK" if (member_of or inside) else "clear"
    if member_of or inside:
        leaked.append(fid)
    print(f"  face {fid:>3} ⌀{dia:.3f}in  (X,Z)=({p[0]:+7.2f},{p[1]:+7.2f})  "
          f"member_of={member_of}  nearest=cluster {nearest} @ {d_near:.2f} mm "
          f"(footprint {clusters[nearest]['spread']:.2f} mm) -> {status}")

print("\n" + "=" * 70)
if leaked:
    print(f"RESULT: FAIL — central-hole faces leaked into pocket clusters: {leaked}")
    print("STOP: option-1 ordering has an ownership leak; adjust the candidate")
    print("filter before writing the pocket pass.")
    raise SystemExit(1)
else:
    print("RESULT: PASS — the 2 central holes fall in NONE of the 7 pocket clusters")
    print("(neither as members nor inside any cluster's (X,Z) footprint).")
    print("Safe to build pocket_detection.py sourcing from the full pool.")
