"""probe_pockets_blend.py — blend-ring probe (diagnostic only).

Two questions before speccing the pocket pass:

(A) Do the B-spline floors + spheres attach to a BLEND RING of non-wall-family
    cyl/cone faces that physically bounds each pocket floor? Characterize it via
    edge_index: the non-wall direct neighbors of floors/spheres, their surface_type
    / diameter-family composition, whether blend+floor+sphere graph-connects, and
    whether that structure clusters 7-fold spatially (same K=7 angular method).

(B) Is the pocket grouping truly spatial-proximity (generalizes to any layout) or
    only works because this part is a symmetric ring of 7? Re-cluster the full
    pocket pool by GENERIC Euclidean proximity in the (X,Z) plane (inline
    single-linkage / distance-threshold union-find, numpy only, no sklearn) and
    compare membership against the polar-angle method.

Run: /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_pockets_blend.py
"""
from collections import Counter, defaultdict

import numpy as np

from feature_params import analyze_step

STEP_PATH = "96260B_REAR_XR004_PCD PLATE.stp copy"
GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"
MM_PER_IN = 25.4

WALL_TARGETS = {"0.800": 0.800, "0.500": 0.500, "3.453": 3.453}
WALL_TAGS = tuple(WALL_TARGETS)


def dia_in(f):
    return (2 * f.radius) / MM_PER_IN if f.radius else None


def near(x, target, tol=0.05):
    return x is not None and abs(x - target) < tol


def wall_family(f):
    """Return '0.800'/'0.500'/'3.453' if f is a wall-family cyl/cone, else None."""
    if f.surface_type not in ("cylinder", "cone") or not f.radius:
        return None
    di = dia_in(f)
    for name, t in WALL_TARGETS.items():
        if near(di, t):
            return name
    return None


def tag_of(f):
    """Descriptive tag used for composition histograms."""
    wf = wall_family(f)
    if wf is not None:
        return wf
    st = f.surface_type
    if st in ("bspline", "bezier"):
        return "bspline"
    if st == "sphere":
        return "sphere"
    if st in ("cylinder", "cone"):
        di = dia_in(f)
        d = f"{di:.3f}" if di is not None else "0.000"
        return f"{'cyl' if st == 'cylinder' else 'cone'}\u2300{d}"
    return st  # plane, torus, ...


# ---------------------------------------------------------------------------
# Load faces + graph adjacency
# ---------------------------------------------------------------------------
faces = analyze_step(STEP_PATH)
by_index = {f.index: f for f in faces}
tag_by_index = {f.index: tag_of(f) for f in faces}

data = np.load(GRAPH_NPZ)
ei = data["edge_index"]
adj = defaultdict(set)
for a, b in zip(ei[0], ei[1]):
    adj[int(a)].add(int(b))
    adj[int(b)].add(int(a))


def induced_components(face_ids):
    """Connected components of the subgraph induced on `face_ids`."""
    fset = set(face_ids)
    seen = set()
    comps = []
    for s in face_ids:
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u] & fset:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


# ---------------------------------------------------------------------------
# Face sets
# ---------------------------------------------------------------------------
wall_faces = {f.index for f in faces if wall_family(f) is not None}
floor_faces = {f.index for f in faces if f.surface_type in ("bspline", "bezier")}
sphere_faces = {f.index for f in faces if f.surface_type == "sphere"}
fs_faces = floor_faces | sphere_faces  # floors + spheres

print("=" * 78)
print("PROBE A — blend ring characterization")
print("=" * 78)
print(f"wall-family faces : {len(wall_faces)}  "
      f"(0.800\u00d7{sum(1 for i in wall_faces if tag_by_index[i]=='0.800')}, "
      f"0.500\u00d7{sum(1 for i in wall_faces if tag_by_index[i]=='0.500')}, "
      f"3.453\u00d7{sum(1 for i in wall_faces if tag_by_index[i]=='3.453')})")
print(f"bspline floors    : {len(floor_faces)}")
print(f"spheres           : {len(sphere_faces)}")

# ---------------------------------------------------------------------------
# A: direct non-wall neighbors of floors + spheres
# ---------------------------------------------------------------------------
nonwall_nbrs = set()
for i in fs_faces:
    for nb in adj[i]:
        if nb not in wall_faces:
            nonwall_nbrs.add(nb)

# "blend" faces = non-wall neighbors that are NOT themselves floors/spheres
# (i.e. the genuinely new faces that bound the floor/sphere set).
blend_faces = nonwall_nbrs - fs_faces

print(f"\ndirect non-wall neighbors of floors/spheres: {len(nonwall_nbrs)} faces")
nbr_hist = Counter(tag_by_index[i] for i in nonwall_nbrs)
print(f"  bucketed by tag: {dict(sorted(nbr_hist.items()))}")
print(f"    (of these, {len(nonwall_nbrs & fs_faces)} are floors/spheres themselves)")

blend_hist = Counter(tag_by_index[i] for i in blend_faces)
print(f"\nBLEND faces (non-wall, non-floor/sphere neighbors): {len(blend_faces)}")
print(f"  bucketed by tag: {dict(sorted(blend_hist.items()))}")
# cyl/cone blend faces broken out by diameter family explicitly
cc_blend = defaultdict(list)
for i in sorted(blend_faces):
    f = by_index[i]
    if f.surface_type in ("cylinder", "cone"):
        cc_blend[tag_by_index[i]].append(i)
if cc_blend:
    print("  cyl/cone blend by diameter family:")
    for k, v in sorted(cc_blend.items()):
        print(f"    {k}: {len(v)} faces  {v}")
print(f"  blend face_ids = {sorted(blend_faces)}")

# ---------------------------------------------------------------------------
# A: does blend + floors + spheres graph-connect?
# ---------------------------------------------------------------------------
struct = sorted(blend_faces | fs_faces)
comps = induced_components(struct)
comp_sizes = sorted((len(c) for c in comps), reverse=True)
print(f"\nblend+floor+sphere structure: {len(struct)} faces -> "
      f"{len(comps)} connected components (sizes {comp_sizes})")
for ci, c in enumerate(sorted(comps, key=len, reverse=True)):
    ch = Counter(tag_by_index[i] for i in c)
    print(f"  comp {ci}: {len(c)} faces  {dict(sorted(ch.items()))}")

# ---------------------------------------------------------------------------
# Angular (polar-gap) clustering — the established K=7 method
# ---------------------------------------------------------------------------
def cluster_by_angle_gaps(angles: np.ndarray, K: int) -> np.ndarray:
    """Cut the circle at its K largest angular gaps -> K contiguous arcs."""
    n = len(angles)
    order = np.argsort(angles)
    sa = angles[order]
    gaps = np.array([
        (sa[(i + 1) % n] - sa[i]) + (2 * np.pi if i == n - 1 else 0.0)
        for i in range(n)
    ])
    cuts = set(np.argsort(gaps)[-K:].tolist())
    start = (max(cuts) + 1) % n
    labels_sorted = np.empty(n, dtype=int)
    cur = 0
    for k in range(n):
        i = (start + k) % n
        labels_sorted[i] = cur
        if i in cuts:
            cur += 1
    labels = np.empty(n, dtype=int)
    labels[order] = labels_sorted % K
    return labels


def xz(indices):
    return np.array([[by_index[i].centroid[0], by_index[i].centroid[2]]
                     for i in indices], float)


def angular_labels(indices, K=7):
    P = xz(indices)
    center = P.mean(axis=0)
    rel = P - center
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    return cluster_by_angle_gaps(ang, K), center


def print_groups(indices, labels, title):
    groups = defaultdict(list)
    for j, lab in enumerate(labels):
        groups[int(lab)].append(indices[j])
    # order groups by mean polar angle for readability
    P = xz(indices)
    center = P.mean(axis=0)
    ang = np.arctan2(P[:, 1] - center[1], P[:, 0] - center[0])
    ang_by_idx = {indices[j]: ang[j] for j in range(len(indices))}
    gids = sorted(groups, key=lambda g: np.mean([ang_by_idx[i] for i in groups[g]]))
    signatures = []
    print(f"\n{title}")
    for gnum, gid in enumerate(gids):
        members = sorted(groups[gid])
        ch = Counter(tag_by_index[i] for i in members)
        sig = tuple(sorted(ch.items()))
        signatures.append(sig)
        print(f"  group {gnum}: {len(members)} faces  {dict(sorted(ch.items()))}")
        print(f"     face_ids={members}")
    uniform = len(set(signatures)) == 1
    print(f"  UNIFORM composition across {len(gids)} groups: {uniform}"
          + (f"  template={dict(sorted(signatures[0]))}" if uniform else ""))
    if not uniform:
        modal, nmodal = Counter(signatures).most_common(1)[0]
        print(f"  MODAL template ({nmodal}/{len(gids)} groups): {dict(sorted(modal))}")
        for gnum, sig in enumerate(signatures):
            if sig != modal:
                extra = dict(sorted(set(sig) - set(modal)))
                missing = dict(sorted(set(modal) - set(sig)))
                print(f"    group {gnum} deviates: extra={extra} missing={missing}")
    return uniform


# A: does blend+floor+sphere structure cluster 7-fold spatially?
struct_lbl, _ = angular_labels(struct, K=7)
struct_uniform = print_groups(
    struct, struct_lbl,
    "blend+floor+sphere structure, angular K=7:")

# ---------------------------------------------------------------------------
# PROBE B — generic distance-based clustering vs angular
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PROBE B — generic (X,Z) Euclidean proximity vs polar-angle clustering")
print("=" * 78)

# Full pocket pool = walls + floors + spheres (same pool the angular method uses).
pool = sorted(wall_faces | floor_faces | sphere_faces)
pool_hist = Counter(tag_by_index[i] for i in pool)
print(f"pool: {len(pool)} faces  {dict(sorted(pool_hist.items()))}")

P = xz(pool)
n = len(pool)

# pairwise (X,Z) Euclidean distances
diff = P[:, None, :] - P[None, :, :]
D = np.sqrt((diff ** 2).sum(axis=2))


class UF:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

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
        return len({self.find(k) for k in self.p})

    def labels(self):
        roots = {}
        out = []
        for k in self.p:
            r = self.find(k)
            roots.setdefault(r, len(roots))
            out.append(roots[r])
        return np.array(out)


# Single-linkage (agglomerative): add edges in increasing distance until we
# reach K=7 components. This is parameter-free and equivalent to a
# distance-threshold union-find swept to the point that yields 7 clusters.
iu, ju = np.triu_indices(n, k=1)
order = np.argsort(D[iu, ju])
edges = [(D[iu[o], ju[o]], iu[o], ju[o]) for o in order]

uf = UF(range(n))
merge_dist_to_reach = {}  # #components -> distance of the edge that produced it
last_edge_dist = 0.0
for dist, a, b in edges:
    if uf.find(a) != uf.find(b):
        prev = uf.ncomp()
        uf.union(a, b)
        now = uf.ncomp()
        if now < prev:
            merge_dist_to_reach[now] = dist
        last_edge_dist = dist
    if uf.ncomp() == 7:
        break

# rebuild single-linkage labels at exactly 7 clusters
uf7 = UF(range(n))
for dist, a, b in edges:
    if uf7.ncomp() <= 7:
        break
    uf7.union(a, b)
dist_lbl = uf7.labels()

# angular labels on the same pool
ang_lbl, _ = angular_labels(pool, K=7)


def align_agreement(la, lb):
    """Greedy label alignment; return (matched_faces, mapping)."""
    ca, cb = np.unique(la), np.unique(lb)
    cont = np.zeros((len(ca), len(cb)), int)
    ai = {c: i for i, c in enumerate(ca)}
    bi = {c: i for i, c in enumerate(cb)}
    for x, y in zip(la, lb):
        cont[ai[x], bi[y]] += 1
    pairs = sorted(
        ((cont[i, j], i, j) for i in range(len(ca)) for j in range(len(cb))),
        reverse=True,
    )
    used_a, used_b, matched = set(), set(), 0
    mapping = {}
    for cnt, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        mapping[ca[i]] = cb[j]
        matched += cnt
    return matched, mapping


matched, mapping = align_agreement(ang_lbl, dist_lbl)
agree_pct = 100.0 * matched / n
n_diff = n - matched

# What distance gap makes it separable?
# max intra-cluster spread (per angular cluster: max dist from centroid),
# vs min inter-cluster centroid distance.
ang_groups = defaultdict(list)
for j, lab in enumerate(ang_lbl):
    ang_groups[int(lab)].append(j)
intra_spreads = []
centroids = []
for lab, members in sorted(ang_groups.items()):
    pts = P[members]
    c = pts.mean(axis=0)
    centroids.append(c)
    intra_spreads.append(float(np.sqrt(((pts - c) ** 2).sum(axis=1)).max()))
centroids = np.array(centroids)
cd = np.sqrt(((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
np.fill_diagonal(cd, np.inf)
min_inter_centroid = float(cd.min())
max_intra_spread = float(max(intra_spreads))

# single-linkage dendrogram gap: distance to reach 7 clusters vs to drop to 6
d7 = merge_dist_to_reach.get(7, last_edge_dist)
d6 = None
uf6 = UF(range(n))
for dist, a, b in edges:
    if uf6.find(a) != uf6.find(b):
        prev = uf6.ncomp()
        uf6.union(a, b)
        if uf6.ncomp() == 6 and prev == 7:
            d6 = dist
            break
    if uf6.ncomp() < 6:
        break

print(f"\nnumber of distance-clusters (single-linkage @ 7 target): "
      f"{len(np.unique(dist_lbl))}")
print(f"angular vs distance agreement: {matched}/{n} faces "
      f"({agree_pct:.1f}%), {n_diff} faces differ")
print(f"cluster label mapping (angular->distance): {mapping}")

if n_diff:
    print("\nfaces where the two methods disagree:")
    for j in range(n):
        if mapping.get(ang_lbl[j]) != dist_lbl[j]:
            print(f"  face {pool[j]:>4}  tag={tag_by_index[pool[j]]:<10} "
                  f"angular={ang_lbl[j]} distance={dist_lbl[j]}")

print("\nseparability (centroid-based):")
print(f"  max intra-cluster spread   = {max_intra_spread:.2f} mm")
print(f"  min inter-cluster centroid = {min_inter_centroid:.2f} mm")
print(f"  gap ratio (inter/intra)    = {min_inter_centroid / max_intra_spread:.2f}x")
print("separability (single-linkage dendrogram):")
print(f"  max merge distance to form 7 clusters  = {d7:.2f} mm")
print(f"  merge distance that would drop 7->6     = "
      f"{d6:.2f} mm" if d6 is not None else "  merge distance 7->6 = n/a")
if d6 is not None:
    print(f"  dendrogram gap (7->6 / within-7)        = {d6 / d7:.2f}x")

# Dump both partitions when they disagree.
print("\nangular partition:")
_ = print_groups(pool, ang_lbl, "  (polar-angle K=7)")
if n_diff:
    print("\ndistance partition:")
    _ = print_groups(pool, dist_lbl, "  (single-linkage K=7)")

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
blend_per_comp = len(blend_faces) / len(comps) if comps else 0
print("blend ring:")
print(f"  - exists per pocket?      {'YES' if blend_faces else 'NO'} "
      f"({len(blend_faces)} blend faces, {dict(sorted(blend_hist.items()))})")
print(f"  - floor+blend+sphere connected? "
      f"{len(comps)} components (sizes {comp_sizes})")
print(f"  - components per pocket (K=7): {len(comps)}/7 = "
      f"{len(comps) / 7:.1f} components each")
print(f"  - blend+floor+sphere clusters 7-fold uniform? {struct_uniform}")
print("grouping generalizability:")
print(f"  - distance-based (non-angular) gives 7 clean pockets? "
      f"{'YES' if len(np.unique(dist_lbl)) == 7 else 'NO'}")
print(f"  - angular vs distance agreement: {agree_pct:.1f}% "
      f"({n_diff} faces differ)")
separable = min_inter_centroid > max_intra_spread
if agree_pct >= 99.5 and separable:
    verdict = ("GENERALIZABLE (proximity). Distance clustering reproduces the 7 "
               "angular pockets; inter-pocket centroid gap exceeds intra-pocket "
               "spread, so pockets are Euclidean-separable without ring assumptions.")
    primitive = ("generic (X,Z) single-linkage / distance-threshold union-find "
                 "on face centroids (no polar-angle assumption)")
elif agree_pct >= 99.5 and not separable:
    verdict = ("GENERALIZABLE but MARGINAL. Distance clustering matches angular, "
               "but intra spread >= inter centroid gap — threshold is fragile.")
    primitive = ("distance union-find, but pick threshold from the single-linkage "
                 "dendrogram gap rather than a fixed value")
else:
    verdict = ("RING-SPECIFIC (angular only). Distance clustering does NOT reproduce "
               "the 7 angular pockets — grouping relies on the symmetric ring layout.")
    primitive = ("polar-angle gap clustering is ring-specific; need a different "
                 "grouping primitive (e.g. floor-seeded region growth) for non-ring parts")
print(f"  - is pocket grouping GENERALIZABLE or RING-SPECIFIC?\n      {verdict}")
print(f"  - recommended pocket-pass grouping primitive:\n      {primitive}")
