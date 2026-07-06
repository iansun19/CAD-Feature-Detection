"""probe_pockets_spatial.py — spatial-proximity probe for the pocket pass.

Diagnostic only. Prior probes showed a "pocket" here is NOT a coaxial cluster:
56 wall faces (⌀0.800×28, ⌀0.500×14, ⌀3.453×14) each on its own axis, all sharing
the ±Y opening direction, arranged in 14 quartets around a ring; 42 bspline floors
(graph-isolated) + 28 spheres form the blend/floor set. Toolpath reports 7 filleted
pockets. Hypothesis: 14 quartets = 7 through-pockets × 2 opening directions.

This probe clusters the face pool by POLAR ANGLE about the ring center in the (X,Z)
plane (Y = opening direction projected out) for K=7 and K=14, then runs three
cross-checks (Y-sign split, per-group graph connectivity, floor/sphere assignment)
and prints a verdict.

Run: /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_pockets_spatial.py
"""
from collections import Counter, defaultdict

import numpy as np

from feature_params import analyze_step

STEP_PATH = "96260B_REAR_XR004_PCD PLATE.stp copy"
GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"
MM_PER_IN = 25.4

WALL_TAGS = ("0.800", "0.500", "3.453")
TAG_ORDER = ("0.800", "0.500", "3.453", "bspline", "sphere")


def dia_in(f):
    return (2 * f.radius) / MM_PER_IN if f.radius else None


def near(x, target, tol=0.05):
    return x is not None and abs(x - target) < tol


# ---------------------------------------------------------------------------
# Build the tagged face pool
# ---------------------------------------------------------------------------
faces = analyze_step(STEP_PATH)
st_of = {f.index: f.surface_type for f in faces}

pool = []  # each: dict(tag, index, centroid(3,), axis(3,))
for f in faces:
    tag = None
    if f.surface_type in ("cylinder", "cone") and f.radius:
        di = dia_in(f)
        if near(di, 0.800):
            tag = "0.800"
        elif near(di, 0.500):
            tag = "0.500"
        elif near(di, 3.453):
            tag = "3.453"
    elif f.surface_type in ("bspline", "bezier"):
        tag = "bspline"
    elif f.surface_type == "sphere":
        tag = "sphere"
    if tag is None:
        continue
    axis = np.array(f.axis, float) if f.axis is not None else np.zeros(3)
    pool.append({
        "tag": tag,
        "index": int(f.index),
        "centroid": np.array(f.centroid, float),
        "axis": axis,
    })

n_pool = len(pool)
tag_totals = Counter(p["tag"] for p in pool)
print(f"pool: {n_pool} faces  {dict(tag_totals)}")

# ---------------------------------------------------------------------------
# Ring geometry: project out Y, work in (X, Z); polar angle about ring center
# ---------------------------------------------------------------------------
XZ = np.array([[p["centroid"][0], p["centroid"][2]] for p in pool])  # (n, 2)
center = XZ.mean(axis=0)
rel = XZ - center
angles = np.arctan2(rel[:, 1], rel[:, 0])  # radians in (-pi, pi]
radii = np.linalg.norm(rel, axis=1)
print(f"ring center (X,Z) = ({center[0]:.2f}, {center[1]:.2f}) mm; "
      f"radial spread min/mean/max = {radii.min():.1f}/{radii.mean():.1f}/{radii.max():.1f} mm")


def cluster_by_angle_gaps(angles: np.ndarray, K: int) -> np.ndarray:
    """Cut the circle at its K largest angular gaps -> K contiguous arcs."""
    n = len(angles)
    order = np.argsort(angles)
    sa = angles[order]
    gaps = np.array([
        (sa[(i + 1) % n] - sa[i]) + (2 * np.pi if i == n - 1 else 0.0)
        for i in range(n)
    ])
    cuts = set(np.argsort(gaps)[-K:].tolist())  # boundary after sorted position i
    start = (max(cuts) + 1) % n  # begin numbering right after the last cut
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


# ---------------------------------------------------------------------------
# Graph adjacency
# ---------------------------------------------------------------------------
data = np.load(GRAPH_NPZ)
ei = data["edge_index"]
adj = defaultdict(set)
for a, b in zip(ei[0], ei[1]):
    adj[int(a)].add(int(b))
    adj[int(b)].add(int(a))


def induced_components(face_ids):
    """# connected components of the subgraph induced on face_ids (+ comp sizes)."""
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
        comps.append(comp)
    return comps


def two_hop_wall_to_floor(walls, floors):
    """Count wall->mid->floor triples (mid adjacent to both); tally mid types."""
    wset = set(walls)
    count = 0
    mid_types = Counter()
    for fl in floors:
        for mid in adj[fl]:
            hit = adj[mid] & wset
            if hit:
                count += len(hit)
                mid_types[st_of.get(mid, "?")] += 1
    return count, mid_types


def comp_hist(members):
    c = Counter(pool[i]["tag"] for i in members)
    return {t: c[t] for t in TAG_ORDER if c[t]}


def hist_signature(members):
    c = Counter(pool[i]["tag"] for i in members)
    return tuple(c[t] for t in TAG_ORDER)


# ---------------------------------------------------------------------------
# Run for K = 7 and K = 14
# ---------------------------------------------------------------------------
def run_K(K: int):
    print("\n" + "=" * 78)
    print(f"K = {K}  (cluster by {K} largest angular gaps)")
    print("=" * 78)
    labels = cluster_by_angle_gaps(angles, K)
    groups = defaultdict(list)  # gid -> [pool idx]
    for pi, lab in enumerate(labels):
        groups[int(lab)].append(pi)

    # order groups by mean angle for readability
    gids = sorted(groups, key=lambda g: np.mean([angles[i] for i in groups[g]]))

    signatures = []
    sizes = []
    ysign_ok_all = True
    conn_summary = []
    for gnum, gid in enumerate(gids):
        members = groups[gid]
        sizes.append(len(members))
        signatures.append(hist_signature(members))
        face_ids = sorted(pool[i]["index"] for i in members)
        hist = comp_hist(members)

        # Y-sign split of the 0.800 walls
        pos = sum(1 for i in members if pool[i]["tag"] == "0.800" and pool[i]["axis"][1] > 0)
        neg = sum(1 for i in members if pool[i]["tag"] == "0.800" and pool[i]["axis"][1] < 0)
        if not (pos > 0 and neg > 0):
            ysign_ok_all = False

        # graph connectivity of the group
        comps = induced_components(face_ids)
        walls = [pool[i]["index"] for i in members if pool[i]["tag"] in WALL_TAGS]
        floors = [pool[i]["index"] for i in members if pool[i]["tag"] == "bspline"]
        n2, mid_types = two_hop_wall_to_floor(walls, floors)
        conn_summary.append((len(comps), n2))

        print(f"\n group {gnum} (gid={gid}): {len(members)} faces  {hist}")
        print(f"   0.800 Y-split: +Y={pos}  -Y={neg}")
        print(f"   induced components: {len(comps)} (sizes {sorted((len(c) for c in comps), reverse=True)})")
        print(f"   two-hop wall->mid->floor triples: {n2}  mid-types={dict(mid_types)}")
        print(f"   face_ids={face_ids}")

    uniform = len(set(signatures)) == 1
    print(f"\n K={K} group-size multiset: {sorted(sizes)}")
    print(f" K={K} composition UNIFORM across groups: {uniform}"
          + (f"  template={dict(zip(TAG_ORDER, signatures[0]))}" if uniform else ""))
    print(f" K={K} every group has both +Y and -Y 0.800 walls: {ysign_ok_all}")
    max_comps = max(c for c, _ in conn_summary)
    print(f" K={K} max induced components in any group: {max_comps}")
    return {
        "uniform": uniform,
        "template": dict(zip(TAG_ORDER, signatures[0])) if uniform else None,
        "ysign_ok": ysign_ok_all,
        "labels": labels,
        "max_components": max_comps,
        "conn": conn_summary,
    }


res7 = run_K(7)
res14 = run_K(14)

# ---------------------------------------------------------------------------
# Floor / sphere assignment audit (uses K=7 labels)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("floor/sphere assignment audit")
print("=" * 78)
bspline_pool = [i for i, p in enumerate(pool) if p["tag"] == "bspline"]
sphere_pool = [i for i, p in enumerate(pool) if p["tag"] == "sphere"]
print(f"bspline floors in pool: {len(bspline_pool)} (expect 42)")
print(f"spheres in pool:        {len(sphere_pool)} (expect 28)")
# each pool face has exactly one label by construction; report per-group tallies
for K, res in (("K=7", res7), ("K=14", res14)):
    lab = res["labels"]
    fl = Counter(int(lab[i]) for i in bspline_pool)
    sp = Counter(int(lab[i]) for i in sphere_pool)
    print(f"  {K}: bspline per group = {dict(sorted(fl.items()))}")
    print(f"  {K}: sphere  per group = {dict(sorted(sp.items()))}")
orphans = [p["index"] for p in pool if p["tag"] in ("bspline", "sphere")
           and False]  # none possible: every pooled face is clustered
print(f"unassigned floors/spheres: {orphans} (none possible — full pool is clustered)")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
print(f"  K=7  uniform? {res7['uniform']}"
      + (f"  template={res7['template']}" if res7['uniform'] else ""))
print(f"  K=14 uniform? {res14['uniform']}"
      + (f"  template={res14['template']}" if res14['uniform'] else ""))
print(f"  each K=7 group has both +Y and -Y 0.800 walls? {res7['ysign_ok']}")
print(f"  K=7 floors+spheres attach to walls within-group via graph? "
      f"max_components={res7['max_components']} "
      f"(two-hop triples/group: {[n for _, n in res7['conn']]})")
if res7["uniform"] and res7["ysign_ok"]:
    rec = ("YES — 'pocket = XY-spatial cluster, Y-sign collapsed' matches Toolpath's 7. "
           "Each K=7 group is one through-pocket with both openings.")
elif res14["uniform"] and not res7["uniform"]:
    rec = ("K=7 is NOT uniform but K=14 IS — Toolpath's 7 is coarser than the geometry. "
           "The clean grouping unit is 14 quartets; do NOT force 7 from geometry alone.")
else:
    rec = ("Neither K gives a clean uniform template — spatial angular clustering is not "
           "the right grouping unit as-is; inspect the dumped face_ids before deciding.")
print(f"  RECOMMENDATION: {rec}")
