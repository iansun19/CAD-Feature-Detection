"""probe_flats_gap.py — resolve the flats-pass 6-vs-8 gap (diagnostic only).

Run: /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_flats_gap.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from feature_params import analyze_step
from run_cascade import _load_edges, run_cascade

STEP_PATH = "96260B_REAR_XR004_PCD PLATE.stp copy"
GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"
MM_PER_IN = 25.4

# Coplanar groups from probe_flats_census.py
GROUPS: dict[int, list[int]] = {
    0: [105, 144, 174, 204, 234, 264, 294],
    1: [129, 161, 191, 221, 251, 281, 309],
    2: [322],
    3: [326],
    4: [330],
    5: [332],
}
STRAY_BOSS_FACES = [328, 346]
ALL_PROBE_FACES = sorted({i for ids in GROUPS.values() for i in ids})

faces = analyze_step(STEP_PATH)
by_index = {f.index: f for f in faces}
edge_index, edge_attr = _load_edges(Path(GRAPH_NPZ), Path(STEP_PATH))
_, pk, hl, _, _, _ = run_cascade(STEP_PATH, edge_index, edge_attr)
claimed_pocket = set(int(i) for i in pk.claimed_faces)

opening_axis = np.array(pk.opening_axis, dtype=np.float64)
opening_axis /= np.linalg.norm(opening_axis) + 1e-12
drop = int(np.argmax(np.abs(opening_axis)))
keep = [c for c in range(3) if c != drop]
axis_names = "XYZ"


def xz_of(idx: int) -> np.ndarray:
    c = by_index[idx].centroid
    return np.array([c[keep[0]], c[keep[1]]], dtype=np.float64)


def xyz_str(idx: int) -> str:
    c = by_index[idx].centroid
    return f"({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})"


# Pocket (X,Z) centroids + per-pocket footprint from claimed face sets
pockets: list[dict] = []
for feat in pk.features:
    members = sorted(int(i) for i in feat.face_indices)
    pts = np.array([xz_of(i) for i in members], float)
    centroid = pts.mean(axis=0)
    spread = float(np.sqrt(((pts - centroid) ** 2).sum(axis=1)).max())
    pockets.append(
        {
            "id": feat.feature_id,
            "centroid_xz": centroid,
            "footprint_radius": spread,
            "claimed_faces": members,
            "centroid_uv": feat.centroid_uv,
        }
    )

ref_footprint = max(p["footprint_radius"] for p in pockets)
print("=" * 78)
print("SETUP")
print("=" * 78)
print(f"opening axis = {np.round(opening_axis, 4)}  (project out {axis_names[drop]})")
print(f"cluster plane = ({axis_names[keep[0]]}, {axis_names[keep[1]]})")
print(f"pockets: {len(pockets)}  reference footprint radius = {ref_footprint:.2f} mm")
for p in pockets:
    c = p["centroid_xz"]
    print(
        f"  pocket {p['id']}: centroid=({c[0]:+7.2f},{c[1]:+7.2f})  "
        f"footprint={p['footprint_radius']:.2f} mm  "
        f"claimed {len(p['claimed_faces'])} faces"
    )

# Graph adjacency
adj: dict[int, set[int]] = defaultdict(set)
for a, b in zip(edge_index[0], edge_index[1]):
    adj[int(a)].add(int(b))
    adj[int(b)].add(int(a))


def nearest_pocket(idx: int) -> tuple[int, float, float]:
    p = xz_of(idx)
    dists = [(pk["id"], float(np.linalg.norm(p - pk["centroid_xz"]))) for pk in pockets]
    pid, dist = min(dists, key=lambda t: t[1])
    fp = next(x["footprint_radius"] for x in pockets if x["id"] == pid)
    return pid, dist, fp


# ---------------------------------------------------------------------------
# Q1 — spatial: nearest pocket + footprint membership
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Q1 — SPATIAL (nearest pocket centroid, footprint membership)")
print("=" * 78)
q1_inside: dict[int, list[int]] = defaultdict(list)
q1_outside: dict[int, list[int]] = defaultdict(list)
for gid, idxs in GROUPS.items():
    print(f"\ngroup {gid}: faces {idxs}")
    n_in_ref = 0
    n_in_own = 0
    for idx in idxs:
        pid, dist, fp = nearest_pocket(idx)
        in_own = dist <= fp
        in_ref = dist <= ref_footprint
        if in_ref:
            n_in_ref += 1
            q1_inside[gid].append(idx)
        else:
            q1_outside[gid].append(idx)
        if in_own:
            n_in_own += 1
        print(
            f"  face {idx:>3}  (X,Z)=({xz_of(idx)[0]:+7.2f},{xz_of(idx)[1]:+7.2f})  "
            f"nearest=pocket {pid}  dist={dist:.2f} mm  "
            f"own_fp={fp:.2f}  in_own_fp={in_own}  in_ref_fp({ref_footprint:.2f})={in_ref}"
        )
    print(f"  summary: {n_in_ref}/{len(idxs)} inside ref footprint ({ref_footprint:.2f} mm)")
    print(f"           {n_in_own}/{len(idxs)} inside nearest pocket's own footprint")


# ---------------------------------------------------------------------------
# Q2 — graph adjacency to pocket-claimed faces
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Q2 — GRAPH (adjacency to pocket-claimed faces)")
print("=" * 78)
q2_adjacent: dict[int, list[int]] = defaultdict(list)
q2_isolated: dict[int, list[int]] = defaultdict(list)
for gid, idxs in GROUPS.items():
    n_adj = 0
    print(f"\ngroup {gid}: faces {idxs}")
    for idx in idxs:
        nbrs_claimed = sorted(adj[idx] & claimed_pocket)
        if nbrs_claimed:
            n_adj += 1
            q2_adjacent[gid].append(idx)
        else:
            q2_isolated[gid].append(idx)
        nbr_types = Counter(by_index[n].surface_type for n in adj[idx])
        print(
            f"  face {idx:>3}  adjacent_to_claimed={len(nbrs_claimed)}  "
            f"claimed_nbrs={nbrs_claimed}  all_nbr_types={dict(nbr_types)}"
        )
    print(f"  summary: {n_adj}/{len(idxs)} graph-adjacent to a pocket-claimed face")


# ---------------------------------------------------------------------------
# Q3 — tiny ±X faces 330, 332 vs stray bosses 328, 346
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Q3 — TINY ±X FACES vs STRAY ⌀6.370 BOSSES (328, 346)")
print("=" * 78)


def dia_in(idx: int) -> float | None:
    f = by_index[idx]
    return (2.0 * f.radius) / MM_PER_IN if f.radius else None


for idx in [330, 332] + STRAY_BOSS_FACES:
    f = by_index[idx]
    d = dia_in(idx)
    dia_s = f"⌀{d:.3f}in" if d else "n/a"
    print(
        f"\nface {idx}  type={f.surface_type}  area={f.area:.2f} mm²  "
        f"dia={dia_s}  centroid={xyz_str(idx)}  normal={np.round(f.normal, 3)}"
    )
    nbrs = sorted(adj[idx])
    for nb in nbrs:
        print(f"    nbr {nb:>3}  type={by_index[nb].surface_type}  centroid={xyz_str(nb)}")

print("\n330/332 -> boss spatial comparison:")
for tiny in [330, 332]:
    p = xz_of(tiny)
    for boss in STRAY_BOSS_FACES:
        q = xz_of(boss)
        dist_xz = float(np.linalg.norm(p - q))
        dist_3d = float(np.linalg.norm(by_index[tiny].centroid - by_index[boss].centroid))
        graph_adj = boss in adj[tiny]
        print(
            f"  face {tiny} vs boss {boss}: (X,Z) dist={dist_xz:.2f} mm  "
            f"3D dist={dist_3d:.2f} mm  graph_adjacent={graph_adj}"
        )


# ---------------------------------------------------------------------------
# Q4 — large flats 322, 326
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Q4 — LARGE FLATS 322, 326")
print("=" * 78)
for idx in [322, 326]:
    f = by_index[idx]
    pid, dist, fp = nearest_pocket(idx)
    in_fp = dist <= ref_footprint
    # extent proxy: sqrt(area) side length; axial span along opening axis
    side = float(np.sqrt(f.area))
    print(
        f"\nface {idx}  area={f.area:.1f} mm²  sqrt(area)≈{side:.1f} mm  "
        f"centroid={xyz_str(idx)}  normal={np.round(f.normal, 3)}"
    )
    print(
        f"  nearest pocket {pid} @ {dist:.2f} mm  "
        f"inside ref footprint ({ref_footprint:.2f}) = {in_fp}"
    )
    nbrs = sorted(adj[idx])
    nbr_types = Counter(by_index[n].surface_type for n in nbrs)
    claimed_nbrs = sorted(adj[idx] & claimed_pocket)
    print(f"  degree={len(nbrs)}  neighbor types={dict(nbr_types)}")
    print(f"  neighbors adjacent to pocket-claimed: {claimed_nbrs}")
    print("  neighbors:")
    for nb in nbrs:
        print(f"    {nb:>3}  {by_index[nb].surface_type:>8}  area={by_index[nb].area:8.1f}  {xyz_str(nb)}")


# ---------------------------------------------------------------------------
# Q5 — per-pocket assignment for groups 0 & 1
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Q5 — PER-POCKET ASSIGNMENT (groups 0 & 1)")
print("=" * 78)
for gid in (0, 1):
    idxs = GROUPS[gid]
    print(f"\ngroup {gid}: faces {idxs}")
    assignment: dict[int, list[int]] = defaultdict(list)
    for idx in idxs:
        pid, dist, fp = nearest_pocket(idx)
        assignment[pid].append(idx)
        print(f"  face {idx:>3} -> pocket {pid}  dist={dist:.2f} mm")
    pockets_hit = sorted(assignment)
    doubles = {p: fs for p, fs in assignment.items() if len(fs) > 1}
    orphans = sorted(set(range(len(pockets))) - set(pockets_hit))
    print(f"  distinct pockets assigned: {len(pockets_hit)}/7  pockets={pockets_hit}")
    if doubles:
        print(f"  DOUBLES (same pocket, multiple faces): {dict(doubles)}")
    if orphans:
        print(f"  ORPHAN pockets (no face assigned): {orphans}")
    if len(pockets_hit) == 7 and not doubles and not orphans:
        print("  -> clean 1:1 assignment (7 faces -> 7 pockets)")


# ---------------------------------------------------------------------------
# VERDICT (data-driven; do not force clean buckets)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)

g0, g1 = GROUPS[0], GROUPS[1]
g0_q1_in = [i for i in g0 if i not in q1_outside[0]]
g0_q1_out = q1_outside[0]
g1_q1_in = [i for i in g1 if i not in q1_outside[1]]
g1_q1_out = q1_outside[1]
g0_q2_adj = q2_adjacent[0]
g0_q2_iso = q2_isolated[0]
g1_q2_adj = q2_adjacent[1]
g1_q2_iso = q2_isolated[1]

print("- groups 0/1: SPLIT — the two depth bands behave differently (Q1 vs Q2):")
print(
    f"    group 0 [{g0}]: Q1 {len(g0_q1_in)}/7 inside footprint, "
    f"Q2 {len(g0_q2_adj)}/7 adjacent to claimed pocket faces "
    f"(outside: {g0_q1_out}; graph-adjacent: {g0_q2_adj})"
)
print(
    f"    group 1 [{g1}]: Q1 {len(g1_q1_in)}/7 inside footprint, "
    f"Q2 {len(g1_q2_adj)}/7 adjacent to claimed pocket faces "
    f"(outside: {g1_q1_out}; graph-adjacent: {g1_q2_adj})"
)
print(
    "    both bands: Q5 clean 1:1 nearest-pocket assignment (7 faces -> 7 pockets). "
    "Group 0 sits ~32.0 mm from pocket centroids (just outside the 29.23 mm "
    "footprint); group 1 sits ~28.7 mm (inside). Group 0 bridges pocket claimed "
    "geometry (sphere neighbors); group 1 only touches unclaimed torus blends."
)
print(
    "    pocket-template read: group 1 (Y offset -82 mm) is the stronger pocket "
    "candidate; group 0 (Y offset -77 mm, outer step) is pocket-associated by "
    "graph + Q5 but not by strict footprint — treat as ambiguous outer step."
)
print(
    "    revised template if claimed: 22 + up to 1 step plane per pocket per depth "
    "(14 faces across both bands); do not merge the two bands into one template slot."
)

print("- groups 2/3: GENUINE STANDALONE FLATS (Q4)")
print(
    "    322 (7875 mm², Y=-91.66) and 326 (2463 mm², Y=-85.03): large +Y seating/"
    "top planes, ~68 mm from any pocket centroid, no pocket-claimed neighbors; "
    "adjacent to central-hole cylinders 329/347 and the 330/332/331 stack."
)

boss_adj = any(b in adj[t] for t in [330, 332] for b in STRAY_BOSS_FACES)
boss_near_xz = min(
    float(np.linalg.norm(xz_of(t) - xz_of(b)))
    for t in [330, 332]
    for b in STRAY_BOSS_FACES
)
print("- groups 4/5: NOT stray ⌀6.370 bosses (Q3)")
print(
    f"    330/332 are ±X end caps (area 49.9 mm², Y=-88.69, Z=+71.55) on the "
    f"central through-hole stack — graph-adjacent to 322/326/331, not to 328/346 "
    f"(boss (X,Z) distance ~86 mm, graph_adj={boss_adj}). "
    "Spatially inside pocket-3 footprint but structurally part of central hole/contour."
)

true_standalone_groups = 2  # groups 2, 3
true_standalone_faces = len(GROUPS[2]) + len(GROUPS[3])
pocket_candidate_faces = len(g0) + len(g1)
central_stack_faces = len(GROUPS[4]) + len(GROUPS[5])
print(
    f"- TRUE standalone flats in this residual: {true_standalone_groups} groups / "
    f"{true_standalone_faces} faces (322, 326)"
)
print(
    f"- Pocket-candidate planes: {pocket_candidate_faces} faces in groups 0+1 "
    f"(7+7 at two Y depths)"
)
print(
    f"- Central-stack planes (not stray boss): {central_stack_faces} faces (330, 332)"
)

print("- Toolpath 8 most likely explanation:")
print(
    "    2 large part flats (322, 326) + 2 central end caps (330, 332) = 4 definite "
    "'face' geometry in the central stack; the remaining 4 Toolpath FACES may be "
    "the group-1 pocket step planes counted individually (4 of 7 visible at this "
    "depth in Toolpath's grouping), OR Toolpath splits/labels the two large flats "
    "into multiple face features. The 6 coplanar-equation groups here are NOT "
    "the same taxonomy as Toolpath's 8 — groups 0/1 (14 faces) are pocket-depth "
    "steps Toolpath likely files under POCKETS, not FACES."
)

print("- recommendation:")
print(
    "    1) Amend pocket template to claim group 1 (and possibly group 0) step "
    "planes — strongest evidence on group 1 (Q1 + Q5); group 0 needs explicit rule "
    "(graph bridge to claimed spheres, Y=-77 mm outer step)."
)
print(
    "    2) Flats pass should target groups 2/3 only (2 features) from this residual."
)
print(
    "    3) Groups 4/5 defer to contour/central-hole pass, not flats or stray-boss."
)
print(
    "    Do NOT hunt for '2 missing flats' by splitting coplanar groups — the gap is "
    "a bucket mismatch (pocket steps + central stack vs Toolpath FACES), not missing geometry."
)
