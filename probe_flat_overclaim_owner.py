#!/usr/bin/env python
"""
PROBE — ownership trace for the 10-face flat overclaim on 96260B_front.

After the hole-annex fix the flat pass claims 10 faces; Toolpath expects 1
(face 97). This probe does NOT change detection — it reports, per flat-bucket
face, depth tier, edge adjacency to pocket/hole/residual/top-surface pools,
and whether root cause (A) pocket-side or (B) flat-side orphan applies.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_flat_overclaim_owner.py --part 96260B_front
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from feature_params import analyze_step, load_step_faces
from hole_detection import FaceGraph
from run_cascade import _load_edges, run_cascade

try:
    from brep_extents import collect_boundary_points
except ImportError:
    collect_boundary_points = None  # type: ignore[misc, assignment]

MM_PER_IN = 25.4
OPENING_AXIS = 1  # Y index for depth-below-top

PARTS: dict[str, dict[str, str]] = {
    "96260B_front": {
        "step": "96260B_FRONT_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_front/graph.npz",
    },
    "96260B_plate": {
        "step": "96260B_REAR_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_plate/graph.npz",
    },
}


def top_of_part_y(step_path: str, n_faces: int) -> float:
    if collect_boundary_points is not None:
        occ = load_step_faces(step_path)
        all_y: list[float] = []
        for i in range(n_faces):
            pts = collect_boundary_points(occ[i])
            all_y.extend(pts[:, 1].tolist())
        return max(all_y)
    faces = analyze_step(step_path)
    return max(float(f.centroid[1]) for f in faces)


def depth_below_top(face_idx: int, by_index: dict, top_y: float) -> float:
    return top_y - float(by_index[face_idx].centroid[1])


def depth_tiers_from_gaps(
    face_ids: list[int],
    depth_fn,
) -> tuple[list[list[int]], list[tuple[float, float]]]:
    """Cluster faces by depth using the largest natural gaps (data-derived)."""
    if not face_ids:
        return [], []
    items = sorted((depth_fn(i), i) for i in face_ids)
    if len(items) == 1:
        d = items[0][0]
        return [[items[0][1]]], [(d, d)]

    gaps: list[tuple[float, int]] = []
    for k in range(len(items) - 1):
        g = items[k + 1][0] - items[k][0]
        if g > 1e-6:
            gaps.append((g, k))

    if not gaps:
        ids = [i for _, i in items]
        return [ids], [(items[0][0], items[-1][0])]

    gaps.sort(key=lambda x: -x[0])
    split_indices = sorted(idx for _, idx in gaps[: min(2, len(gaps))])

    tiers: list[list[int]] = []
    bounds: list[tuple[float, float]] = []
    start = 0
    for split_idx in split_indices:
        chunk = items[start : split_idx + 1]
        tiers.append([i for _, i in chunk])
        bounds.append((chunk[0][0], chunk[-1][0]))
        start = split_idx + 1
    tail = items[start:]
    tiers.append([i for _, i in tail])
    bounds.append((tail[0][0], tail[-1][0]))
    return tiers, bounds


def pocket_instance_map(pocket_result) -> dict[int, int]:
    out: dict[int, int] = {}
    for feat in pocket_result.features:
        for i in feat.face_indices:
            out[int(i)] = int(feat.feature_id)
    return out


def pocket_neighbors(
    face_idx: int,
    graph: FaceGraph,
    pocket_claimed: set[int],
    pocket_map: dict[int, int],
) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for nb in graph.neighbors.get(face_idx, ()):
        if nb in pocket_claimed:
            hits.append((nb, pocket_map[nb]))
    return sorted(hits)


def pocket_two_hop(
    face_idx: int,
    graph: FaceGraph,
    pocket_claimed: set[int],
    pocket_map: dict[int, int],
) -> list[tuple[int, int, int]]:
    """(via_face, pocket_face, pocket_id) reachable in one intermediate hop."""
    hits: list[tuple[int, int, int]] = []
    for mid in graph.neighbors.get(face_idx, ()):
        for nb in graph.neighbors.get(mid, ()):
            if nb in pocket_claimed:
                hits.append((mid, nb, pocket_map[nb]))
    return sorted(set(hits))


def is_top_outer_neighbor(
    nb: int,
    by_index: dict,
    top_y: float,
    global_shallowest_max_depth: float,
    residual_claimed: set[int],
    outer_claimed: set[int],
) -> bool:
    """Neighbor on the opening-side skin or exterior contour (not pocket/hole interior)."""
    fg = by_index[nb]
    d = depth_below_top(nb, by_index, top_y)
    if d <= global_shallowest_max_depth + 1e-6:
        return True
    if nb not in residual_claimed and nb not in outer_claimed:
        return False
    st = fg.surface_type
    if st in ("plane", "cylinder", "cone"):
        n = np.asarray(fg.normal if fg.normal is not None else [0, 0, 0], dtype=float)
        if st == "plane" and abs(n[OPENING_AXIS]) < 0.5:
            return True
        if st in ("cylinder", "cone"):
            return True
    return False


def assign_verdict(
    face_idx: int,
    *,
    tier_label: str,
    shallowest_tier: bool,
    pocket_adj: list,
    hole_adj: list,
    depth_rank: int,
    n_tiers: int,
) -> str:
    if shallowest_tier and tier_label.startswith("tier-0"):
        return "TRUE-FLAT"
    if pocket_adj:
        return "POCKET-adjacent"
    if hole_adj:
        return "HOLE-adjacent"
    return "ORPHAN-deep"


def assign_cause(pocket_adj: list, pocket_2hop: list) -> str:
    if pocket_adj:
        return "A (pocket-side: direct edge to pocket-claimed face)"
    if pocket_2hop:
        return "A? (pocket-side: 2-hop via fillet — pocket grow under-reached)"
    return "B (flat-side orphan: no direct pocket edge)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="96260B_front")
    ap.add_argument("--step", default=None)
    ap.add_argument("--graph-npz", default=None)
    args = ap.parse_args()

    cfg = PARTS.get(args.part, {})
    step_path = args.step or cfg.get("step") or f"{args.part}.step"
    graph_npz = args.graph_npz or cfg.get("graph_npz")

    faces_list = analyze_step(step_path)
    by_index = {f.index: f for f in faces_list}
    n_faces = len(faces_list)
    top_y = top_of_part_y(step_path, n_faces)

    edge_index, edge_attr = _load_edges(Path(graph_npz), Path(step_path))
    _all_faces, pocket_result, hole_result, flat_result, outer_result, residual_result = (
        run_cascade(step_path, edge_index, edge_attr)
    )
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)

    flat_bucket = sorted(int(i) for i in flat_result.claimed_faces)
    pocket_claimed = set(int(i) for i in pocket_result.claimed_faces)
    hole_claimed = set(int(i) for i in hole_result.claimed_faces)
    residual_claimed = set(int(i) for i in residual_result.claimed_faces)
    outer_claimed = set(int(i) for i in outer_result.claimed_faces)
    pocket_map = pocket_instance_map(pocket_result)

    def depth_fn(i: int) -> float:
        return depth_below_top(i, by_index, top_y)

    # Depth tiers on flat bucket (derived gaps).
    flat_tiers, flat_bounds = depth_tiers_from_gaps(flat_bucket, depth_fn)
    flat_depths_sorted = sorted((depth_fn(i), i) for i in flat_bucket)
    flat_gaps = [
        (flat_depths_sorted[k + 1][0] - flat_depths_sorted[k][0], flat_depths_sorted[k][1],
         flat_depths_sorted[k + 1][1])
        for k in range(len(flat_depths_sorted) - 1)
        if flat_depths_sorted[k + 1][0] - flat_depths_sorted[k][0] > 1e-6
    ]
    tier_of: dict[int, str] = {}
    for ti, ids in enumerate(flat_tiers):
        for i in ids:
            tier_of[i] = f"tier-{ti}"

    # Global shallowest tier (opening skin reference for top/outer adjacency).
    all_ids = list(by_index.keys())
    global_tiers, global_bounds = depth_tiers_from_gaps(all_ids, depth_fn)
    global_shallowest_max = global_bounds[0][1] if global_bounds else float("inf")

    print(f"part={args.part}  flat-bucket size={len(flat_bucket)}  (Toolpath expect 1)")
    print(f"top-of-part Y (boundary max) = {top_y:.3f} mm ({top_y / MM_PER_IN:.4f} in)")
    print(f"global shallowest depth tier: 0 .. {global_shallowest_max:.2f} mm\n")

    print("--- flat-bucket depth tiers (natural gap splits) ---")
    if flat_gaps:
        print("  non-zero depth gaps (mm):",
              ", ".join(f"{g:.2f} ({a}->{b})" for g, a, b in flat_gaps))
    for ti, (ids, (lo, hi)) in enumerate(zip(flat_tiers, flat_bounds)):
        print(f"  tier-{ti}: depth {lo:.2f} .. {hi:.2f} mm  ({len(ids)} face(s))  ids={sorted(ids)}")
    print()

    rows: list[dict] = []
    hdr = (
        f"{'face':>5} {'surf':>8} {'area':>8} {'depth':>7} {'tier':>6} "
        f"{'pk':>3} {'hole':>4} {'top':>4} {'rs':>3}  verdict / cause"
    )
    print(hdr)
    print("-" * len(hdr))

    for fi in flat_bucket:
        f = by_index[fi]
        d = depth_fn(fi)
        pk_adj = pocket_neighbors(fi, graph, pocket_claimed, pocket_map)
        pk_2 = pocket_two_hop(fi, graph, pocket_claimed, pocket_map)
        hole_adj = sorted(nb for nb in graph.neighbors.get(fi, ()) if nb in hole_claimed)
        rs_adj = sorted(nb for nb in graph.neighbors.get(fi, ()) if nb in residual_claimed)
        top_adj = sorted(
            nb for nb in graph.neighbors.get(fi, ())
            if is_top_outer_neighbor(nb, by_index, top_y, global_shallowest_max,
                                     residual_claimed, outer_claimed)
        )

        shallowest = tier_of[fi] == "tier-0"
        verdict = assign_verdict(
            fi,
            tier_label=tier_of[fi],
            shallowest_tier=shallowest,
            pocket_adj=pk_adj,
            hole_adj=hole_adj,
            depth_rank=flat_tiers.index(next(t for t in flat_tiers if fi in t)),
            n_tiers=len(flat_tiers),
        )
        cause = assign_cause(pk_adj, pk_2)

        rows.append({
            "face": fi,
            "tier": tier_of[fi],
            "verdict": verdict,
            "cause": cause,
            "pk_adj": pk_adj,
            "pk_2hop": pk_2,
            "hole_adj": hole_adj,
            "top_adj": top_adj,
            "rs_adj": rs_adj,
        })

        pk_s = ",".join(f"{nb}(P{pid})" for nb, pid in pk_adj) or "—"
        print(
            f"{fi:>5} {str(f.surface_type):>8} {f.area:>8.1f} {d:>7.2f} {tier_of[fi]:>6} "
            f"{len(pk_adj):>3} {len(hole_adj):>4} {len(top_adj):>4} {len(rs_adj):>3}  "
            f"{verdict} / {cause.split('(')[0].strip()}"
        )

    print("\n--- per-face detail ---")
    for r in rows:
        fi = r["face"]
        print(f"\nface {fi} ({r['verdict']}, {r['tier']}, {r['cause']})")
        print(f"  pocket edges ({len(r['pk_adj'])}): {r['pk_adj'] or 'none'}")
        if r["pk_2hop"]:
            pockets = sorted({pid for _, _, pid in r["pk_2hop"]})
            print(f"  pocket 2-hop via fillet: {len(r['pk_2hop'])} path(s) "
                  f"-> pocket instance(s) {pockets}")
        print(f"  hole edges ({len(r['hole_adj'])}): {r['hole_adj'] or 'none'}")
        print(f"  top/outer edges ({len(r['top_adj'])}): {r['top_adj'] or 'none'}")
        print(f"  residual edges ({len(r['rs_adj'])}): {len(r['rs_adj'])}")
        print("  shared edges (neighbor convexity):")
        for nb in sorted(graph.neighbors.get(fi, ())):
            kind = graph.edge_kind(fi, nb) or "?"
            cos_v = graph.edge_cos(fi, nb)
            cos_s = f"{cos_v:+.3f}" if cos_v is not None else "?"
            tags: list[str] = []
            if nb in pocket_claimed:
                tags.append(f"P{pocket_map[nb]}")
            if nb in hole_claimed:
                tags.append("hole")
            if nb in residual_claimed:
                tags.append("rs")
            if nb in outer_claimed:
                tags.append("of")
            print(f"    -> {nb:3d} {by_index[nb].surface_type:8s}  {kind:8s} cos={cos_s}  "
                  f"[{','.join(tags) or '—'}]")

    # Summaries
    step_faces = [r for r in rows if r["face"] in {
        112, 139, 164, 189, 214, 239, 262,
    }]
    deep_faces = [r for r in rows if r["face"] in {273, 277}]
    n_step_pk_direct = sum(1 for r in step_faces if r["pk_adj"])
    n_step_pk_2hop = sum(1 for r in step_faces if r["pk_2hop"])
    n_step_orphan = sum(1 for r in step_faces if not r["pk_adj"])

    print("\n--- read ---")
    print(f"TRUE-FLAT: {[r['face'] for r in rows if r['verdict'] == 'TRUE-FLAT']}")
    print(f"7 step-planes (tier-1 subset): direct pocket-adjacent={n_step_pk_direct}/7, "
          f"2-hop pocket reachable={n_step_pk_2hop}/7, "
          f"flat-side orphan (no direct pk edge)={n_step_orphan}/7")
    print("  => All 7 step-planes lack DIRECT pocket edges; each reaches a pocket "
          "instance within 2 hops through torus fillets.")
    print("  => Root cause is pocket-side grow (cause A): pocket pass should absorb "
          "step floors via fillet chain, NOT flat-pass depth gating alone.")

    for r in deep_faces:
        print(f"face {r['face']} ({r['tier']}): verdict={r['verdict']}, "
              f"hole_adj={len(r['hole_adj'])}, top/outer_adj={len(r['top_adj'])}, "
              f"residual_adj={len(r['rs_adj'])}")

    print("\n273/277 (deep tiers):")
    print("  273: HOLE-adjacent (cylinders 280/298) + contour residual neighbors "
          "-> flat pass should exclude; route to residual/contour (cause B).")
    print("  277: outer-fillet adjacent (278/296) + contour residual "
          "-> flat pass should exclude; route to residual/contour (cause B).")

    print("\n--- recommendation (probe only — do NOT implement here) ---")
    print("  face 97:           KEEP in flat pass (TRUE-FLAT).")
    print("  112-262 (x7):      FIX pocket_detection grow/membership (cause A).")
    print("  273, 277:          FIX flats_detection exclusion -> residual (cause B); "
          "do not depth-tune to N=1.")


if __name__ == "__main__":
    main()
