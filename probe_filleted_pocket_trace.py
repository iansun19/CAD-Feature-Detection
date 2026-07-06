"""Trace lobe-6 closed tier admission for feature_id=1 stray faces (read-only)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cascade_instrumentation import _trace_split_lobe_pool_assignments
from feature_params import analyze_step
from hole_detection import FaceGraph
from lobe_tier_detection import LobeTierConfig, detect_filleted_lobe_tiers

STEP = "96260B_REAR_XR004_PCD PLATE.stp copy"
GRAPH = Path("pipeline_out/96260B_plate/graph.npz")
TARGET = {97, 103, 344}
FEAT1 = [97, 103, 300, 302, 303, 304, 305, 306, 307, 308, 309, 310, 316, 317, 318, 319, 320, 321, 344]


def load_graph():
    data = np.load(GRAPH)
    return data["edge_index"], data["edge_attr"]


def induced_components(face_ids, adj):
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


def trace_region_grow_admissions(pool, mouth_step, deep_step, graph, by_index, occ_map, opening_axis, cfg):
    """Mirror _region_grow_tier closed tier with per-face admission log."""
    from lobe_tier_detection import (
        _mouth_tier_boundary_tol_mm,
        _region_grow_tier,
        _lobe_traversable,
        _composition_tier,
        _lobe_edge_ok,
        _axial_side_of_mouth,
        _fillet_above_mouth_step,
        _fillet_has_concave_cone_seam,
        _deep_wall_pair_member,
        WALL_SURF_TYPES,
        FILLET_TYPES,
    )

    mouth_boundary_tol = _mouth_tier_boundary_tol_mm(
        pool, mouth_step, deep_step, by_index, occ_map, graph, opening_axis, cfg,
    )
    deep_steps = {deep_step}
    seeds = {deep_step}
    growing = set(seeds) & pool
    admissions = {s: {"via": None, "seed": True, "from": None} for s in growing}

    from collections import deque
    q = deque(sorted(growing))
    while q:
        u = q.popleft()
        for v in graph.neighbors.get(u, ()):
            if v not in pool or v in growing:
                continue
            if not _lobe_traversable(v, by_index, occ_map, opening_axis, cfg):
                continue
            kind = graph.edge_kind(u, v)
            ufg = by_index[u]
            vfg = by_index[v]
            comp = _composition_tier(v, by_index, occ_map, graph, cfg)
            tier_hint = "closed"
            if tier_hint == "closed" and vfg.surface_type in FILLET_TYPES:
                if _fillet_above_mouth_step(v, mouth_step, by_index, opening_axis):
                    continue
                if _fillet_has_concave_cone_seam(v, graph, by_index):
                    continue
            if kind == "convex" and tier_hint == "closed":
                if vfg.surface_type in WALL_SURF_TYPES:
                    continue
                if vfg.surface_type == "sphere" and ufg.surface_type in WALL_SURF_TYPES:
                    continue
            if kind == "concave" and tier_hint == "closed":
                if (
                    ufg.surface_type in WALL_SURF_TYPES
                    and vfg.surface_type in WALL_SURF_TYPES
                    and not _deep_wall_pair_member(v, by_index, occ_map, graph, cfg)
                ):
                    continue
                if ufg.surface_type == "sphere" and vfg.surface_type == "sphere":
                    continue
            if kind == "smooth" and tier_hint == "closed":
                if ufg.surface_type in FILLET_TYPES and vfg.surface_type in WALL_SURF_TYPES:
                    from lobe_tier_detection import _wall_interior_and_axis, _near_dia_in
                    vinfo = _wall_interior_and_axis(vfg, occ_map.get(v) if occ_map else None)
                    if vinfo and _near_dia_in(vinfo[2], 0.800, cfg.dia_family_tol_in):
                        continue
            if not _lobe_edge_ok(graph, u, v, growing):
                continue
            if comp == "open":
                continue
            growing.add(v)
            admissions[v] = {"via": "_region_grow_tier", "seed": False, "from": u, "edge_kind": kind}
            q.append(v)

    # compare to library impl
    lib = _region_grow_tier(
        seeds, pool, graph, by_index, occ_map, opening_axis, cfg,
        tier_hint="closed", mouth_step=mouth_step,
        mouth_boundary_tol_mm=mouth_boundary_tol, deep_step_faces=deep_steps,
    )
    diff = lib.symmetric_difference(growing)
    return growing, admissions, lib, diff, mouth_boundary_tol


def main():
    faces = analyze_step(STEP)
    ei, ea = load_graph()
    by_index = {f.index: f for f in faces}
    occ_map = None
    graph = FaceGraph.from_edge_tensors(ei, ea, len(faces))
    cfg = LobeTierConfig()

    lobe = detect_filleted_lobe_tiers(
        faces, ei, ea, occ_faces=None, opening_axis=[0, 1, 0], config=cfg,
    )
    print(f"lobes: {len(lobe.lobes)}")
    lobe6 = None
    for i, lob in enumerate(lobe.lobes):
        if TARGET & lob.pool_faces:
            print(f"\nlobe index={i} pool_size={len(lob.pool_faces)}")
            print(f"  mouth_step={lob.mouth_step_face} deep_step={lob.deep_step_face}")
            print(f"  open={len(lob.open_faces)} closed={len(lob.closed_faces)}")
            hits = TARGET & (lob.open_faces | lob.closed_faces)
            print(f"  target hits in tier sets: {sorted(hits)}")
            if TARGET <= lob.pool_faces:
                lobe6 = lob
                lobe_idx = i

    if lobe6 is None:
        for i, lob in enumerate(lobe.lobes):
            if TARGET & lob.pool_faces:
                lobe6 = lob
                lobe_idx = i
                break

    pool = lobe6.pool_faces
    mouth_step = lobe6.mouth_step_face
    deep_step = lobe6.deep_step_face
    opening_axis = np.array([0.0, 1.0, 0.0])

    # adjacency for pool / closed
    adj = defaultdict(set)
    for a, b in zip(ei[0], ei[1]):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    open_f, closed_f, assignments = _trace_split_lobe_pool_assignments(
        pool, mouth_step, deep_step, graph, by_index, occ_map, opening_axis, cfg,
    )

    print(f"\n=== split_lobe_pool trace (lobe {lobe_idx}) ===")
    print(f"closed tier size: {len(closed_f)}")
    print(f"closed tier components in pool: {len(induced_components(sorted(closed_f), adj))}")
    print(f"closed tier components sizes: {[len(c) for c in induced_components(sorted(closed_f), adj)]}")

    for fid in sorted(TARGET):
        in_closed = fid in closed_f
        in_open = fid in open_f
        assign = assignments.get(fid, "NOT_IN_TIER")
        print(
            f"  face {fid}: closed={in_closed} open={in_open} "
            f"tier_assignment={assign} surf={by_index[fid].surface_type}"
        )

    growing, admissions, lib_closed, diff, tol = trace_region_grow_admissions(
        pool, mouth_step, deep_step, graph, by_index, occ_map, opening_axis, cfg,
    )
    print(f"\nregion_grow replay diff vs lib: {sorted(diff) if diff else 'none'}")
    for fid in sorted(TARGET):
        if fid in admissions:
            a = admissions[fid]
            in_growing = fid in growing
            print(f"  face {fid} region_grow admission: {a} in_set={in_growing}")

    # After full trace: which function added targets not via region_grow?
    print("\n=== post-region_grow additions ===")
    open_seeds, closed_seeds = {mouth_step}, {deep_step}
    rg_open = set()
    rg_closed = set(growing)
    for fid in sorted(TARGET):
        if fid in closed_f:
            if fid in rg_closed:
                fn = "_region_grow_tier"
                detail = admissions.get(fid, {})
            elif assignments.get(fid) == "annexed_fillet":
                fn = "_annex_fillets"
                detail = {"reason": "torus_borders_pocket_walls or sphere_cap"}
            elif assignments.get(fid) == "orphan_reassigned":
                fn = "_reassign_orphan_lobe_faces"
                detail = {"reason": assignments[fid]}
            elif assignments.get(fid) == "closed_tier_open_ext":
                fn = "_migrate_closed_tier_opening_extension_walls / _closed_tier_opening_extension_wall"
                detail = {}
            else:
                fn = f"unknown ({assignments.get(fid)})"
                detail = {}
            # edge adjacency to rest of closed tier at admission is hard to replay;
            # check current adjacency to closed_f - {fid}
            others = closed_f - {fid}
            nbrs = sorted(adj[fid] & others)
            print(f"  face {fid}: adding_function={fn} assignment={assignments.get(fid)} "
                  f"edge_adjacent_to_other_closed={nbrs}")

    # Feature 1 vs lobe closed set
    feat1_set = set(FEAT1)
    print(f"\nfeature_id=1 faces not in lobe closed: {sorted(feat1_set - closed_f)}")
    print(f"feature_id=1 faces not in lobe open: {sorted(feat1_set - open_f)}")
    print(f"lobe closed not in feature1: {sorted(closed_f - feat1_set)[:20]}...")

    # Step 4 detail
    print("\n=== faces 195/196/199/201 ===")
    for node in json.load(open("pipeline_out/96260B_rear/feature_graph_cascade.json"))["nodes"]:
        hits = set(node["face_ids"]) & {195, 196, 199, 201}
        if hits:
            comps = induced_components(node["face_ids"], adj)
            print(
                f"fid={node['feature_id']} {node['class_name']} hits={sorted(hits)} "
                f"n_comp={len(comps)} sizes={[len(c) for c in comps]}"
            )


if __name__ == "__main__":
    main()
