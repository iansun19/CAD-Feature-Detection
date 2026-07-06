"""
probe_cascade_24face_trace.py — instrumented claim/release trace for the 24
96260B_front mislabeled faces. Logs which pass claims or releases each face and
the exact predicate that fired.

Run:
  python probe_cascade_24face_trace.py
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from coaxial_stack_detection import detect_coaxial_stack
from eval_cascade import build_cascade_feature_graph
from feature_params import analyze_step, load_step_faces
from flats_detection import detect_flats
from hole_detection import (
    FaceGraph,
    HoleDetectionConfig,
    detect_holes,
    _classify_cluster,
    _split_hole_stack_tiers,
    _attach_faces,
    _primary_cylinder,
    _wall_axial_span,
)
from hole_detection import _cluster_candidates  # noqa: F401 — used indirectly
from lobe_tier_detection import (
    LobeTierConfig,
    _adjacent_shallow_bore_cylinder,
    _annex_fillets,
    _collect_axial_step_planes,
    _composition_tier,
    _open_fillet_should_drop,
    _prune_closed_tier_fillets,
    _prune_deep_step_interface_tori,
    _prune_open_tier_fillets,
    _prune_paired_floor_spheres,
    _mouth_tier_boundary_tol_mm,
    _region_grow_tier,
    _resolve_overlap,
    assign_lobe_faces_angular,
    detect_filleted_lobe_tiers,
    discover_mouth_and_deep_steps,
    pair_mouth_deep_steps,
    refine_pool_by_fillet_connectivity,
    split_lobe_pool,
)
from outer_fillet_detection import detect_outer_fillets
from pocket_detection import (
    PocketDetectionConfig,
    WALL_SURF_TYPES,
    _is_grow_candidate,
    _part_axis_top,
    _pocket_candidates_for_face,
    _pick_pocket_assignment,
    _wall_interior_and_axis,
    apply_filleted_lobe_tiers_to_result,
    detect_pockets,
    resolve_pocket_setup_for_run,
)
from profile_detection import detect_profiles
from residual_detection import detect_residual_candidates
from run_cascade import _load_edges
from wall_detection import detect_walls

TARGET = frozenset({
    192, 193, 252, 254, 102, 122, 137, 147, 162, 167, 172, 187,
    212, 217, 222, 229, 237, 247, 270,
    52, 84,
    234, 238, 244,
})

STEP = Path("96260B_FRONT_XR004_PCD PLATE.stp copy")
GRAPH = Path("pipeline_out/96260B_front/graph.npz")


@dataclass
class Event:
    face: int
    action: str  # claim | release
    stage: str
    predicate: str
    detail: str = ""


@dataclass
class TraceLog:
    events: dict[int, list[Event]] = field(default_factory=lambda: defaultdict(list))

    def log(self, face: int, action: str, stage: str, predicate: str, detail: str = "") -> None:
        if face in TARGET:
            self.events[face].append(Event(face, action, stage, predicate, detail))

    def render(self) -> str:
        lines: list[str] = []
        for fid in sorted(TARGET):
            lines.append(f"\n{'=' * 72}")
            lines.append(f"Face {fid}")
            lines.append(f"{'=' * 72}")
            evs = self.events.get(fid, [])
            if not evs:
                lines.append("  (no events — face never touched by instrumented stages)")
                continue
            for ev in evs:
                lines.append(f"  [{ev.action:7s}] {ev.stage}")
                lines.append(f"           predicate: {ev.predicate}")
                if ev.detail:
                    lines.append(f"           detail:    {ev.detail}")
        return "\n".join(lines)


def _trace_spatial_pocket(
    log: TraceLog,
    pk,
    faces,
    occ_map,
    edge_index,
    edge_attr,
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
) -> None:
    by_index = {i: faces[i] for i in range(len(faces))}
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(faces))

    for fid in sorted(TARGET):
        if fid in pk.claimed_faces:
            log.log(
                fid, "claim", "pass1_spatial_pocket",
                "detect_pockets: interior wall in diameter band + within wall_attach_dist of floor seed cluster",
            )
        elif fid in pk.remaining_faces:
            fg = by_index[fid]
            if fg.surface_type in WALL_SURF_TYPES:
                info = _wall_interior_and_axis(fg, occ_map.get(fid))
                if info is None:
                    log.log(fid, "release", "pass1_spatial_pocket", "not a wall surface (_wall_interior_and_axis=None)")
                elif config.require_interior_wall and not info[0]:
                    log.log(fid, "release", "pass1_spatial_pocket", "require_interior_wall and face is convex/exterior")
                else:
                    d = info[2] or 0.0
                    if config.wall_dia_min_mm is not None and d < config.wall_dia_min_mm:
                        log.log(fid, "release", "pass1_spatial_pocket",
                                 f"wall_dia_min_mm: diameter {d:.2f}mm < {config.wall_dia_min_mm}")
                    elif config.wall_dia_max_mm is not None and d > config.wall_dia_max_mm:
                        log.log(fid, "release", "pass1_spatial_pocket",
                                 f"wall_dia_max_mm: diameter {d:.2f}mm > {config.wall_dia_max_mm}")
                    elif fid in pk.unattached_wall_faces:
                        log.log(fid, "release", "pass1_spatial_pocket",
                                 f"wall_attach_dist: UV centroid > {pk.wall_attach_dist:.2f}mm from all pocket seeds")
                    else:
                        grow = _is_grow_candidate(fg, occ_map.get(fid), opening_axis, config)
                        if not grow:
                            log.log(fid, "release", "pass1_spatial_pocket",
                                     f"_is_grow_candidate=False (surface_type={fg.surface_type})")
                        else:
                            log.log(fid, "release", "pass1_spatial_pocket",
                                     "not wall-attached and membership grow did not assign (no adjacent claimed pocket)")
            elif fg.surface_type == "torus":
                log.log(fid, "release", "pass1_spatial_pocket",
                         "_is_grow_candidate=False: torus rejected by grow gate (not wall-attached seed)")
            else:
                log.log(fid, "release", "pass1_spatial_pocket",
                         f"not a pocket wall candidate (surface_type={fg.surface_type})")


def _trace_lobe_tier_split(log: TraceLog, pk_spatial, pk_tier, lobe, faces, occ_map, edge_index, edge_attr) -> None:
    cfg = LobeTierConfig()
    by_index = {i: faces[i] for i in range(len(faces))}
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(faces))
    axis = np.asarray(pk_spatial.opening_axis, dtype=float)

    pool_all: set[int] = set()
    for lob in lobe.lobes:
        pool_all |= lob.pool_faces

    for fid in sorted(TARGET):
        if fid not in pool_all:
            if fid in pk_spatial.claimed_faces:
                log.log(fid, "release", "pass1b_lobe_tier_split",
                         "apply_filleted_lobe_tiers: face NOT in any lobe pool — spatial claim kept")
            continue

        if fid in pk_spatial.claimed_faces:
            log.log(fid, "release", "pass1b_lobe_tier_split",
                     "apply_filleted_lobe_tiers: spatial pocket discarded (face_indices & pool_all)")

        if fid in pk_tier.claimed_faces:
            for lob in lobe.lobes:
                if fid in lob.open_faces:
                    log.log(fid, "claim", "pass1b_lobe_tier_split",
                             "lobe tier: assigned to open_faces",
                             f"lobe_id={lob.lobe_id} mouth={lob.mouth_step_face} deep={lob.deep_step_face}")
                    break
                if fid in lob.closed_faces:
                    log.log(fid, "claim", "pass1b_lobe_tier_split",
                             "lobe tier: assigned to closed_faces",
                             f"lobe_id={lob.lobe_id} mouth={lob.mouth_step_face} deep={lob.deep_step_face}")
                    break
            continue

        # Unassigned — trace tier split predicates
        for lob in lobe.lobes:
            if fid not in lob.pool_faces:
                continue
            mouth, deep = lob.mouth_step_face, lob.deep_step_face
            pool = lob.pool_faces
            mouth_boundary_tol = _mouth_tier_boundary_tol_mm(
                pool, mouth, deep, by_index, occ_map, axis, cfg,
            )
            open0 = _region_grow_tier(
                {mouth}, pool, graph, by_index, occ_map, axis, cfg,
                tier_hint="open", mouth_step=mouth,
                mouth_boundary_tol_mm=mouth_boundary_tol, deep_step_faces={deep},
            )
            closed0 = _region_grow_tier(
                {deep}, pool, graph, by_index, occ_map, axis, cfg,
                tier_hint="closed", mouth_step=mouth,
                mouth_boundary_tol_mm=mouth_boundary_tol, deep_step_faces={deep},
            )

            fg = by_index[fid]
            comp = _composition_tier(fid, by_index, occ_map, graph, cfg)
            shallow = _adjacent_shallow_bore_cylinder(fid, by_index, occ_map, graph, cfg)

            if fid in open0:
                stage_pred = "region_grow_tier(open): reached from mouth step"
            elif fid in closed0:
                stage_pred = "region_grow_tier(closed): reached from deep step"
            elif fg.surface_type in WALL_SURF_TYPES:
                if shallow:
                    stage_pred = (
                        "region_grow blocked: _adjacent_shallow_bore_cylinder "
                        "(⌀0.800in concave-adjacent to ⌀2.867in) excluded from open tier; "
                        "not reached by closed tier"
                    )
                else:
                    stage_pred = (
                        "region_grow blocked: closed tier skips convex edge into WALL_SURF_TYPES; "
                        "open tier composition/comp gate blocked wall"
                    )
            elif fg.surface_type == "torus":
                stage_pred = "_is_grow_candidate/spatial: torus not in tier grow seeds; annex/prune dropped or skipped"
            else:
                stage_pred = f"region_grow: not reached from mouth/deep seeds (surface_type={fg.surface_type})"

            open1, sphere_cap_logs = _annex_fillets(
                open0, pool, graph, by_index,
                occ_map=occ_map,
                cfg=cfg,
                skip=lambda f, o0=open0: _open_fillet_should_drop(
                    f, o0, graph, by_index, occ_map, cfg, {deep},
                ),
            )
            closed1, _ = _annex_fillets(
                closed0, pool, graph, by_index,
                occ_map=occ_map,
                cfg=cfg,
            )
            if sphere_cap_logs:
                closed1 -= {entry["sphere_face"] for entry in sphere_cap_logs}
            open2 = _prune_open_tier_fillets(open1, graph, by_index, occ_map, cfg, {deep})
            closed2 = _prune_closed_tier_fillets(
                closed1, mouth, deep, graph, by_index, occ_map, cfg, axis,
            )
            closed3 = _prune_paired_floor_spheres(closed2, by_index, graph, axis)

            prune_detail = []
            if fid in closed1 and fid not in closed2:
                if fg.surface_type in ("torus", "cylinder", "cone", "sphere", "bspline", "bezier"):
                    after_iface = _prune_deep_step_interface_tori(set(closed1), deep, graph, by_index, axis)
                    if fid in closed1 and fid not in after_iface:
                        prune_detail.append("_prune_deep_step_interface_tori: shallow interface torus at deep step")
            if fid in closed2 and fid not in closed3:
                prune_detail.append("_prune_paired_floor_spheres: deeper sphere of concave pair discarded")

            if fid in lob.unassigned_faces:
                log.log(
                    fid, "release", "pass1b_lobe_tier_split",
                    f"split_lobe_pool → unassigned_faces: {stage_pred}",
                    "; ".join(prune_detail) if prune_detail else (
                        f"lobe_id={lob.lobe_id} comp={comp} open0={fid in open0} closed0={fid in closed0} "
                        f"open_final={fid in lob.open_faces} closed_final={fid in lob.closed_faces}"
                    ),
                )
            log.log(
                fid, "release", "pass1b_lobe_tier_remaining",
                "remaining_faces |= released spatial claims; remaining -= tier claims",
                f"in pk_tier.remaining_faces={fid in pk_tier.remaining_faces}",
            )
            break


def _trace_holes(log: TraceLog, pk_tier, hl, faces, occ_map, edge_index, edge_attr) -> None:
    hole_cfg = HoleDetectionConfig(max_hole_diameter_mm=150.0)
    by_index = {i: faces[i] for i in range(len(faces))}
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(faces))

    for fid in sorted(TARGET):
        if fid not in hl.claimed_faces:
            if fid in pk_tier.remaining_faces:
                log.log(fid, "release", "pass2_holes", "not clustered as hole candidate (no matching cluster)")
            continue

        for i, feat in enumerate(hl.features):
            if fid not in feat.face_indices:
                continue
            kind = feat.kind
            if kind == "through_hole":
                pred = (
                    "_classify_cluster: cluster has cylindrical wall(s), "
                    "nominal_diameter <= max_hole_diameter_mm, "
                    "body_floors empty → kind='through_hole' (wall open at both ends)"
                )
            else:
                pred = f"_classify_cluster: kind={kind!r}"
            log.log(
                fid, "claim", "pass2_holes", pred,
                f"hole_feature #{i} faces={sorted(feat.face_indices)} "
                f"blend={getattr(feat, 'blend_face_indices', [])}",
            )

            # Deferred boundary fillets
            for grp in hl.deferred_feature_fillet_groups:
                if fid in grp:
                    log.log(
                        fid, "release", "pass2_holes_deferred",
                        "_split_hole_stack_tiers → boundary_fillets deferred to outer_fillet pass",
                        f"deferred_group={sorted(grp)}",
                    )


def _trace_outer_fillets(log: TraceLog, of, hl) -> None:
    deferred = {i for grp in hl.deferred_feature_fillet_groups for i in grp}
    for fid in sorted(TARGET):
        if fid not in of.claimed_faces:
            continue
        for i, feat in enumerate(of.features):
            if fid not in feat.face_indices:
                continue
            if fid in deferred:
                log.log(
                    fid, "claim", "pass5_outer_fillets",
                    "detect_outer_fillets Pass A: stack_boundary_fillet_groups from hole pass deferral",
                    f"outer_fillet_feature #{i} faces={sorted(feat.face_indices)}",
                )
            else:
                log.log(
                    fid, "claim", "pass5_outer_fillets",
                    "detect_outer_fillets Pass B/C: structural or hub-perimeter outer fillet",
                    f"outer_fillet_feature #{i}",
                )


def _trace_residual(log: TraceLog, rs, graph_labels: dict[int, str]) -> None:
    for fid in sorted(TARGET):
        if fid in rs.claimed_faces:
            log.log(
                fid, "claim", "pass8_residual",
                "detect_residual_candidates: unclaimed face → contour_surface (residual pool)",
                f"final_class={graph_labels.get(fid, '?')}",
            )
        elif fid in graph_labels:
            log.log(
                fid, "release", "pass8_residual",
                f"claimed by earlier pass → final_class={graph_labels[fid]}",
            )


def run_trace() -> TraceLog:
    log = TraceLog()
    edge_index, edge_attr = _load_edges(GRAPH, STEP)
    config = PocketDetectionConfig(setup=resolve_pocket_setup_for_run(STEP, machining_side="front"))
    faces = analyze_step(STEP)
    occ_list = load_step_faces(STEP)
    occ_map = {i: occ_list[i] for i in range(len(faces))}

    pk_spatial = detect_pockets(
        faces, edge_index, edge_attr, occ_faces=occ_list, config=config,
    )
    _trace_spatial_pocket(log, pk_spatial, faces, occ_map, edge_index, edge_attr,
                          np.asarray(pk_spatial.opening_axis), config)

    lobe = detect_filleted_lobe_tiers(
        faces, edge_index, edge_attr, occ_faces=occ_list,
        opening_axis=pk_spatial.opening_axis, config=LobeTierConfig(),
    )
    pk_tier = apply_filleted_lobe_tiers_to_result(
        pk_spatial, faces, edge_index, edge_attr, occ_list, config,
    )
    _trace_lobe_tier_split(log, pk_spatial, pk_tier, lobe, faces, occ_map, edge_index, edge_attr)

    hl = detect_holes(
        faces, edge_index, edge_attr, occ_faces=occ_list,
        candidate_faces=pk_tier.remaining_faces,
    )
    _trace_holes(log, pk_tier, hl, faces, occ_map, edge_index, edge_attr)

    opening_axis = pk_tier.opening_axis
    part_top = _part_axis_top(faces, np.asarray(opening_axis), occ_map)
    cx = detect_coaxial_stack(
        faces, edge_index, edge_attr,
        candidate_faces=hl.remaining_faces,
        hole_claimed_faces=hl.claimed_faces,
        hole_features=hl.features,
        opening_axis=opening_axis,
        part_axis_top=part_top,
    )
    hub_open = {i for f in cx.features if f.kind == "open_pocket" for i in f.face_indices}
    fl = detect_flats(
        faces, edge_index, edge_attr,
        candidate_faces=cx.remaining_faces,
        hole_claimed_faces=hl.claimed_faces,
        pocket_claimed_faces=pk_tier.claimed_faces,
        pocket_floor_absorbed_faces=pk_tier.floor_absorbed_faces,
        hub_flat_faces=cx.hub_flat_faces,
        opening_axis=opening_axis,
        occ_faces=occ_list,
    )
    of = detect_outer_fillets(
        faces, edge_index, edge_attr,
        candidate_faces=fl.remaining_faces,
        pocket_claimed_faces=pk_tier.claimed_faces,
        stack_boundary_fillet_groups=hl.deferred_feature_fillet_groups,
        flat_claimed_faces=fl.claimed_faces,
        hub_open_pocket_faces=hub_open,
        hub_perimeter_context=cx.hub_perimeter_context,
        opening_axis=opening_axis,
        part_axis_top=part_top,
    )
    _trace_outer_fillets(log, of, hl)

    wl = detect_walls(
        faces, edge_index, edge_attr,
        candidate_faces=of.remaining_faces,
        pocket_claimed_faces=pk_tier.claimed_faces,
        hole_claimed_faces=hl.claimed_faces,
        hub_open_pocket_faces=hub_open,
        opening_axis=opening_axis,
    )
    pr = detect_profiles(
        faces, edge_index, edge_attr,
        candidate_faces=wl.remaining_faces,
        pocket_claimed_faces=pk_tier.claimed_faces,
        hole_claimed_faces=hl.claimed_faces,
        hub_open_pocket_faces=hub_open,
        wall_claimed_faces=wl.claimed_faces,
        wall_seed_faces=wl.seed_faces,
        opening_axis=opening_axis,
    )
    rs = detect_residual_candidates(
        faces, edge_index, edge_attr,
        candidate_faces=pr.remaining_faces,
    )

    graph = build_cascade_feature_graph(
        "96260B_front", len(faces), pk_tier, hl, cx, fl, of, wl, pr, rs, edge_index,
    )
    labels = {}
    for node in graph["nodes"]:
        for fid in node["face_ids"]:
            labels[fid] = node["class_name"]
    _trace_residual(log, rs, labels)
    return log


def main() -> int:
    trace = run_trace()
    print(trace.render())

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY — final owning pass per face")
    print("=" * 72)
    print(f"{'face':>4}  {'final_pass':<22}  key predicate chain")
    print("-" * 72)
    for fid in sorted(TARGET):
        evs = trace.events.get(fid, [])
        claims = [e for e in evs if e.action == "claim"]
        final = claims[-1].stage if claims else "?"
        # compress predicate chain
        chain_parts = []
        for e in evs:
            if e.action == "release" and "lobe_tier_split" in e.stage:
                chain_parts.append("tier-unassigned")
            elif e.action == "claim" and e.stage == "pass2_holes":
                chain_parts.append("hole:through")
            elif e.action == "claim" and e.stage == "pass5_outer_fillets":
                chain_parts.append("outer_fillet")
            elif e.action == "claim" and e.stage == "pass8_residual":
                chain_parts.append("contour")
            elif e.action == "claim" and "lobe_tier" in e.stage:
                chain_parts.append("tier-pocket")
        print(f"{fid:4d}  {final:<22}  {' → '.join(chain_parts) or '(see above)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
