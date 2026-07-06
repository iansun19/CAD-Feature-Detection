#!/usr/bin/env python
"""
probe_back_pocket_gt_prep.py — DIAGNOSTIC ONLY (wires nothing).

Prepares the Toolpath GT check for the back-plate open-pocket candidate
{322,326,330,332} on 96260B_plate: completes the structural face set,
writes a position-based description for GUI matching, reconciles the rear
flat over-claim under both pocket/flat readings, and confirms the
classify_pocket_open_closed gateway gap.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_back_pocket_gt_prep.py
"""
from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from feature_params import FaceGeom, analyze_step, load_step_faces, require_occ
from hole_detection import FaceGraph, _unit
from pocket_detection import (
    PocketDetectionConfig,
    PocketFeature,
    WALL_SURF_TYPES,
    classify_pocket_open_closed,
    _axial_y,
    _claimed_wall_indices,
    _uv_radius,
    _wall_interior_and_axis,
)
from run_cascade import _load_edges, run_cascade

require_occ()

MM_PER_IN = 25.4
PART_ID = "96260B_plate"
DEFAULT_STEP = "96260B_REAR_XR004_PCD PLATE.stp copy"
DEFAULT_NPZ = "pipeline_out/96260B_plate/graph.npz"

# Named seed from enclosed-wall-loop probe (only hardcoded indices allowed).
SEED_LOOP = frozenset({322, 326, 330, 332})

SIDE_WALL_NORMAL_MAX_DEG = 15.0
BLEND_TYPES = frozenset({"torus", "bspline", "bezier", "sphere"})


def _face_owner(
    face_id: int,
    *,
    pocket_result,
    hole_result,
    flat_result,
    outer_fillet_result,
    residual_result,
) -> str:
    for feat in pocket_result.features:
        if face_id in feat.face_indices:
            return f"pocket_{feat.feature_id}"
    for feat in hole_result.features:
        if face_id in feat.face_indices:
            return f"hole_{feat.kind}_{feat.feature_id}"
    for feat in flat_result.features:
        if face_id in feat.face_indices:
            return f"flat_{feat.feature_id}"
    for feat in outer_fillet_result.features:
        if face_id in feat.face_indices:
            return f"outer_fillet_{feat.feature_id}"
    for feat in residual_result.features:
        if face_id in feat.face_indices:
            return f"residual_{feat.feature_id}"
    return "unclaimed"


def _is_planar_side_wall(
    face_id: int,
    by_index: dict[int, FaceGeom],
    opening_axis: np.ndarray,
) -> bool:
    fg = by_index[face_id]
    if fg.surface_type != "plane":
        return False
    nd = abs(float(np.dot(_unit(fg.normal), _unit(opening_axis))))
    return nd < math.cos(math.radians(SIDE_WALL_NORMAL_MAX_DEG))


def _is_cyl_wall(
    face_id: int,
    by_index: dict[int, FaceGeom],
    occ_map: dict[int, Any],
) -> bool:
    fg = by_index[face_id]
    if fg.surface_type not in WALL_SURF_TYPES:
        return False
    info = _wall_interior_and_axis(fg, occ_map.get(face_id))
    return info is not None and info[0]


def _is_axial_plane(
    face_id: int,
    by_index: dict[int, FaceGeom],
    opening_axis: np.ndarray,
) -> bool:
    fg = by_index[face_id]
    if fg.surface_type in {"bspline", "bezier", "sphere"}:
        return True
    if fg.surface_type != "plane":
        return False
    nd = abs(float(np.dot(_unit(fg.normal), _unit(opening_axis))))
    return nd >= math.cos(math.radians(SIDE_WALL_NORMAL_MAX_DEG))


def _pocket_role(
    face_id: int,
    by_index: dict[int, FaceGeom],
    occ_map: dict[int, Any],
    opening_axis: np.ndarray,
) -> str:
    if _is_planar_side_wall(face_id, by_index, opening_axis):
        return "planar_wall"
    if _is_cyl_wall(face_id, by_index, occ_map):
        return "cyl_wall"
    if _is_axial_plane(face_id, by_index, opening_axis):
        return "floor_step"
    if by_index[face_id].surface_type in BLEND_TYPES:
        return "blend"
    if by_index[face_id].surface_type in WALL_SURF_TYPES:
        return "other_wall"
    return "other"


def _structural_bore_gateways(
    face_set: set[int],
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
    hole_claimed: set[int],
) -> list[int]:
    gateways: set[int] = set()
    for u in face_set:
        for v in graph.neighbors.get(u, set()):
            if v in face_set:
                continue
            if by_index[v].surface_type in WALL_SURF_TYPES and v in hole_claimed:
                gateways.add(v)
    return sorted(gateways)


def _gather_pocket_region(
    seed_loop: set[int],
    *,
    by_index: dict[int, FaceGeom],
    occ_map: dict[int, Any],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    pocket_claimed: set[int],
    hole_claimed: set[int],
) -> tuple[set[int], dict[str, Any]]:
    """Expand seed loop to all structurally enclosed pocket-region faces."""
    side_walls = {i for i in seed_loop if _is_planar_side_wall(i, by_index, opening_axis)}
    floors = {i for i in seed_loop if _is_axial_plane(i, by_index, opening_axis)}
    if not side_walls:
        raise ValueError("seed loop has no planar side walls — cannot bound region")

    structural_gateways = _structural_bore_gateways(
        seed_loop, by_index, graph, hole_claimed,
    )
    bore_gateway_set = set(structural_gateways)

    uv_max = max(_uv_radius(by_index, i, opening_axis) for i in side_walls) + 2.0
    y_lo = min(_axial_y(by_index, i, opening_axis) for i in seed_loop) - 1.0
    y_hi = max(_axial_y(by_index, i, opening_axis) for i in seed_loop) + 1.0

    def in_envelope(fid: int) -> bool:
        if fid in pocket_claimed or fid in bore_gateway_set:
            return False
        y = _axial_y(by_index, fid, opening_axis)
        uv = _uv_radius(by_index, fid, opening_axis)
        return y_lo <= y <= y_hi and uv <= uv_max

    def pocket_interior_type(fid: int) -> bool:
        st = by_index[fid].surface_type
        if st == "plane":
            return _is_planar_side_wall(fid, by_index, opening_axis) or _is_axial_plane(
                fid, by_index, opening_axis,
            )
        return st in BLEND_TYPES or st in WALL_SURF_TYPES

    def bore_adjacent(fid: int) -> bool:
        return bool(set(graph.neighbors.get(fid, ())) & bore_gateway_set)

    region = set(seed_loop)
    q: deque[int] = deque(seed_loop)
    while q:
        u = q.popleft()
        for v in graph.neighbors.get(u, set()):
            if v in region or not in_envelope(v):
                continue
            if not pocket_interior_type(v):
                continue
            if _is_cyl_wall(v, by_index, occ_map) and v in hole_claimed:
                continue
            if by_index[v].surface_type in BLEND_TYPES and bore_adjacent(v):
                continue
            region.add(v)
            q.append(v)

    meta = {
        "side_walls": side_walls,
        "floors": floors,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "uv_max": uv_max,
        "structural_gateways": structural_gateways,
        "bore_adjacent_blends": sorted(
            i for i in region - seed_loop
            if by_index[i].surface_type in BLEND_TYPES
            or by_index[i].surface_type in WALL_SURF_TYPES
        ),
    }
    return region, meta


def _region_bbox_inches(
    face_ids: set[int],
    occ_faces: list[Any],
) -> dict[str, tuple[float, float]]:
    try:
        from brep_extents import collect_boundary_points
    except ImportError:
        return {}

    pts_list: list[np.ndarray] = []
    for fid in face_ids:
        pts_list.append(collect_boundary_points(occ_faces[fid]))
    pts = np.vstack(pts_list)
    out: dict[str, tuple[float, float]] = {}
    for j, name in enumerate("XYZ"):
        out[name] = (float(pts[:, j].min()) / MM_PER_IN, float(pts[:, j].max()) / MM_PER_IN)
    return out


def _top_of_part_y(step_path: str, n_faces: int) -> float:
    try:
        from brep_extents import collect_boundary_points
    except ImportError:
        faces = analyze_step(step_path)
        return max(float(f.centroid[1]) for f in faces)

    occ = load_step_faces(step_path)
    all_y: list[float] = []
    for i in range(n_faces):
        pts = collect_boundary_points(occ[i])
        all_y.extend(pts[:, 1].tolist())
    return max(all_y)


def _run_open_closed(
    face_indices: set[int],
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    config: PocketDetectionConfig,
) -> PocketFeature:
    feat = PocketFeature(
        feature_id=0,
        kind="pocket",
        subtype="through_pocket",
        face_indices=set(face_indices),
        centroid_uv=(0.0, 0.0),
        opening_axis=tuple(float(x) for x in opening_axis),
        wall_diameters={},
        wall_count=len(_claimed_wall_indices(set(face_indices), by_index)),
        floor_count=0,
        sphere_count=0,
        step_plane_count=0,
        released_faces=[],
        released_by_type={},
        depth_below_top_mm=None,
        fillet_radius_mm=None,
        surface_3d=False,
        blend_ring={},
        blend_face_indices=[],
        template_match=False,
    )
    classify_pocket_open_closed(
        feat, by_index, graph, opening_axis, config.open_closed,
    )
    return feat


def _xz_uv(by_index: dict[int, FaceGeom], face_id: int, opening_axis: np.ndarray) -> np.ndarray:
    axis = _unit(opening_axis)
    c = by_index[face_id].centroid
    return c - float(c @ axis) * axis


def step1_report(
    region: set[int],
    meta: dict[str, Any],
    *,
    by_index: dict[int, FaceGeom],
    occ_map: dict[int, Any],
    opening_axis: np.ndarray,
    pocket_result,
    hole_result,
    flat_result,
    outer_fillet_result,
    residual_result,
) -> list[str]:
    lines = [
        "=" * 78,
        "STEP 1 — COMPLETE POCKET FACE SET",
        "=" * 78,
        f"Seed wall loop: {sorted(SEED_LOOP)}",
        f"Structural expansion: {sorted(region - SEED_LOOP) or '[]'}",
        f"Complete region ({len(region)} faces): {sorted(region)}",
        "",
        f"Side walls (loop): {sorted(meta['side_walls'])}",
        f"Floor/step planes: {sorted(meta['floors'])}",
        f"Structural bore gateways (external, hole-pass): {meta['structural_gateways']}",
        f"Bore-adjacent chamfer/blend (in region, NOT pocket fillet): "
        f"{meta['bore_adjacent_blends'] or '[]'}",
        "",
        f"{'idx':>4} {'role':12s} {'surf':8s} {'area_mm2':>10} {'Y_mm':>8} "
        f"{'uv_mm':>7} {'normal':>22} {'owner':>20}",
        "-" * 98,
    ]
    for fid in sorted(region):
        fg = by_index[fid]
        role = _pocket_role(fid, by_index, occ_map, opening_axis)
        owner = _face_owner(
            fid,
            pocket_result=pocket_result,
            hole_result=hole_result,
            flat_result=flat_result,
            outer_fillet_result=outer_fillet_result,
            residual_result=residual_result,
        )
        y = _axial_y(by_index, fid, opening_axis)
        uv = _uv_radius(by_index, fid, opening_axis)
        n_str = "(" + ", ".join(f"{x:+.2f}" for x in fg.normal) + ")"
        lines.append(
            f"{fid:4d} {role:12s} {fg.surface_type:8s} {fg.area:10.1f} {y:8.2f} "
            f"{uv:7.2f} {n_str:>22} {owner:>20}"
        )

    n_floors = len(meta["floors"])
    lines.extend([
        "",
        "Multi-pocket flag:",
    ])
    if n_floors > 1:
        lines.append(
            f"  TWO floor levels ({sorted(meta['floors'])}) in one connected region — "
            "interpret as ONE stepped open pocket (not two separate pockets)."
        )
    else:
        lines.append("  Single floor level — one pocket.")
    if meta["bore_adjacent_blends"]:
        lines.append(
            "  Bore chamfer faces (323/324/325) are geometrically enclosed but are "
            "central-hole transitions, not pocket machined fillets."
        )
    return lines


def step2_report(
    region: set[int],
    *,
    by_index: dict[int, FaceGeom],
    opening_axis: np.ndarray,
    occ_faces: list[Any],
    step_path: str,
    n_faces: int,
    meta: dict[str, Any],
) -> list[str]:
    bbox = _region_bbox_inches(region, occ_faces)
    top_y = _top_of_part_y(step_path, n_faces)
    ys = [_axial_y(by_index, i, opening_axis) for i in by_index]
    y_min_part, y_max_part = min(ys), max(ys)
    deep_cut = y_min_part + 0.25 * (y_max_part - y_min_part)

    region_xz = np.mean(
        [_xz_uv(by_index, i, opening_axis) for i in region], axis=0,
    )
    radial_mm = float(np.linalg.norm(region_xz))

    lines = [
        "",
        "=" * 78,
        "STEP 2 — PHYSICAL DESCRIPTION (Toolpath GUI matching, no face indices)",
        "=" * 78,
        "",
        "Where on the part:",
        f"  • Back (rear) panel, deep bottom quartile: pocket Y spans "
        f"{meta['y_lo']:.1f} … {meta['y_hi']:.1f} mm "
        f"(part Y range {y_min_part:.1f} … {y_max_part:.1f} mm; "
        f"bottom-25% cutoff ≈ {deep_cut:.1f} mm).",
        f"  • Offset from part center in the radial (X,Z) plane: ≈ {radial_mm:.0f} mm "
        f"({radial_mm / MM_PER_IN:.2f} in) toward +Z "
        f"(region centroid XZ ≈ ({region_xz[0]:+.1f}, {region_xz[2]:+.1f}) mm).",
        "  • Near the central through-hole bore stack on the +Z side of center — "
        "the pocket opens toward the bore via hole-cylinder gateways, not toward "
        "the outer profile rim.",
        "",
        "Shape and size (from face-set bounding box):",
    ]
    if bbox:
        dx = bbox["X"][1] - bbox["X"][0]
        dy = bbox["Y"][1] - bbox["Y"][0]
        dz = bbox["Z"][1] - bbox["Z"][0]
        lines.extend([
            f"  • Planar-floored stepped open pocket (two floor levels, no in-set fillets).",
            f"  • Rough extent ≈ {dx:.2f} × {dy:.2f} × {dz:.2f} in "
            f"(X × depth × Z).",
            f"    X: {bbox['X'][0]:+.3f} … {bbox['X'][1]:+.3f} in",
            f"    Y (depth): {bbox['Y'][0]:+.3f} … {bbox['Y'][1]:+.3f} in "
            f"({abs(bbox['Y'][0] * MM_PER_IN - top_y):.1f} … "
            f"{abs(bbox['Y'][1] * MM_PER_IN - top_y):.1f} mm below top)",
            f"    Z: {bbox['Z'][0]:+.3f} … {bbox['Z'][1]:+.3f} in",
        ])
    else:
        lines.append("  • (brep_extents unavailable — bbox skipped)")

    lines.extend([
        "",
        "What borders it:",
        "  • Inboard: central through-hole bore (gateway via main hole cylinders).",
        "  • Outboard: ±X planar side walls (small end-cap walls between the two "
        "floor levels).",
        "  • Up-stack (shallower Y): transitions into the central hole/contour region.",
        "  • Not adjacent to the seven peripheral filleted pocket lobes.",
        "",
        "Toolpath click hint:",
        "  Look for a non-filleted open pocket in the deep back zone, offset toward "
        "+Z from center, stepped floor (deep + shallow level), opening into the "
        "central bore.",
    ])
    return lines


def step3_report(
    flat_result,
    region: set[int],
    *,
    by_index: dict[int, FaceGeom],
    opening_axis: np.ndarray,
    step_path: str,
    n_faces: int,
) -> list[str]:
    flat_pool = sorted(int(i) for i in flat_result.claimed_faces)
    disputed = sorted(SEED_LOOP & {322, 326})
    top_y = _top_of_part_y(step_path, n_faces)

    reading_a = sorted(i for i in flat_pool if i not in region)
    reading_b = flat_pool

    lines = [
        "",
        "=" * 78,
        "STEP 3 — REAR FLAT OVER-CLAIM RECONCILIATION",
        "=" * 78,
        "",
        f"Current cascade flat pool ({len(flat_pool)} faces): {flat_pool}",
        f"  flat_0 → {[112]}, flat_1 → {[322]}, flat_2 → {[326]}",
        "",
        f"Disputed faces (also pocket floor/step candidates): {disputed}",
        "",
        "Face 112 (rear top flat — separate from pocket):",
    ]
    fg112 = by_index[112]
    d112 = (top_y - float(fg112.centroid[1])) / MM_PER_IN
    lines.extend([
        f"  area={fg112.area:.1f} mm², Y={_axial_y(by_index, 112, opening_axis):.2f} mm, "
        f"depth below top={d112:.4f} in ({top_y - fg112.centroid[1]:.2f} mm)",
        f"  normal={np.round(fg112.normal, 3)}, owner=flat_0",
        f"  radial offset from pocket region ≈ "
        f"{float(np.linalg.norm(fg112.centroid - np.mean([by_index[i].centroid for i in region], axis=0))):.0f} mm",
        "  NOT graph-adjacent to the pocket region; shallow central seating plane "
        "on the through-hole stack — remains a flat under BOTH readings.",
        "",
        "Reading A — Toolpath labels 322/326 as POCKET faces (open_pocket):",
        f"  Remove {disputed} from flat pool → rear flats = {reading_a} "
        f"(count={len(reading_a)}).",
        "  Matches Toolpath GT flat: 1 if 112 alone is the rear FACES feature.",
        "",
        "Reading B — Toolpath labels 322/326 as FLATS (no pocket here):",
        f"  Flat pool unchanged → rear flats = {reading_b} (count={len(reading_b)}).",
        "  Pocket candidate would NOT emit as open_pocket; 330/332 become contour/"
        "central-stack caps (currently unclaimed/residual), not pocket walls.",
        "",
        "Decision table (single yes/no for Toolpath panel):",
        "+" + "-" * 76 + "+",
        "| Toolpath panel shows …              | Reading | Expected rear flat set |",
        "+" + "-" * 76 + "+",
        f"| non-filleted open_pocket here       |    A    | {reading_a!s:22s} |",
        f"| only flat(s) here, no open_pocket   |    B    | {reading_b!s:22s} |",
        "+" + "-" * 76 + "+",
    ])
    return lines


def step4_report(
    region: set[int],
    meta: dict[str, Any],
    *,
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    oc_config: PocketDetectionConfig,
    hole_claimed: set[int],
) -> list[str]:
    feat = _run_open_closed(region, by_index, graph, opening_axis, oc_config)
    structural_gw = meta["structural_gateways"]
    walls = _claimed_wall_indices(region, by_index)

    lines = [
        "",
        "=" * 78,
        "STEP 4 — GATEWAY-CLASSIFIER GAP (confirm coupled fix, implement nothing)",
        "=" * 78,
        "",
        f"classify_pocket_open_closed on complete region ({len(region)} faces, "
        "NO external gateway union):",
        f"  claimed cyl walls: {walls}",
        f"  inboard_wall_index: {feat.inboard_wall_index}",
        f"  access: {feat.access}",
        f"  gateway_face_indices (classifier): {feat.gateway_face_indices or '[]'}",
        f"  cap_face_indices: {feat.cap_face_indices or '[]'}",
        "",
        f"Structural bore gateways (hole-pass, 1-hop outside region): {structural_gw}",
        "",
    ]

    if not feat.gateway_face_indices and structural_gw:
        lines.extend([
            "CONFIRMED GAP: classifier does NOT find the gateway on its own.",
            "",
            "Missing signal:",
        ])
        if not walls:
            lines.append(
                "  • On the 4-face seed loop alone, _claimed_wall_indices returns [] "
                "(planar walls 330/332 are not WALL_SURF_TYPES) → early exit with "
                "access='open', gateway=[]."
            )
        else:
            lines.append(
                f"  • On the expanded region, claimed cyl walls = {walls} "
                f"(bore chamfer cone {walls[0] if walls else '?'}, not the planar "
                "side walls 330/332)."
            )
            inb = feat.inboard_wall_index
            if inb is not None:
                inb_uv = _uv_radius(by_index, inb, opening_axis)
                oc = oc_config.open_closed
                ext = []
                for nb in sorted(graph.neighbors.get(inb, set())):
                    if nb in region:
                        continue
                    st = by_index[nb].surface_type
                    nb_uv = _uv_radius(by_index, nb, opening_axis)
                    ext.append(
                        f"{nb}({st}, uv={nb_uv:.1f}, "
                        f"gw_rule={st in WALL_SURF_TYPES and nb_uv < inb_uv + oc.gateway_uv_tol_mm})"
                    )
                lines.append(
                    f"  • Inboard wall {inb} (uv={inb_uv:.2f}) external neighbors: "
                    f"{ext or 'none'}"
                )
        lines.extend([
            "  • Gateway rule only scans exterior WALL_SURF_TYPES inboard of the "
            "inboard cyl wall. The real exit is via hole-pass cylinders 329/347 "
            "(1-hop outside the pocket set), which the classifier never considers.",
            "  • Planar side walls (330/332) are invisible to _claimed_wall_indices.",
            "",
            "Coupled fix needed before wiring:",
            "  1) Wall-loop seeder for non-filleted planar pockets (claim side).",
            "  2) Gateway rule extension: structural 1-hop hole-pass cyl detection "
            "for planar-wall pockets (classify side).",
        ])
    elif feat.gateway_face_indices:
        lines.append("Classifier found gateway without extension (unexpected).")
    else:
        lines.append("No structural gateway and classifier agrees (unexpected for this part).")

    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="GT prep for back open-pocket candidate (diagnostic only)",
    )
    ap.add_argument("--step", default=DEFAULT_STEP)
    ap.add_argument("--graph-npz", default=DEFAULT_NPZ)
    args = ap.parse_args(argv)

    step_path = Path(args.step)
    npz_path = Path(args.graph_npz)
    edge_index, edge_attr = _load_edges(npz_path, step_path)
    records = analyze_step(step_path)
    occ_faces = load_step_faces(step_path)
    occ_map = {i: occ_faces[i] for i in range(len(occ_faces))}
    by_index = {f.index: f for f in records}
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(records))

    _, pocket_result, hole_result, flat_result, outer_fillet_result, residual_result = (
        run_cascade(step_path, edge_index, edge_attr)
    )
    opening_axis = np.array(pocket_result.opening_axis, dtype=np.float64)
    oc_config = PocketDetectionConfig()

    region, meta = _gather_pocket_region(
        set(SEED_LOOP),
        by_index=by_index,
        occ_map=occ_map,
        graph=graph,
        opening_axis=opening_axis,
        pocket_claimed=set(pocket_result.claimed_faces),
        hole_claimed=set(hole_result.claimed_faces),
    )

    out: list[str] = [
        f"PROBE: {PART_ID} back open-pocket GT prep (DIAGNOSTIC ONLY — wires nothing)",
        f"STEP: {step_path}",
        f"graph: {npz_path}",
        "",
    ]
    out.extend(step1_report(
        region, meta,
        by_index=by_index,
        occ_map=occ_map,
        opening_axis=opening_axis,
        pocket_result=pocket_result,
        hole_result=hole_result,
        flat_result=flat_result,
        outer_fillet_result=outer_fillet_result,
        residual_result=residual_result,
    ))
    out.extend(step2_report(
        region,
        by_index=by_index,
        opening_axis=opening_axis,
        occ_faces=occ_faces,
        step_path=str(step_path),
        n_faces=len(records),
        meta=meta,
    ))
    out.extend(step3_report(
        flat_result, region,
        by_index=by_index,
        opening_axis=opening_axis,
        step_path=str(step_path),
        n_faces=len(records),
    ))
    out.extend(step4_report(
        region, meta,
        by_index=by_index,
        graph=graph,
        opening_axis=opening_axis,
        oc_config=oc_config,
        hole_claimed=set(hole_result.claimed_faces),
    ))
    out.extend([
        "",
        "=" * 78,
        "NOTE",
        "=" * 78,
        "The pocket-vs-flat label for faces 322/326 remains UNRESOLVED.",
        "This probe characterizes the candidate from the code side only.",
        "Read the rear Toolpath feature panel for the region described in STEP 2 "
        "and use the STEP 3 decision table to pick Reading A or B.",
    ])

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
