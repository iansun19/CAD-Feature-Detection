#!/usr/bin/env python
"""
probe_wall_loop_pocket_signature.py — DIAGNOSTIC ONLY (wires nothing).

Prove whether an "enclosed wall-loop" signature structurally distinguishes:
  • CLAIM 1: back-plate candidate {322,326,330,332} (missed non-filleted open pocket)
  • CLAIM 2: front contour slabs {273,277} (must FAIL — safety discriminator)
  • CROSS-CHECK: 7 known filleted pockets on 96260B_plate (should PASS topology)

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_wall_loop_pocket_signature.py
"""
from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_params import FaceGeom, analyze_step, load_step_faces, require_occ
from hole_detection import FaceGraph, _unit
from pocket_detection import (
    PocketDetectionConfig,
    PocketFeature,
    WALL_SURF_TYPES,
    classify_pocket_open_closed,
    _axial_y,
    _wall_interior_and_axis,
)
from run_cascade import _load_edges, run_cascade

require_occ()

SIDE_WALL_NORMAL_MAX_DEG = 15.0
BLEND_TYPES = frozenset({"torus", "bspline", "bezier", "sphere"})
SCULPTED_FLOOR_TYPES = frozenset({"bspline", "bezier", "sphere"})

PARTS = {
    "96260B_plate": {
        "step": "96260B_REAR_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_plate/graph.npz",
    },
    "96260B_front": {
        "step": "96260B_FRONT_XR004_PCD PLATE.stp copy",
        "graph_npz": "pipeline_out/96260B_front/graph.npz",
    },
}

BACK_WALL_CANDIDATES = frozenset({322, 326, 330, 332})
FRONT_CONTOUR_FACES = frozenset({273, 277})


@dataclass
class WallLoopSubtest:
    name: str
    passed: bool
    detail: str


@dataclass
class WallLoopSignature:
    label: str
    candidate_faces: set[int]
    side_walls: set[int] = field(default_factory=set)
    cyl_walls: set[int] = field(default_factory=set)
    planar_side_walls: set[int] = field(default_factory=set)
    floor_faces: list[int] = field(default_factory=list)
    subtests: list[WallLoopSubtest] = field(default_factory=list)
    adjacency_cycle: list[int] = field(default_factory=list)
    gateway_faces: list[int] = field(default_factory=list)
    structural_gateway: list[int] = field(default_factory=list)
    cap_faces: list[int] = field(default_factory=list)
    access: str = "?"
    in_pocket_blends: list[int] = field(default_factory=list)
    external_blends: list[int] = field(default_factory=list)
    connecting_faces: set[int] = field(default_factory=set)

    @property
    def loop_closes(self) -> bool:
        return any(s.name == "wall_loop_closes" and s.passed for s in self.subtests)

    @property
    def floor_bounded(self) -> bool:
        return any(s.name == "floor_bounded" and s.passed for s in self.subtests)

    @property
    def gateway_real(self) -> bool:
        return any(s.name == "gateway_real" and s.passed for s in self.subtests)

    @property
    def no_fillet(self) -> bool:
        return any(s.name == "fillet_absence" and s.passed for s in self.subtests)

    @property
    def topology_pass(self) -> bool:
        return self.loop_closes and self.floor_bounded and self.gateway_real

    @property
    def full_pass(self) -> bool:
        return self.topology_pass and self.no_fillet

    def failed_subtests(self) -> list[str]:
        return [s.name for s in self.subtests if not s.passed]


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


def _is_floor_face(
    face_id: int,
    by_index: dict[int, FaceGeom],
    opening_axis: np.ndarray,
) -> bool:
    fg = by_index[face_id]
    if fg.surface_type in SCULPTED_FLOOR_TYPES:
        return True
    if fg.surface_type != "plane":
        return False
    nd = abs(float(np.dot(_unit(fg.normal), _unit(opening_axis))))
    return nd >= math.cos(math.radians(SIDE_WALL_NORMAL_MAX_DEG))


def _normal_angle_to_axis(normal: np.ndarray, axis: np.ndarray) -> float:
    n = _unit(np.asarray(normal, dtype=np.float64))
    a = _unit(axis)
    c = abs(float(np.dot(n, a)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _connected_component(seeds: set[int], allowed: set[int], graph: FaceGraph) -> set[int]:
    if not seeds:
        return set()
    comp: set[int] = set()
    q: deque[int] = deque([min(seeds)])
    while q:
        u = q.popleft()
        if u in comp:
            continue
        comp.add(u)
        for v in graph.neighbors.get(u, set()):
            if v in allowed and v not in comp:
                q.append(v)
    return comp


def _internal_adjacency(face_set: set[int], graph: FaceGraph) -> dict[int, set[int]]:
    return {
        i: {v for v in graph.neighbors.get(i, set()) if v in face_set}
        for i in face_set
    }


def _hamiltonian_cycle(
    face_set: set[int],
    adj: dict[int, set[int]],
    *,
    max_nodes: int = 8,
) -> list[int] | None:
    nodes = sorted(face_set)
    if len(nodes) < 3 or len(nodes) > max_nodes:
        return None
    for perm in permutations(nodes):
        if all(perm[(i + 1) % len(perm)] in adj[perm[i]] for i in range(len(perm))):
            return list(perm)
    return None


def _classify_roles(
    face_set: set[int],
    by_index: dict[int, FaceGeom],
    occ_map: dict[int, Any],
    opening_axis: np.ndarray,
) -> tuple[set[int], set[int], set[int], list[int]]:
    cyl: set[int] = set()
    planar_sw: set[int] = set()
    floors: list[int] = []
    for fid in face_set:
        if _is_cyl_wall(fid, by_index, occ_map):
            cyl.add(fid)
        elif _is_planar_side_wall(fid, by_index, opening_axis):
            planar_sw.add(fid)
        if _is_floor_face(fid, by_index, opening_axis):
            floors.append(fid)
    side_walls = cyl | planar_sw
    return side_walls, cyl, planar_sw, floors


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


def _in_set_wall_adjacent_blends(
    wall_ids: set[int],
    face_set: set[int],
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
) -> tuple[list[int], list[int]]:
    """In-set pocket fillet ring vs external 1-hop blends (structural contrast)."""
    in_set: set[int] = set()
    external: set[int] = set()
    for wid in wall_ids:
        for nb in graph.neighbors.get(wid, set()):
            if by_index[nb].surface_type not in BLEND_TYPES:
                continue
            if nb in face_set:
                in_set.add(nb)
            else:
                external.add(nb)
    return sorted(in_set), sorted(external)


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
        wall_count=len([i for i in face_indices if by_index[i].surface_type in WALL_SURF_TYPES]),
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


def _floor_bounded_test(
    floors_in_set: list[int],
    side_walls: set[int],
    cyl_walls: set[int],
    face_set: set[int],
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
    opening_axis: np.ndarray,
) -> tuple[bool, str, list[int]]:
    """Floor bounded by wall loop: side-wall contact or lobed pocket shell."""
    if not floors_in_set:
        return False, "no floor faces in candidate set", []

    if len(cyl_walls) >= 3:
        detail = (
            f"{len(floors_in_set)} floor/sculpt face(s) inside lobed pocket shell "
            f"({len(cyl_walls)} cyl walls)"
        )
        return True, detail, floors_in_set

    bounded_ids: list[int] = []
    for fid in sorted(floors_in_set):
        sw_touch = sorted(
            n for n in graph.neighbors.get(fid, set()) if n in side_walls
        )
        if len(sw_touch) >= 2:
            fg = by_index[fid]
            ang = (
                _normal_angle_to_axis(fg.normal, opening_axis)
                if fg.surface_type == "plane"
                else float("nan")
            )
            note = (
                f"face {fid} ({fg.surface_type}, normal∥axis={ang:.1f}°) "
                f"bounded by side walls {sw_touch}"
            )
            bounded_ids.append(fid)
            return True, note, bounded_ids

    return (
        False,
        f"floors {floors_in_set} lack ≥2 side-wall neighbors each "
        f"(side_walls={sorted(side_walls)})",
        [],
    )


def evaluate_wall_loop_signature(
    label: str,
    candidate_faces: set[int],
    *,
    by_index: dict[int, FaceGeom],
    occ_map: dict[int, Any],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    oc_config: PocketDetectionConfig,
    hole_claimed: set[int],
    check_fillet: bool = True,
    expect_open_gateway: bool | None = None,
) -> WallLoopSignature:
    sig = WallLoopSignature(label=label, candidate_faces=set(candidate_faces))
    face_set = sig.candidate_faces

    side_walls, cyl_walls, planar_sw, floors_in_set = _classify_roles(
        face_set, by_index, occ_map, opening_axis,
    )
    sig.side_walls = side_walls
    sig.cyl_walls = cyl_walls
    sig.planar_side_walls = planar_sw

    adj = _internal_adjacency(face_set, graph)
    component = _connected_component(face_set, face_set, graph)
    cycle = _hamiltonian_cycle(face_set, adj)
    sig.adjacency_cycle = cycle or []

    sig.connecting_faces = {
        v for u in face_set for v in graph.neighbors.get(u, set()) if v not in face_set
    }
    structural_gw = _structural_bore_gateways(face_set, by_index, graph, hole_claimed)
    sig.structural_gateway = structural_gw

    # --- (a) wall loop closes ---
    # Planar pocket: connected envelope + explicit edge-adjacency cycle + ≥2 planar side walls.
    # Filleted pocket: ≥3 interior cyl walls + ≥1 floor (lobed wall ring; no single graph cycle).
    planar_loop = (
        len(planar_sw) >= 2
        and component == face_set
        and cycle is not None
    )
    cyl_loop = len(cyl_walls) >= 3 and len(floors_in_set) >= 1
    loop_pass = planar_loop or cyl_loop

    if cycle:
        cycle_str = " -> ".join(str(i) for i in cycle) + f" -> {cycle[0]}"
        loop_detail = (
            f"edge-adjacency cycle ({len(cycle)} faces): {cycle_str}; "
            f"planar_side_walls={sorted(planar_sw)} cyl_walls={sorted(cyl_walls)}"
        )
    elif cyl_loop:
        loop_detail = (
            f"lobed cyl wall ring: {len(cyl_walls)} interior walls, "
            f"{len(floors_in_set)} floor(s) in set; no single planar cycle "
            f"(expected for filleted pockets)"
        )
    else:
        loop_detail = (
            f"no enclosing loop: connected={component == face_set} "
            f"planar_side_walls={sorted(planar_sw)} cyl_walls={sorted(cyl_walls)} "
            f"axial-only={not side_walls and bool(floors_in_set)}"
        )
    sig.subtests.append(WallLoopSubtest("wall_loop_closes", loop_pass, loop_detail))

    # --- (b) floor bounded by loop ---
    floor_pass, floor_detail, bounded_ids = _floor_bounded_test(
        floors_in_set, side_walls, cyl_walls, face_set, by_index, graph, opening_axis,
    )
    sig.floor_faces = bounded_ids or floors_in_set
    sig.subtests.append(WallLoopSubtest("floor_bounded", floor_pass, floor_detail))

    # --- (c) gateway / open-closed cap test ---
    classify_set = face_set | set(structural_gw)
    feat = _run_open_closed(
        classify_set, by_index, graph, opening_axis, oc_config,
    )
    sig.access = feat.access
    sig.cap_faces = list(feat.cap_face_indices)
    sig.gateway_faces = list(feat.gateway_face_indices) or structural_gw

    if expect_open_gateway is True:
        gateway_pass = (
            feat.access == "open"
            and not feat.cap_face_indices
            and bool(structural_gw)
        )
        gw_detail = (
            f"open/closed classify: access={feat.access} caps={feat.cap_face_indices or '[]'}; "
            f"structural bore gateway (hole-pass cyl)={structural_gw}; "
            f"classify gateway={feat.gateway_face_indices or '[]'}"
        )
    elif expect_open_gateway is False:
        gateway_pass = bool(feat.gateway_face_indices) or bool(structural_gw)
        gw_detail = (
            f"closed pocket: access={feat.access} caps={feat.cap_face_indices}; "
            f"classify gateway={feat.gateway_face_indices}; structural={structural_gw}"
        )
    else:
        gateway_pass = bool(structural_gw) or bool(feat.gateway_face_indices)
        gw_detail = (
            f"access={feat.access} caps={feat.cap_face_indices or '[]'} "
            f"classify_gateway={feat.gateway_face_indices or '[]'} "
            f"structural_gateway={structural_gw}"
        )
    sig.subtests.append(WallLoopSubtest("gateway_real", gateway_pass, gw_detail))

    # --- (d) fillet absence: in-set wall-adjacent blends (filleted pockets have 2 each) ---
    in_blends, ext_blends = _in_set_wall_adjacent_blends(
        side_walls or face_set, face_set, by_index, graph,
    )
    sig.in_pocket_blends = in_blends
    sig.external_blends = ext_blends
    if check_fillet:
        fillet_pass = len(in_blends) == 0
        fillet_detail = (
            f"in-set wall-adjacent blends={in_blends or '[]'} (expect 0; "
            f"filleted pockets have 2); external 1-hop={ext_blends}"
        )
    else:
        fillet_pass = True
        fillet_detail = f"fillet clause skipped; in-set blends={in_blends}"
    sig.subtests.append(WallLoopSubtest("fillet_absence", fillet_pass, fillet_detail))

    return sig


def _render_signature(sig: WallLoopSignature) -> list[str]:
    lines = [
        f"=== {sig.label} ===",
        f"  candidate faces: {sorted(sig.candidate_faces)}",
        f"  side walls: {sorted(sig.side_walls)}  (planar={sorted(sig.planar_side_walls)} cyl={sorted(sig.cyl_walls)})",
        f"  floors (bounded): {sig.floor_faces}",
        f"  connecting 1-hop: {sorted(sig.connecting_faces)}",
        "",
    ]
    for st in sig.subtests:
        mark = "PASS" if st.passed else "FAIL"
        lines.append(f"  [{mark}] {st.name}: {st.detail}")
    lines.append("")
    lines.append(
        f"  topology: {'PASS' if sig.topology_pass else 'FAIL'}"
        f"  |  full (+fillet): {'PASS' if sig.full_pass else 'FAIL'}"
    )
    if sig.failed_subtests():
        lines.append(f"  failed: {sig.failed_subtests()}")
    return lines


def load_part(part_id: str, step_override: str | None = None, npz_override: str | None = None):
    cfg = PARTS[part_id]
    step_path = Path(step_override or cfg["step"])
    npz_path = Path(npz_override or cfg["graph_npz"])
    edge_index, edge_attr = _load_edges(npz_path, step_path)
    records = analyze_step(step_path)
    occ_faces = load_step_faces(step_path)
    occ_map = {i: occ_faces[i] for i in range(len(occ_faces))}
    by_index = {f.index: f for f in records}
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(records))
    _, pocket_result, hole_result, _, _, _ = run_cascade(step_path, edge_index, edge_attr)
    opening_axis = np.array(pocket_result.opening_axis, dtype=np.float64)
    oc_config = PocketDetectionConfig()
    return {
        "part_id": part_id,
        "step_path": step_path,
        "npz_path": npz_path,
        "by_index": by_index,
        "occ_map": occ_map,
        "graph": graph,
        "pocket_result": pocket_result,
        "hole_claimed": set(hole_result.claimed_faces),
        "opening_axis": opening_axis,
        "oc_config": oc_config,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prove enclosed wall-loop pocket signature (diagnostic only)",
    )
    ap.add_argument("--back-step", default=PARTS["96260B_plate"]["step"])
    ap.add_argument("--back-npz", default=PARTS["96260B_plate"]["graph_npz"])
    ap.add_argument("--front-step", default=PARTS["96260B_front"]["step"])
    ap.add_argument("--front-npz", default=PARTS["96260B_front"]["graph_npz"])
    args = ap.parse_args(argv)

    out: list[str] = [
        "PROBE: enclosed wall-loop pocket signature (DIAGNOSTIC ONLY)",
        "",
    ]

    back = load_part("96260B_plate", args.back_step, args.back_npz)
    out.append("=" * 78)
    out.append("CLAIM 1 — back candidate {322,326,330,332} (96260B_plate)")
    out.append("=" * 78)

    claim1 = evaluate_wall_loop_signature(
        "back open-pocket candidate",
        BACK_WALL_CANDIDATES,
        by_index=back["by_index"],
        occ_map=back["occ_map"],
        graph=back["graph"],
        opening_axis=back["opening_axis"],
        oc_config=back["oc_config"],
        hole_claimed=back["hole_claimed"],
        check_fillet=True,
        expect_open_gateway=True,
    )
    out.extend(_render_signature(claim1))

    out.append("  edge-adjacency among candidate set:")
    adj1 = _internal_adjacency(BACK_WALL_CANDIDATES, back["graph"])
    for fid in sorted(BACK_WALL_CANDIDATES):
        fg = back["by_index"][fid]
        y = _axial_y(back["by_index"], fid, back["opening_axis"])
        out.append(
            f"    {fid} ({fg.surface_type}, Y={y:.1f}): -> {sorted(adj1[fid])}"
        )
    out.append("")
    out.append(
        f"CLAIM 1 VERDICT: {'PASS' if claim1.full_pass else 'FAIL'} — "
        + (
            "topological pocket (enclosed loop + bounded floor + open gateway + no in-set fillet)"
            if claim1.full_pass
            else f"failed subtests: {claim1.failed_subtests()}"
        )
    )

    front = load_part("96260B_front", args.front_step, args.front_npz)
    out.append("")
    out.append("=" * 78)
    out.append("CLAIM 2 — front contour slabs {273,277} (96260B_front)")
    out.append("=" * 78)

    claim2 = evaluate_wall_loop_signature(
        "front contour pair {273,277}",
        FRONT_CONTOUR_FACES,
        by_index=front["by_index"],
        occ_map=front["occ_map"],
        graph=front["graph"],
        opening_axis=front["opening_axis"],
        oc_config=front["oc_config"],
        hole_claimed=front["hole_claimed"],
        check_fillet=True,
    )
    out.extend(_render_signature(claim2))

    per_face: list[WallLoopSignature] = []
    out.append("  per-face isolation:")
    for fid in sorted(FRONT_CONTOUR_FACES):
        sig = evaluate_wall_loop_signature(
            f"front face {fid} alone",
            {fid},
            by_index=front["by_index"],
            occ_map=front["occ_map"],
            graph=front["graph"],
            opening_axis=front["opening_axis"],
            oc_config=front["oc_config"],
            hole_claimed=front["hole_claimed"],
            check_fillet=True,
        )
        per_face.append(sig)
        out.append(
            f"    face {fid}: topology={'PASS' if sig.topology_pass else 'FAIL'} "
            f"failed={sig.failed_subtests()}"
        )

    any_front_pass = claim2.topology_pass or any(s.topology_pass for s in per_face)
    if any_front_pass:
        out.append("")
        out.append(
            "STOP — front 273/277 PASS the pocket signature. "
            "Wall-loop seeder would over-claim contour slabs; UNSAFE as specified."
        )
        claim2_ok = False
    else:
        out.append("")
        out.append(
            f"CLAIM 2 VERDICT: PASS (safe discriminator) — "
            f"273/277 FAIL; failed subtests: {claim2.failed_subtests()}"
        )
        claim2_ok = True

    out.append("")
    out.append("=" * 78)
    out.append("CROSS-CHECK — 7 filleted pockets on 96260B_plate (topology only)")
    out.append("=" * 78)

    pocket_result = back["pocket_result"]
    cross_pass = 0
    for feat in sorted(pocket_result.features, key=lambda f: f.feature_id):
        sig = evaluate_wall_loop_signature(
            f"filleted pocket id={feat.feature_id}",
            set(feat.face_indices),
            by_index=back["by_index"],
            occ_map=back["occ_map"],
            graph=back["graph"],
            opening_axis=back["opening_axis"],
            oc_config=back["oc_config"],
            hole_claimed=back["hole_claimed"],
            check_fillet=False,
            expect_open_gateway=False,
        )
        if sig.topology_pass:
            cross_pass += 1
        out.append(
            f"  pocket {feat.feature_id}: topology={'PASS' if sig.topology_pass else 'FAIL'} "
            f"n_faces={len(feat.face_indices)} failed={sig.failed_subtests() or 'none'}"
        )

    n_total = len(pocket_result.features)
    out.append("")
    out.append(f"CROSS-CHECK pass rate: {cross_pass}/{n_total} filleted pockets pass topology")

    out.append("")
    out.append("=" * 78)
    out.append("FINAL — is 'enclosed wall-loop' a safe seeding signal?")
    out.append("=" * 78)

    claim1_ok = claim1.full_pass
    cross_ok = cross_pass == n_total and n_total == 7

    if claim1_ok and claim2_ok and cross_ok:
        final = (
            "YES — enclosed wall-loop captures the missed back pocket, excludes "
            "front contour slabs, and matches all 7 filleted pockets."
        )
    else:
        missing: list[str] = []
        if not claim1_ok:
            missing.append(f"CLAIM1 ({claim1.failed_subtests()})")
        if not claim2_ok:
            missing.append("CLAIM2 front 273/277 pass (over-claim)")
        if not cross_ok:
            missing.append(f"CROSS-CHECK {cross_pass}/{n_total}")
        final = "NO — seeder not yet safe. Missing: " + "; ".join(missing)

    out.append(final)
    out.append("")
    out.append(
        "NOTE: structural finding only — does not resolve Toolpath labels for "
        "322/326 as pocket vs flat (requires rear Toolpath GT)."
    )

    print("\n".join(out))
    return 0 if (claim1_ok and claim2_ok and cross_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
