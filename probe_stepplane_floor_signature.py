#!/usr/bin/env python
"""
PROBE — structural floor signature for step-planes leaking into the flat pass.

Diagnostic only — wires nothing into pocket_detection or flats_detection.

For each step-plane candidate (flat-bucket leak on front; pocket-claimed
reference on rear when already absorbed), map via local fillet chain to a
pocket instance and test the FLOOR signature (i)-(iv). Reports safety hazards
if fillet chains reach non-floor targets (neighbor pocket walls, contour,
outer fillets, holes).

Run both reference parts:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_stepplane_floor_signature.py
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_stepplane_floor_signature.py --part 96260B_front
"""
from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from feature_params import analyze_step
from hole_detection import FaceGraph, _unit
from pocket_detection import (
    PocketDetectionConfig,
    WALL_SURF_TYPES,
    _pocket_footprint,
    _project_uv,
)
from run_cascade import _load_edges, run_cascade

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

FILLET_TYPES = frozenset({"torus"})
BLEND_TRAVERSABLE = frozenset({"torus", "sphere", "bspline", "bezier"})
CONTOUR_TYPES = frozenset({"plane", "cylinder", "cone"})


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


def normal_parallel_to_axis(normal, axis: np.ndarray, tol_deg: float) -> tuple[bool, float]:
    n = _unit(np.asarray(normal, dtype=np.float64))
    a = _unit(axis)
    cos_v = abs(float(np.dot(n, a)))
    cos_tol = math.cos(math.radians(tol_deg))
    return cos_v >= cos_tol, math.degrees(math.acos(max(-1.0, min(1.0, cos_v))))


def axial_proj(centroid, axis: np.ndarray) -> float:
    return float(np.dot(np.asarray(centroid, dtype=np.float64), _unit(axis)))


def pocket_instance_map(pocket_result) -> dict[int, int]:
    out: dict[int, int] = {}
    for feat in pocket_result.features:
        for i in feat.face_indices:
            out[int(i)] = int(feat.feature_id)
    return out


def feat_by_id(pocket_result) -> dict[int, Any]:
    return {int(f.feature_id): f for f in pocket_result.features}


def nearest_pocket_by_uv(
    face_idx: int,
    by_index: dict[int, Any],
    pocket_result,
    opening_axis: np.ndarray,
) -> tuple[int, float, float]:
    uv = _project_uv(np.asarray(by_index[face_idx].centroid, dtype=np.float64), opening_axis)
    dists: list[tuple[float, int]] = []
    for feat in pocket_result.features:
        c = np.array(feat.centroid_uv, dtype=np.float64)
        dists.append((float(np.linalg.norm(uv - c)), int(feat.feature_id)))
    dists.sort()
    return dists[0][1], dists[0][0], dists[1][0] - dists[0][0] if len(dists) > 1 else float("inf")


@dataclass
class FilletChainHit:
    terminal_face: int
    terminal_kind: str
    pocket_id: int | None
    path: list[int]
    edge_kinds: list[str]


@dataclass
class StepPlaneDiagnosis:
    face_id: int
    cohort: str
    mapped_pocket: int | None
    map_method: str
    fillet_pockets: list[int]
    spatial_nearest: int
    spatial_margin_mm: float
    tests: dict[str, tuple[bool, str]]
    signals: dict[str, Any] = field(default_factory=dict)
    safety_hits: list[FilletChainHit] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return (
            self.mapped_pocket is not None
            and len(self.fillet_pockets) <= 2
            and all(ok for ok, _ in self.tests.values())
        )


def local_fillet_wall_hits(
    step_face: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    pocket_map: dict[int, int],
    *,
    max_hops: int = 3,
) -> list[tuple[int, list[int], list[str]]]:
    """step -> fillet[*] -> wall paths (bounded hop count, no pocket interior walk)."""
    hits: list[tuple[int, list[int], list[str]]] = []
    q: deque[tuple[int, list[int], list[str], int]] = deque([(step_face, [step_face], [], 0)])
    seen_paths: set[tuple[int, ...]] = set()

    while q:
        u, path, edges, depth = q.popleft()
        if depth >= max_hops:
            continue
        for v in graph.neighbors.get(u, ()):
            ek = graph.edge_kind(u, v) or "?"
            st = by_index[v].surface_type
            if v in pocket_map and st in WALL_SURF_TYPES:
                hits.append((v, path + [v], edges + [ek]))
                continue
            if st in FILLET_TYPES or (depth > 0 and st in BLEND_TRAVERSABLE):
                key = tuple(path + [v])
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                q.append((v, path + [v], edges + [ek], depth + 1))
    return hits


def _classify_terminal(
    face_id: int,
    st: str,
    *,
    pocket_map: dict[int, int],
    pocket_claimed: set[int],
    hole_claimed: set[int],
    outer_claimed: set[int],
    assigned_pocket: int | None,
) -> tuple[str, int | None]:
    if face_id in pocket_claimed:
        pid = pocket_map.get(face_id)
        if st in WALL_SURF_TYPES:
            kind = "neighbor_pocket_wall" if pid != assigned_pocket else "assigned_pocket_wall"
        elif st == "plane":
            kind = "neighbor_pocket_plane" if pid != assigned_pocket else "pocket_opening_plane"
        else:
            kind = f"pocket_{st}"
        return kind, pid
    if face_id in hole_claimed:
        return "hole", None
    if face_id in outer_claimed:
        return "outer_fillet", None
    if st in CONTOUR_TYPES:
        return f"contour_{st}", None
    return f"other_{st}", None


def local_fillet_safety_trace(
    start: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    *,
    pocket_map: dict[int, int],
    pocket_claimed: set[int],
    hole_claimed: set[int],
    outer_claimed: set[int],
    assigned_pocket: int | None,
    max_hops: int = 3,
) -> list[FilletChainHit]:
    """Bounded blend traversal from a step-plane (hop-limited safety audit)."""
    out: list[FilletChainHit] = []
    q: deque[tuple[int, list[int], list[str], int]] = deque([(start, [start], [], 0)])
    seen: set[int] = {start}

    while q:
        u, path, edges, depth = q.popleft()
        if depth >= max_hops:
            continue
        for v in graph.neighbors.get(u, ()):
            if v in seen:
                continue
            ek = graph.edge_kind(u, v) or "?"
            st = by_index[v].surface_type

            if v in pocket_claimed or v in hole_claimed or v in outer_claimed or st in CONTOUR_TYPES:
                kind, pid = _classify_terminal(
                    v, st,
                    pocket_map=pocket_map,
                    pocket_claimed=pocket_claimed,
                    hole_claimed=hole_claimed,
                    outer_claimed=outer_claimed,
                    assigned_pocket=assigned_pocket,
                )
                out.append(FilletChainHit(v, kind, pid, path + [v], edges + [ek]))
                continue

            if st in BLEND_TRAVERSABLE:
                seen.add(v)
                q.append((v, path + [v], edges + [ek], depth + 1))

    return out


def torus_connects_pocket_walls(
    torus_id: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    pocket_map: dict[int, int],
    pocket_walls: set[int],
) -> bool:
    """True when a fillet face touches a wall of the target pocket (≤2 hops)."""
    for w in graph.neighbors.get(torus_id, ()):
        if w in pocket_walls and by_index[w].surface_type in WALL_SURF_TYPES:
            return True
        if by_index[w].surface_type not in FILLET_TYPES:
            continue
        for w2 in graph.neighbors.get(w, ()):
            if w2 in pocket_walls and by_index[w2].surface_type in WALL_SURF_TYPES:
                return True
    return False


TEST_ROMAN = {
    "i_wall_loop": "i",
    "ii_normal_parallel": "ii",
    "iii_closed_axial_end": "iii",
    "iv_concave_fillet": "iv",
}

LOCAL_UNSAFE_KINDS = frozenset({
    "neighbor_pocket_wall", "neighbor_pocket_plane", "hole",
    "outer_fillet", "contour_plane", "contour_cylinder", "contour_cone",
})


def identify_step_plane_candidates(
    flat_faces: list[int],
    pocket_faces: list[int],
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    normal_tol_deg: float,
    depth_fn,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return (flat_leak cohort, pocket_reference cohort) of (face_id, label)."""
    flat_planes = [
        i for i in flat_faces
        if by_index[i].surface_type == "plane"
        and normal_parallel_to_axis(by_index[i].normal, opening_axis, normal_tol_deg)[0]
    ]
    flat_tiers, _ = depth_tiers_from_gaps(flat_planes, depth_fn)
    shallowest_flat = set(flat_tiers[0]) if flat_tiers else set()
    flat_leak = [
        (i, "flat_leak")
        for i in flat_planes
        if i not in shallowest_flat
    ]

    pocket_planes = [
        i for i in pocket_faces
        if by_index[i].surface_type == "plane"
        and normal_parallel_to_axis(by_index[i].normal, opening_axis, normal_tol_deg)[0]
    ]
    pocket_ref = [(i, "pocket_claimed") for i in sorted(pocket_planes)]
    return flat_leak, pocket_ref


def diagnose_step_plane(
    face_id: int,
    cohort: str,
    *,
    by_index: dict[int, Any],
    graph: FaceGraph,
    pocket_result,
    pocket_map: dict[int, int],
    pocket_claimed: set[int],
    hole_claimed: set[int],
    outer_claimed: set[int],
    residual_claimed: set[int],
    opening_axis: np.ndarray,
    normal_tol_deg: float,
) -> StepPlaneDiagnosis:
    fg = by_index[face_id]
    feats = feat_by_id(pocket_result)
    spatial_pid, spatial_dist, spatial_margin = nearest_pocket_by_uv(
        face_id, by_index, pocket_result, opening_axis,
    )

    # Local fillet chain -> walls
    wall_paths = local_fillet_wall_hits(face_id, graph, by_index, pocket_map)
    fillet_pockets = sorted({pocket_map[w] for w, _, _ in wall_paths})
    walls_by_pocket: dict[int, set[int]] = {}
    for w, path, edges in wall_paths:
        pid = pocket_map[w]
        walls_by_pocket.setdefault(pid, set()).add(w)

    # Map to pocket: spatial nearest when fillet reaches it, else spatial alone
    if spatial_pid in fillet_pockets:
        mapped = spatial_pid
        map_method = "fillet_chain+spatial"
    elif len(fillet_pockets) == 1:
        mapped = fillet_pockets[0]
        map_method = "fillet_chain_only"
    elif fillet_pockets:
        mapped = spatial_pid
        map_method = "spatial_tiebreak"
    else:
        mapped = spatial_pid
        map_method = "spatial_only"

    feat = feats[mapped]
    axis = _unit(np.asarray(feat.opening_axis, dtype=np.float64))
    pocket_walls = {
        i for i in feat.face_indices if by_index[i].surface_type in WALL_SURF_TYPES
    }
    reached_walls = walls_by_pocket.get(mapped, set())

    # (i) wall loop enclosure
    multi_pocket = len(fillet_pockets) > 2 or (
        len(fillet_pockets) == 2 and mapped not in fillet_pockets
    )
    walls_subset = reached_walls <= pocket_walls if reached_walls else False
    has_fillet_walls = len(reached_walls) > 0
    wall_frac = len(reached_walls) / max(len(pocket_walls), 1)
    intermediate_faces = {
        p for _, path, _ in wall_paths for p in path[1:-1]
    }
    no_contour_in_local = all(
        by_index[p].surface_type not in CONTOUR_TYPES or p in pocket_claimed
        for p in intermediate_faces
    )
    test_i = (
        not multi_pocket
        and has_fillet_walls
        and walls_subset
        and no_contour_in_local
    )
    detail_i = (
        f"fillet_pockets={fillet_pockets} reached_walls={sorted(reached_walls)} "
        f"({len(reached_walls)}/{len(pocket_walls)}={wall_frac:.0%} of P{mapped} walls) "
        f"subset={walls_subset} multi={multi_pocket}"
    )

    # (ii) normal parallel to pocket opening axis
    par, ang = normal_parallel_to_axis(fg.normal, axis, normal_tol_deg)
    test_ii = par
    detail_ii = f"|n·axis| angle={ang:.1f}° (tol={normal_tol_deg}°) normal={tuple(round(x,3) for x in fg.normal)}"

    # (iii) closed axial end — deepest plane tier in pocket + this face
    pocket_planes = [
        i for i in feat.face_indices if by_index[i].surface_type == "plane"
    ] + [face_id]
    axials = {i: axial_proj(by_index[i].centroid, axis) for i in pocket_planes}
    # Closed end = extreme axial among planes (min Y for axis +Y opening toward top)
    closed_end_axial = min(axials.values())
    step_axial = axial_proj(fg.centroid, axis)
    plane_axials = sorted(set(axials.values()))
    axial_tiers, _ = depth_tiers_from_gaps(
        list(axials.keys()),
        lambda i: -axials[i],  # rank by depth (more negative = deeper)
    )
    deepest_tier = set(axial_tiers[-1]) if axial_tiers else set()
    at_closed_end = face_id in deepest_tier
    mouth_axial = max(axials[i] for i in feat.face_indices if i in axials)
    test_iii = at_closed_end and step_axial <= mouth_axial + 1e-6
    detail_iii = (
        f"step_axial={step_axial:.2f} closed_end={closed_end_axial:.2f} "
        f"mouth_axial={mouth_axial:.2f} deepest_tier={sorted(deepest_tier)} "
        f"access={feat.access}"
    )

    # (iv) concave step->fillet on floor-to-wall fillet ring (concave step edge + pocket wall)
    step_fillet_edges: list[tuple[int, str]] = []
    pocket_fillet_edges: list[tuple[int, str]] = []
    for nb in graph.neighbors.get(face_id, ()):
        if by_index[nb].surface_type not in FILLET_TYPES:
            continue
        ek = graph.edge_kind(face_id, nb) or "?"
        step_fillet_edges.append((nb, ek))
        if ek != "concave":
            continue
        if torus_connects_pocket_walls(nb, graph, by_index, pocket_map, pocket_walls):
            pocket_fillet_edges.append((nb, ek))
    concave_ok = bool(pocket_fillet_edges) and all(
        ek == "concave" for _, ek in pocket_fillet_edges
    )
    test_iv = concave_ok
    detail_iv = (
        f"pocket_fillet step->torus={pocket_fillet_edges} all_concave={concave_ok} "
        f"(all step->torus={step_fillet_edges})"
    )

    # Safety trace (hop-limited — models one fillet hop of pocket grow)
    safety = local_fillet_safety_trace(
        face_id, graph, by_index,
        pocket_map=pocket_map,
        pocket_claimed=pocket_claimed,
        hole_claimed=hole_claimed,
        outer_claimed=outer_claimed,
        assigned_pocket=mapped,
        max_hops=3,
    )
    safety_bad = [h for h in safety if h.terminal_kind in LOCAL_UNSAFE_KINDS]

    tests = {
        "i_wall_loop": (test_i, detail_i),
        "ii_normal_parallel": (test_ii, detail_ii),
        "iii_closed_axial_end": (test_iii, detail_iii),
        "iv_concave_fillet": (test_iv, detail_iv),
    }

    return StepPlaneDiagnosis(
        face_id=face_id,
        cohort=cohort,
        mapped_pocket=mapped,
        map_method=map_method,
        fillet_pockets=fillet_pockets,
        spatial_nearest=spatial_pid,
        spatial_margin_mm=spatial_margin,
        tests=tests,
        signals={
            "area_mm2": round(float(fg.area), 1),
            "spatial_dist_mm": round(spatial_dist, 2),
            "wall_frac": round(wall_frac, 3),
            "step_fillet_neighbors": [t for t, _ in step_fillet_edges],
            "pocket_fillet_neighbors": [t for t, _ in pocket_fillet_edges],
        },
        safety_hits=safety_bad,
    )


def run_part(part_id: str, step_path: str, graph_npz: str, normal_tol_deg: float) -> dict[str, Any]:
    faces_list = analyze_step(step_path)
    by_index = {f.index: f for f in faces_list}
    n_faces = len(faces_list)

    edge_index, edge_attr = _load_edges(Path(graph_npz), Path(step_path))
    _, pocket_result, hole_result, flat_result, outer_result, residual_result = run_cascade(
        step_path, edge_index, edge_attr,
    )
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)
    pocket_map = pocket_instance_map(pocket_result)
    pocket_claimed = set(int(i) for i in pocket_result.claimed_faces)
    hole_claimed = set(int(i) for i in hole_result.claimed_faces)
    outer_claimed = set(int(i) for i in outer_result.claimed_faces)
    residual_claimed = set(int(i) for i in residual_result.claimed_faces)

    opening_axis = _unit(np.asarray(pocket_result.opening_axis, dtype=np.float64))
    cfg = PocketDetectionConfig()
    normal_tol_deg = normal_tol_deg or cfg.step_plane_normal_tol_deg

    def depth_fn(i: int) -> float:
        top_y = max(float(by_index[j].centroid[1]) for j in by_index)
        return top_y - float(by_index[i].centroid[1])

    flat_leak, pocket_ref = identify_step_plane_candidates(
        sorted(int(i) for i in flat_result.claimed_faces),
        sorted(pocket_claimed),
        by_index,
        opening_axis,
        normal_tol_deg,
        depth_fn,
    )

    # Primary cohort per part: front flat-leak step planes; rear pocket-claimed reference
    if part_id == "96260B_front":
        candidates = list(flat_leak)
    else:
        # Rear: pocket step planes are already claimed; flat-bucket faces are unrelated.
        candidates = list(pocket_ref)

    diagnoses: list[StepPlaneDiagnosis] = []
    for face_id, cohort in candidates:
        diagnoses.append(diagnose_step_plane(
            face_id, cohort,
            by_index=by_index,
            graph=graph,
            pocket_result=pocket_result,
            pocket_map=pocket_map,
            pocket_claimed=pocket_claimed,
            hole_claimed=hole_claimed,
            outer_claimed=outer_claimed,
            residual_claimed=residual_claimed,
            opening_axis=opening_axis,
            normal_tol_deg=normal_tol_deg,
        ))

    overall = (
        bool(diagnoses)
        and all(d.all_pass for d in diagnoses)
        and not any(d.safety_hits for d in diagnoses)
    )
    return {
        "part_id": part_id,
        "flat_claimed": sorted(flat_result.claimed_faces),
        "flat_leak_candidates": flat_leak,
        "pocket_step_candidates": pocket_ref,
        "diagnoses": diagnoses,
        "overall_pass": overall,
    }


def summarize_safety_hits(hits: list[FilletChainHit]) -> list[str]:
    """Compact safety report: counts + one shortest example per kind."""
    if not hits:
        return ["  safety: no local fillet chain (≤3 hop) to forbidden targets"]
    lines = ["  *** SAFETY: local fillet chain reaches non-floor targets ***"]
    by_kind: dict[str, list[FilletChainHit]] = {}
    for h in hits:
        by_kind.setdefault(h.terminal_kind, []).append(h)
    for kind in sorted(by_kind):
        group = by_kind[kind]
        example = min(group, key=lambda h: len(h.path))
        lines.append(
            f"    {kind}: {len(group)} hit(s); example face {example.terminal_face} "
            f"P{example.pocket_id} path={example.path} edges={example.edge_kinds}"
        )
    return lines


def render_part_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    pid = result["part_id"]
    lines.append(f"{'=' * 72}")
    lines.append(f"PART {pid}")
    lines.append(f"{'=' * 72}")
    lines.append(f"flat pass claimed: {result['flat_claimed']}")
    lines.append(
        f"flat-leak step-plane candidates: "
        f"{[f for f, _ in result['flat_leak_candidates']]}"
    )
    lines.append(
        f"pocket-claimed step-plane reference: "
        f"{[f for f, _ in result['pocket_step_candidates']]} "
        f"(n={len(result['pocket_step_candidates'])})"
    )
    lines.append("")

    for d in result["diagnoses"]:
        lines.append(f"--- step-plane face {d.face_id} [{d.cohort}] ---")
        lines.append(
            f"  mapped pocket: P{d.mapped_pocket} ({d.map_method})  "
            f"fillet_pockets={d.fillet_pockets}  spatial_nearest=P{d.spatial_nearest} "
            f"margin={d.spatial_margin_mm:.1f}mm"
        )
        for key in ("i_wall_loop", "ii_normal_parallel", "iii_closed_axial_end", "iv_concave_fillet"):
            ok, detail = d.tests[key]
            roman = TEST_ROMAN[key]
            lines.append(f"  ({roman}) {'PASS' if ok else 'FAIL'}: {detail}")
        lines.append(f"  signals: {d.signals}")

        lines.extend(summarize_safety_hits(d.safety_hits))
        lines.append(f"  face verdict: {'PASS' if d.all_pass and not d.safety_hits else 'FAIL'}")
        lines.append("")

    verdict = "PASS" if result["overall_pass"] else "FAIL"
    lines.append(f"OVERALL {pid}: {verdict}")
    return "\n".join(lines)


def cross_part_report(front: dict[str, Any], rear: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"{'=' * 72}")
    lines.append("CROSS-PART CHECK")
    lines.append(f"{'=' * 72}")

    front_leak = {d.face_id: d for d in front["diagnoses"] if d.cohort == "flat_leak"}
    rear_pocket = {d.face_id: d for d in rear["diagnoses"] if d.cohort == "pocket_claimed"}

    lines.append(
        f"front flat-leak step-planes: {sorted(front_leak)} (n={len(front_leak)})"
    )
    lines.append(
        f"rear  pocket-claimed step-planes: {sorted(rear_pocket)} (n={len(rear_pocket)}) "
        f"(rear flat-bucket has no pocket-floor leak — step planes already claimed)"
    )

    front_pass = sum(1 for d in front_leak.values() if d.all_pass and not d.safety_hits)
    rear_ref_pass = sum(
        1 for d in rear_pocket.values() if d.all_pass and not d.safety_hits
    )
    # Deepest step plane per pocket on rear (closed-end floor only)
    rear_deepest: dict[int, StepPlaneDiagnosis] = {}
    for d in rear_pocket.values():
        if d.tests["iii_closed_axial_end"][0]:
            rear_deepest[d.mapped_pocket] = d
    rear_deep_pass = sum(
        1 for d in rear_deepest.values() if d.all_pass and not d.safety_hits
    )
    lines.append("")
    lines.append(
        f"signature pass rate: front flat-leak {front_pass}/{len(front_leak)}  "
        f"rear pocket-all {rear_ref_pass}/{len(rear_pocket)}  "
        f"rear closed-end floors {rear_deep_pass}/{len(rear_deepest)}"
    )

    # Compare test-level agreement on structural signature (not face indices)
    test_keys = ("i_wall_loop", "ii_normal_parallel", "iii_closed_axial_end", "iv_concave_fillet")
    for tk in test_keys:
        f_rate = sum(1 for d in front_leak.values() if d.tests[tk][0]) / max(len(front_leak), 1)
        if tk == "iii_closed_axial_end":
            r_pool = list(rear_deepest.values())
        else:
            r_pool = list(rear_pocket.values())
        r_rate = sum(1 for d in r_pool if d.tests[tk][0]) / max(len(r_pool), 1)
        flag = ""
        if abs(f_rate - r_rate) > 0.01 and min(f_rate, r_rate) >= 0.99:
            flag = "  *** DIVERGENT ***"
        elif abs(f_rate - r_rate) > 0.25:
            flag = "  *** DIVERGENT ***"
        lines.append(f"  {tk}: front={f_rate:.0%}  rear_ref={r_rate:.0%}{flag}")

    any_unsafe = any(
        d.safety_hits for r in (front, rear) for d in r["diagnoses"]
    )
    if any_unsafe:
        lines.append("")
        lines.append("*** HOP-BASED POCKET GROW IS UNSAFE ***")
        lines.append(
            "At least one step-plane fillet chain reaches a neighbor pocket wall, "
            "contour face, outer fillet, or hole. See per-face SAFETY blocks above."
        )
    else:
        lines.append("")
        lines.append(
            "Safety: no step-plane fillet chain reaches forbidden targets on either part."
        )

    identical = (
        len(front_leak) > 0
        and front_pass == len(front_leak)
        and rear_deep_pass == len(rear_deepest)
        and len(rear_deepest) == 7
        and not any_unsafe
    )
    lines.append("")
    if identical:
        lines.append(
            "CROSS-PART: floor signature HOLDS on both parts "
            "(front flat-leak + rear pocket-reference cohorts)."
        )
    else:
        lines.append(
            "CROSS-PART: signature differs or incomplete — review per-face FAILs "
            "before wiring pocket grow."
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step-plane pocket floor signature probe")
    ap.add_argument("--part", default=None, help="Single part id; default runs both")
    ap.add_argument("--step", default=None)
    ap.add_argument("--graph-npz", default=None)
    ap.add_argument(
        "--normal-tol-deg", type=float, default=None,
        help="Normal||axis tolerance (default PocketDetectionConfig 10°)",
    )
    args = ap.parse_args()

    parts_to_run = [args.part] if args.part else list(PARTS)

    results: dict[str, dict[str, Any]] = {}
    for part_id in parts_to_run:
        cfg = PARTS.get(part_id, {})
        step_path = args.step or cfg.get("step") or f"{part_id}.step"
        graph_npz = args.graph_npz or cfg.get("graph_npz")
        if not graph_npz:
            raise SystemExit(f"No graph npz for part {part_id}")
        results[part_id] = run_part(
            part_id, step_path, graph_npz, args.normal_tol_deg,
        )
        print(render_part_report(results[part_id]))

    if len(results) == 2 and "96260B_front" in results and "96260B_plate" in results:
        print(cross_part_report(results["96260B_front"], results["96260B_plate"]))


if __name__ == "__main__":
    main()
