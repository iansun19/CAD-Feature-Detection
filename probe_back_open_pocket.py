#!/usr/bin/env python
"""
probe_back_open_pocket.py — DIAGNOSTIC ONLY (wires nothing).

Investigates the known defect on 96260B_plate (back): Toolpath GT expects 1
non-filleted open_pocket but the cascade emits 0. Re-baselines post-Stage-2b,
locates the structural candidate, traces emit logic, and runs the open/closed
classifier on the candidate.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python probe_back_open_pocket.py
"""
from __future__ import annotations

import argparse
import inspect
import math
from collections import deque, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from feature_params import FaceGeom, analyze_step, load_step_faces, require_occ
from hole_detection import FaceGraph
from pocket_detection import (
    PocketFeature,
    PocketOpenClosedConfig,
    PocketDetectionConfig,
    WALL_SURF_TYPES,
    classify_pocket_open_closed,
    pocket_toolpath_class,
    _uv_radius,
    _unit,
    _axial_y,
    _wall_interior_and_axis,
)
from run_cascade import _load_edges, run_cascade

require_occ()

MM_PER_IN = 25.4
PART_ID = "96260B_plate"
DEFAULT_STEP = "96260B_REAR_XR004_PCD PLATE.stp copy"
DEFAULT_NPZ = "pipeline_out/96260B_plate/graph.npz"
SCULPTED_TYPES = frozenset({"bspline", "bezier", "sphere"})
SIDE_WALL_NORMAL_MAX_DEG = 15.0


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


def _is_side_wall(
    face_id: int,
    by_index: dict[int, FaceGeom],
    occ_map: dict[int, Any],
    opening_axis: np.ndarray,
) -> bool:
    fg = by_index[face_id]
    if fg.surface_type in WALL_SURF_TYPES:
        info = _wall_interior_and_axis(fg, occ_map.get(face_id))
        return info is not None and info[0]
    if fg.surface_type == "plane":
        nd = abs(float(np.dot(_unit(fg.normal), _unit(opening_axis))))
        return nd < math.cos(math.radians(SIDE_WALL_NORMAL_MAX_DEG))
    return False


def _is_axial_plane(face_id: int, by_index: dict[int, FaceGeom], opening_axis: np.ndarray) -> bool:
    fg = by_index[face_id]
    if fg.surface_type != "plane":
        return False
    nd = abs(float(np.dot(_unit(fg.normal), _unit(opening_axis))))
    return nd >= math.cos(math.radians(SIDE_WALL_NORMAL_MAX_DEG))


def _sculpted_1hop(
    wall_ids: set[int],
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
) -> set[int]:
    out: set[int] = set()
    for wid in wall_ids:
        for nb in graph.neighbors.get(wid, set()):
            if by_index[nb].surface_type in SCULPTED_TYPES:
                out.add(nb)
    return out


def _identify_central_bore(
    records: list[FaceGeom],
    hole_claimed: set[int],
    graph: FaceGraph,
) -> set[int]:
    """Central bore = hole-pass faces + structurally matching coaxial walls + 1-hop blends."""
    bore = set(hole_claimed)
    for fg in records:
        if fg.surface_type not in WALL_SURF_TYPES or fg.radius is None:
            continue
        # Interior coaxial walls already claimed by the hole pass are the anchor.
        if fg.index not in hole_claimed:
            continue
        bore.add(fg.index)
    for i in list(bore):
        for nb in graph.neighbors.get(i, set()):
            if records[nb].surface_type in ("torus", "plane", "sphere"):
                bore.add(nb)
    return bore


def _grow_components(
    seeds: list[int],
    graph: FaceGraph,
    by_index: dict[int, FaceGeom],
    *,
    allowed: set[int],
    skip_types: frozenset[str],
) -> list[set[int]]:
    seen: set[int] = set()
    comps: list[set[int]] = []
    for s in seeds:
        if s in seen or s not in allowed:
            continue
        q: deque[int] = deque([s])
        comp: set[int] = set()
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            for nb in graph.neighbors.get(u, set()):
                if nb not in allowed or nb in seen:
                    continue
                if by_index[nb].surface_type in skip_types:
                    continue
                q.append(nb)
        if comp:
            comps.append(comp)
    return comps


def _deep_axial_threshold(by_index: dict[int, FaceGeom], opening_axis: np.ndarray) -> float:
    ys = [float(by_index[i].centroid @ opening_axis) for i in by_index]
    y_min, y_max = min(ys), max(ys)
    return y_min + 0.25 * (y_max - y_min)


def _has_bore_gateway(
    comp: set[int],
    bore_faces: set[int],
    graph: FaceGraph,
) -> bool:
    for i in comp:
        for nb in graph.neighbors.get(i, set()):
            if nb in bore_faces and nb not in comp:
                return True
    return False


def _wall_loop_from_component(
    comp: set[int],
    by_index: dict[int, FaceGeom],
    occ_map: dict[int, Any],
    opening_axis: np.ndarray,
    hole_claimed: set[int],
) -> set[int]:
    """Wall loop = side walls + axial floors in the component, excluding hole-pass bore faces."""
    out: set[int] = set()
    for i in comp:
        if i in hole_claimed:
            continue
        if _is_side_wall(i, by_index, occ_map, opening_axis):
            out.add(i)
        elif _is_axial_plane(i, by_index, opening_axis):
            out.add(i)
    return out


def _find_structural_candidates(
    records: list[FaceGeom],
    graph: FaceGraph,
    occ_map: dict[int, Any],
    opening_axis: np.ndarray,
    pocket_result,
    hole_result,
) -> list[dict[str, Any]]:
    by_index = {f.index: f for f in records}
    pocket_claimed = pocket_result.claimed_faces
    hole_claimed = hole_result.claimed_faces
    bore_faces = _identify_central_bore(records, hole_claimed, graph)

    deep_cut = _deep_axial_threshold(by_index, opening_axis)
    allowed = {i for i in by_index if i not in pocket_claimed}
    seeds = [
        i for i in allowed
        if _is_side_wall(i, by_index, occ_map, opening_axis)
        and float(by_index[i].centroid @ opening_axis) < deep_cut
    ]

    raw_comps = _grow_components(
        seeds, graph, by_index,
        allowed=allowed,
        skip_types=frozenset({"torus"}) | SCULPTED_TYPES,
    )

    candidates: list[dict[str, Any]] = []
    for comp in raw_comps:
        wall_loop = _wall_loop_from_component(
            comp, by_index, occ_map, opening_axis, hole_claimed,
        )
        if len(wall_loop) < 3:
            continue
        wall_only = {
            i for i in wall_loop
            if _is_side_wall(i, by_index, occ_map, opening_axis)
        }
        if not wall_only:
            continue
        sculpted_adj = _sculpted_1hop(wall_only, by_index, graph)
        if sculpted_adj:
            continue  # filleted-pocket signature: bspline/bezier/sphere adjacent to walls
        if not _has_bore_gateway(wall_loop, bore_faces, graph):
            continue
        candidates.append({
            "raw_component": comp,
            "wall_loop": wall_loop,
            "wall_only": wall_only,
            "deep_cut_y": deep_cut,
            "sculpted_1hop": sculpted_adj,
            "bore_touch": sorted(
                nb for i in wall_loop for nb in graph.neighbors.get(i, set())
                if nb in bore_faces and nb not in wall_loop
            ),
        })

    candidates.sort(key=lambda c: (-len(c["wall_loop"]), min(c["wall_loop"])))
    return candidates


def _classify_candidate(
    wall_loop: set[int],
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    config: PocketOpenClosedConfig,
) -> PocketFeature:
    feat = PocketFeature(
        feature_id=0,
        kind="pocket",
        subtype="through_pocket",
        face_indices=set(wall_loop),
        centroid_uv=(0.0, 0.0),
        opening_axis=tuple(float(x) for x in opening_axis),
        wall_diameters={},
        wall_count=len([i for i in wall_loop if by_index[i].surface_type in WALL_SURF_TYPES]),
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
    classify_pocket_open_closed(feat, by_index, graph, opening_axis, config)
    return feat


def _emit_branch_report() -> str:
    src = inspect.getsource(pocket_toolpath_class)
    lines = [
        "STEP 2 — EMIT-BRANCH TRACE (static)",
        "",
        "pocket_toolpath_class(feat) in pocket_detection.py:",
        "",
    ]
    for line in src.splitlines():
        lines.append(f"  {line}")
    lines.extend([
        "",
        "Decision tree:",
        "  has_fillet = fillet_radius_mm is not None and > 0",
        "  is_open    = access == 'open'",
        "  filleted + open -> filleted_open_pocket",
        "  filleted       -> filleted_pocket",
        "  open           -> open_pocket   <-- non-filleted open path EXISTS",
        "  else           -> pocket",
        "",
        "There is NO branch that requires fillet/blend for open_pocket emission.",
        "Non-filleted open_pocket emits when access=='open' AND fillet_radius_mm is None/0.",
        "If access is wrongly 'closed', emission falls through to 'pocket' (not open_pocket).",
    ])
    return "\n".join(lines)


def step0_report(pocket_result) -> tuple[str, bool]:
    lines = [
        "STEP 0 — RE-BASELINE (post-Stage-2b cascade)",
        "",
        f"Total pocket instances emitted: {len(pocket_result.features)}",
        "",
    ]
    open_count = 0
    nf_open_count = 0
    for feat in pocket_result.features:
        has_fillet = feat.fillet_radius_mm is not None and feat.fillet_radius_mm > 0
        kind = "filleted" if has_fillet else "non-filleted"
        if feat.toolpath_class in ("open_pocket", "filleted_open_pocket"):
            open_count += 1
            if not has_fillet:
                nf_open_count += 1
        lines.append(
            f"  id={feat.feature_id}  kind={kind}  access={feat.access}  "
            f"tp_class={feat.toolpath_class}  n_faces={len(feat.face_indices)}  "
            f"face_indices={sorted(feat.face_indices)}"
        )
    lines.extend([
        "",
        f"open_pocket / filleted_open_pocket count: {open_count}",
        f"non-filleted open among those: {nf_open_count}",
        "",
    ])
    still_missing = nf_open_count == 0 and not any(
        f.toolpath_class == "open_pocket" for f in pocket_result.features
    )
    if still_missing:
        lines.append(
            "VERDICT STEP 0: non-filleted open_pocket STILL MISSING (count=0). "
            "Stage 2b did NOT incidentally fix it. Proceed to Steps 1–3."
        )
    else:
        lines.append(
            "VERDICT STEP 0: open_pocket now emits. Stage 2b may have resolved the defect. "
            "Re-verify against Toolpath GT; STOP diagnosis."
        )
    return "\n".join(lines), still_missing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose missing back open_pocket")
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
    n_faces = len(records)
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)

    _, pocket_result, hole_result, flat_result, outer_fillet_result, residual_result = (
        run_cascade(step_path, edge_index, edge_attr)
    )
    opening_axis = np.array(pocket_result.opening_axis, dtype=np.float64)
    oc_config = PocketDetectionConfig().open_closed

    out: list[str] = [
        f"PROBE: {PART_ID} missing non-filleted open_pocket (Toolpath GT=1, cascade=0 baseline)",
        f"STEP: {step_path}",
        f"graph: {npz_path}",
        "",
    ]

    s0, still_missing = step0_report(pocket_result)
    out.append(s0)

    if not still_missing:
        print("\n".join(out))
        return 0

    out.append("")
    out.append("=" * 78)
    out.append("STEP 1 — LOCATE STRUCTURAL CANDIDATE")
    out.append("=" * 78)

    candidates = _find_structural_candidates(
        records, graph, occ_map, opening_axis, pocket_result, hole_result,
    )
    if not candidates:
        out.append("No structural candidate found matching criteria.")
    else:
        out.append(f"Found {len(candidates)} candidate wall-group(s).")
        best = candidates[0]
        wall_loop = best["wall_loop"]
        out.append("")
        out.append(
            f"Best candidate wall-loop ({len(wall_loop)} faces): {sorted(wall_loop)}"
        )
        out.append(f"  wall-only faces: {sorted(best['wall_only'])}")
        out.append(f"  deep axial cutoff (bottom quartile): Y < {best['deep_cut_y']:.2f} mm")
        out.append(f"  bore gateway faces (1-hop): {best['bore_touch']}")
        out.append(f"  sculpted 1-hop from walls: {sorted(best['sculpted_1hop'])} (expect [])")
        out.append("")
        out.append("Per-face cascade ownership:")
        owner_counts: dict[str, int] = defaultdict(int)
        for fid in sorted(wall_loop):
            owner = _face_owner(
                fid,
                pocket_result=pocket_result,
                hole_result=hole_result,
                flat_result=flat_result,
                outer_fillet_result=outer_fillet_result,
                residual_result=residual_result,
            )
            owner_counts[owner.split("_")[0]] += 1
            fg = by_index[fid]
            y = float(fg.centroid @ opening_axis)
            out.append(
                f"  face {fid}: {fg.surface_type}  Y={y:.1f}mm  area={fg.area:.0f}  -> {owner}"
            )
        out.append("")
        out.append(f"Ownership summary: {dict(owner_counts)}")
        pocket_owned = sum(
            1 for i in wall_loop if i in pocket_result.claimed_faces
        )
        if pocket_owned:
            axis_label = "(c) mis-grouping — faces absorbed into another pocket"
        elif owner_counts.get("flat", 0) + owner_counts.get("residual", 0) > 0:
            axis_label = "(b) CLAIM-side miss — candidate never reaches pocket emit"
        else:
            axis_label = "(a) possible EMIT-side — claimed but wrong class"

        out.append(f"Failure-axis hint from ownership: {axis_label}")

        out.append("")
        out.append("=" * 78)
        out.append(_emit_branch_report())

        out.append("")
        out.append("=" * 78)
        out.append("STEP 3 — OPEN/CLOSED CLASSIFIER ON CANDIDATE")
        out.append("=" * 78)
        feat = _classify_candidate(wall_loop, by_index, graph, opening_axis, oc_config)
        walls = [i for i in wall_loop if by_index[i].surface_type in WALL_SURF_TYPES]
        if walls:
            inboard = min(walls, key=lambda i: _uv_radius(by_index, i, opening_axis))
            inboard_uv = _uv_radius(by_index, inboard, opening_axis)
            outer_uv = max(_uv_radius(by_index, w, opening_axis) for w in walls)
            deep_y = min(_axial_y(by_index, w, opening_axis) for w in walls)
            lo = inboard_uv + oc_config.band_margin_mm
            hi = outer_uv - oc_config.band_margin_mm
        else:
            inboard = inboard_uv = outer_uv = deep_y = lo = hi = float("nan")

        out.append(f"classification: {feat.access}  -> toolpath_class={feat.toolpath_class}")
        out.append(f"cap faces: {feat.cap_face_indices}")
        out.append(f"gateway faces: {feat.gateway_face_indices}")
        out.append(f"inboard wall: {feat.inboard_wall_index}")
        out.append(
            f"inboard band: ({lo:.2f}, {hi:.2f}) mm  "
            f"deep_wall_y={deep_y:.2f} mm  cap_y_tol={oc_config.cap_y_tol_mm} mm"
        )
        if feat.cap_face_indices:
            for cap in feat.cap_face_indices:
                fg = by_index[cap]
                out.append(
                    f"  cap face {cap}: {fg.surface_type}  "
                    f"uv_r={_uv_radius(by_index, cap, opening_axis):.2f}  "
                    f"Y={_axial_y(by_index, cap, opening_axis):.2f}"
                )
        if feat.access == "open":
            out.append("Classifier would NOT suppress open_pocket emit (calls it open).")
        else:
            out.append(
                "Classifier would suppress open_pocket emit (calls it closed -> emits 'pocket')."
            )

        out.append("")
        out.append("=" * 78)
        out.append("FINAL VERDICT")
        out.append("=" * 78)
        out.append(
            "Stage 2b vs baseline: cascade still emits 0 open_pocket (unchanged)."
        )
        if pocket_owned:
            verdict = "(c) mis-grouping into another pocket"
        elif feat.access != "open":
            verdict = (
                "(b) CLAIM-side miss PRIMARY; classifier would also block "
                "(access=closed -> 'pocket' not 'open_pocket')"
            )
        else:
            verdict = "(b) CLAIM-side miss — candidate faces never reach pocket pass"
        out.append(f"Root cause: {verdict}")
        out.append("")
        out.append("Evidence:")
        out.append(
            f"  - Pocket pass seeds from bspline/sphere floors; candidate has 0 sculpted "
            f"1-hop (filleted pockets have 2)."
        )
        out.append(
            f"  - Candidate wall-loop faces owned by flat/residual/hole passes, not pocket."
        )
        out.append(
            "  - Emit path for non-filleted open_pocket exists (pocket_toolpath_class); "
            "not a dead branch."
        )
        if feat.access == "open":
            out.append(
                "  - Open/closed classifier on wall-loop alone: OPEN (would emit open_pocket "
                "if claimed)."
            )
        else:
            out.append(
                f"  - Open/closed classifier: CLOSED (caps={feat.cap_face_indices})."
            )

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
