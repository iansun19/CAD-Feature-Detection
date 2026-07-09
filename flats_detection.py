"""
flats_detection.py — Cascade stage 3: STANDALONE FLAT DETECTION.

Where this fits
---------------
After the pocket pass (which now claims per-pocket step planes via membership
grow) and the hole pass, the flats recognizer consumes the shared residual pool
and claims genuine standalone seating/machining planes. Central-stack end caps
and pocket step planes are explicitly NOT flats — taxonomy differs from
Toolpath's "FACES (8)" count, which files pocket steps under pockets.

    pockets  ->  holes  ->  flats  ->  contour  ->  outer fillets

Integration points (reused, verified against this repo)
-------------------------------------------------------
* Per-face geometry ...... feature_params.analyze_step()  -> list[FaceGeom]
* Face graph ............. edge_index/edge_attr from graph.npz (FaceGraph adapter)
* Hole claimed set ....... passed in so central-stack end caps adjacent to hole
                           walls can be deferred without hardcoded face indices.

Units are mm. numpy only.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from part_scale import resolve_scaled_mm2

import numpy as np

from hole_detection import FaceGraph, _unit
from outer_fillet_detection import (
    OuterFilletDetectionConfig,
    _face_edges,
    _rule_b_larger_exterior_convex,
    _rule_c_smooth_profile_cylinder,
    _rule_d_exterior_not_pocket_enclosed,
)
from pocket_detection import _part_axis_top

logger = logging.getLogger("flats_detection")

_OUTER_FILLET_CFG = OuterFilletDetectionConfig()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class FlatDetectionConfig:
    """Tolerances for coplanar grouping and flat vs end-cap discrimination."""

    # Plane-equation grouping (stable at dist_tol=0.5, ang_round=2 on ref part).
    plane_normal_round: int = 2
    plane_offset_tol_mm: float = 0.5

    # Standalone flat must exceed this total group area (mm²). Absolute if set;
    # when None derived as ``min_flat_area_frac * part scale**2`` so the gate
    # tracks part size instead of 96260B's ~500 mm² plate flats.
    min_flat_area_mm2: float | None = None
    min_flat_area_frac: float = 0.00991  # 500 mm² / (224.6 mm)² ref plate
    # Central-stack end caps are tiny and adjacent to a claimed hole wall — defer to contour/central pass.
    endcap_max_area_mm2: float | None = None
    endcap_max_area_frac: float = 0.003965  # 200 mm² / (224.6 mm)² ref plate
    # BFS hops to reach claimed hole wall geometry from a tiny end cap (330/332 reach
    # 329/347 via the large flat 322 within 2 hops on the ref part).
    central_stack_reach_hops: int = 3

    units: str = "mm"


# ---------------------------------------------------------------------------
# Feature containers
# ---------------------------------------------------------------------------
@dataclass
class FlatFeature:
    feature_id: int
    kind: str  # always "flat"
    face_indices: set[int]
    normal: tuple[float, float, float]
    offset: float  # signed distance from origin along the sign-normalized normal
    area: float

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "face_indices": sorted(self.face_indices),
            "n_faces": self.n_faces,
            "normal": [round(float(x), 6) for x in self.normal],
            "offset": round(float(self.offset), 6),
            "area": round(float(self.area), 3),
        }


@dataclass
class FlatDetectionResult:
    features: list[FlatFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int
    deferred_faces: list[int] = field(default_factory=list)
    deferred_reasons: dict[int, str] = field(default_factory=dict)
    plane_groups: list[list[int]] = field(default_factory=list)
    units: str = "mm"

    def summary(self) -> str:
        return (
            f"{len(self.features)} flats; claimed {len(self.claimed_faces)}/{self.n_faces} "
            f"faces, {len(self.remaining_faces)} remain for contour pass"
        )


# ---------------------------------------------------------------------------
# Plane grouping (coplanar equation keys)
# ---------------------------------------------------------------------------
def _plane_equation_key(
    normal: Sequence[float],
    centroid: Sequence[float],
    config: FlatDetectionConfig,
) -> tuple[tuple[float, ...], float]:
    """Sign-normalized normal + quantized signed offset (coplanar, not co-oriented)."""
    n = _unit(np.asarray(normal, dtype=np.float64))
    if n[int(np.argmax(np.abs(n)))] < 0:
        n = -n
    d = float(np.dot(np.asarray(centroid, dtype=np.float64), n))
    tol = config.plane_offset_tol_mm
    d_q = round(d / tol) * tol
    n_q = tuple(float(x) for x in np.round(n, config.plane_normal_round))
    return n_q, d_q


def _group_planes(
    plane_ids: Sequence[int],
    by_index: dict[int, Any],
    config: FlatDetectionConfig,
) -> dict[tuple[tuple[float, ...], float], list[int]]:
    groups: dict[tuple[tuple[float, ...], float], list[int]] = defaultdict(list)
    for idx in plane_ids:
        fg = by_index[idx]
        key = _plane_equation_key(fg.normal, fg.centroid, config)
        groups[key].append(idx)
    return groups


# ---------------------------------------------------------------------------
# Deep contour-plane exclusion (Cause B — relative to pocket tier on this part)
# ---------------------------------------------------------------------------
def _face_depth_below_top_mm(
    face_idx: int,
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    part_axis_top: float,
) -> float:
    axis = _unit(opening_axis)
    c = np.asarray(by_index[face_idx].centroid, dtype=np.float64)
    return part_axis_top - float(np.dot(c, axis))


def _deepest_pocket_plane_depth_mm(
    pocket_claimed: set[int],
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    part_axis_top: float,
) -> float | None:
    """Deepest claimed pocket plane on this part — the pocket-tier floor."""
    depths = [
        _face_depth_below_top_mm(i, by_index, opening_axis, part_axis_top)
        for i in pocket_claimed
        if by_index[i].surface_type == "plane"
    ]
    return max(depths) if depths else None


def _is_structural_outer_fillet_face(
    face_idx: int,
    *,
    by_index: dict[int, Any],
    pocket_claimed: set[int],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> bool:
    edges = _face_edges(face_idx, edge_index, edge_attr)
    return (
        _rule_b_larger_exterior_convex(
            face_idx,
            by_index=by_index,
            pocket_claimed=pocket_claimed,
            edges=edges,
            exterior_types=_OUTER_FILLET_CFG.exterior_neighbor_types,
        )
        and _rule_c_smooth_profile_cylinder(
            face_idx,
            by_index=by_index,
            pocket_claimed=pocket_claimed,
            edges=edges,
            smooth_neighbor_type=_OUTER_FILLET_CFG.smooth_neighbor_type,
        )
        and _rule_d_exterior_not_pocket_enclosed(
            pocket_claimed=pocket_claimed, edges=edges,
        )
    )


def _has_pocket_blend_neighbor(
    face_idx: int,
    *,
    graph: FaceGraph,
    by_index: dict[int, Any],
    pocket_claimed: set[int],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> bool:
    """Concave edge to an interior pocket fillet that borders a claimed pocket face."""
    for nb in graph.neighbors.get(face_idx, set()):
        kind = graph.edge_kind(face_idx, nb)
        if kind != "concave":
            continue
        st = by_index[nb].surface_type
        if st not in ("torus", "cylinder", "sphere"):
            continue
        if _is_structural_outer_fillet_face(
            nb, by_index=by_index, pocket_claimed=pocket_claimed,
            edge_index=edge_index, edge_attr=edge_attr,
        ):
            continue
        if any(mid in pocket_claimed for mid in graph.neighbors.get(nb, set())):
            return True
    return False


# ---------------------------------------------------------------------------
# Deep contour area-tier helper
# ---------------------------------------------------------------------------
def _area_tiers_from_gaps(areas: Sequence[float]) -> list[list[float]]:
    """Cluster areas by the largest natural gaps (data-derived tiers)."""
    items = sorted(set(float(a) for a in areas))
    if len(items) <= 1:
        return [items] if items else []
    gaps = [
        (items[k + 1] - items[k], k)
        for k in range(len(items) - 1)
        if items[k + 1] - items[k] > 1e-6
    ]
    if not gaps:
        return [items]
    gaps.sort(key=lambda x: -x[0])
    split_at = sorted(idx for _, idx in gaps[: min(len(gaps), max(1, len(items) - 2))])
    tiers: list[list[float]] = []
    start = 0
    for idx in split_at:
        tiers.append(items[start: idx + 1])
        start = idx + 1
    tiers.append(items[start:])
    return tiers


def _tier_index(area: float, tiers: Sequence[Sequence[float]]) -> int:
    a = float(area)
    for i, tier in enumerate(tiers):
        if any(abs(a - t) < max(1.0, 0.01 * t) for t in tier):
            return i
    return -1


def _is_deep_contour_plane(
    face_idx: int,
    *,
    pool_plane_ids: Sequence[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    pocket_claimed: set[int],
    hole_claimed: set[int],
    opening_axis: np.ndarray,
    part_axis_top: float,
    pocket_tier_depth_mm: float,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    pocket_floor_absorbed: set[int] | None = None,
) -> bool:
    """Deep plane leaking into flats on area alone — route to residual/contour.

    Relative gates only (no hardcoded mm/mm²/hop counts):
      * deeper than the deepest pocket-claimed plane on this part;
      * no direct edge to a pocket-claimed face;
      * not connected via a concave pocket-interior blend whose fillet borders
        a claimed pocket wall;
      * among deep pool planes, a smaller-area pocket-step band exists (at least
        one member has a pocket-blend edge above) — larger-area orphans in a
        higher area tier are contour slabs (front-plate signature).
      * when pocket floors were just absorbed, also defer hole-adjacent or
        outer-fillet-adjacent deep contour slabs (front-plate back-geometry leak).
    """
    pocket_floor_absorbed = pocket_floor_absorbed or set()
    fg = by_index[face_idx]
    if fg.surface_type != "plane":
        return False

    depth = _face_depth_below_top_mm(face_idx, by_index, opening_axis, part_axis_top)
    if depth <= pocket_tier_depth_mm + 1e-6:
        return False

    nbrs = graph.neighbors.get(face_idx, set())
    if any(nb in pocket_claimed for nb in nbrs):
        return False
    if _has_pocket_blend_neighbor(
        face_idx, graph=graph, by_index=by_index, pocket_claimed=pocket_claimed,
        edge_index=edge_index, edge_attr=edge_attr,
    ):
        return False

    if pocket_floor_absorbed:
        if any(nb in hole_claimed for nb in nbrs):
            return True
        if any(
            _is_structural_outer_fillet_face(
                nb, by_index=by_index, pocket_claimed=pocket_claimed,
                edge_index=edge_index, edge_attr=edge_attr,
            )
            for nb in nbrs
        ):
            return True

    deep_pool = [
        i for i in pool_plane_ids
        if _face_depth_below_top_mm(i, by_index, opening_axis, part_axis_top)
        > pocket_tier_depth_mm + 1e-6
    ]
    if face_idx not in deep_pool:
        return False

    pocket_deep_steps = [
        i for i in pocket_claimed
        if by_index[i].surface_type == "plane"
        and _face_depth_below_top_mm(i, by_index, opening_axis, part_axis_top)
        > pocket_tier_depth_mm + 1e-6
        and _has_pocket_blend_neighbor(
            i, graph=graph, by_index=by_index, pocket_claimed=pocket_claimed,
            edge_index=edge_index, edge_attr=edge_attr,
        )
    ]
    tier_universe = sorted(set(deep_pool) | set(pocket_deep_steps))
    area_tiers = _area_tiers_from_gaps(float(by_index[i].area) for i in tier_universe)
    if len(area_tiers) < 2:
        return False

    step_tier_idx: int | None = None
    for ti, tier in enumerate(area_tiers):
        rep = tier[0]
        if any(
            i in pocket_deep_steps
            and abs(float(by_index[i].area) - rep) < max(1.0, 0.01 * rep)
            for i in tier_universe
        ):
            step_tier_idx = ti
            break
        if any(
            _has_pocket_blend_neighbor(
                i, graph=graph, by_index=by_index, pocket_claimed=pocket_claimed,
                edge_index=edge_index, edge_attr=edge_attr,
            )
            for i in deep_pool
            if abs(float(by_index[i].area) - rep) < max(1.0, 0.01 * rep)
        ):
            step_tier_idx = ti
            break
    if step_tier_idx is None:
        return False

    face_tier = _tier_index(float(fg.area), area_tiers)
    return face_tier > step_tier_idx


# ---------------------------------------------------------------------------
# Flat vs central-stack end-cap discrimination
# ---------------------------------------------------------------------------
def _reaches_hole_wall(
    face_idx: int,
    graph: FaceGraph,
    hole_claimed: set[int],
    by_index: dict[int, Any],
    max_hops: int,
) -> bool:
    """True when a BFS within `max_hops` hits a claimed hole cylinder/cone wall."""
    if max_hops <= 0:
        return False
    visited: set[int] = {face_idx}
    frontier = [face_idx]
    for _depth in range(max_hops):
        nxt: list[int] = []
        for f in frontier:
            for nb in graph.neighbors.get(f, set()):
                if nb in visited:
                    continue
                visited.add(nb)
                if nb in hole_claimed:
                    st = getattr(by_index[nb], "surface_type", None)
                    if st in ("cylinder", "cone"):
                        return True
                nxt.append(nb)
        frontier = nxt
    return False


def _is_central_stack_endcap(
    face_idx: int,
    fg: Any,
    graph: FaceGraph,
    hole_claimed: set[int],
    by_index: dict[int, Any],
    config: FlatDetectionConfig,
) -> bool:
    """Tiny plane on the central through/blind stack — defer to contour/central pass."""
    if float(fg.area) > config.endcap_max_area_mm2:
        return False
    nbrs = graph.neighbors.get(face_idx, set())
    for nb in nbrs:
        if nb not in hole_claimed:
            continue
        st = getattr(by_index[nb], "surface_type", None)
        if st in ("cylinder", "cone"):
            return True
    return _reaches_hole_wall(
        face_idx, graph, hole_claimed, by_index, config.central_stack_reach_hops,
    )


def _qualifies_as_standalone_flat(
    group_ids: Sequence[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    hole_claimed: set[int],
    config: FlatDetectionConfig,
    *,
    pool_plane_ids: Sequence[int] | None = None,
    pocket_claimed: set[int] | None = None,
    pocket_floor_absorbed: set[int] | None = None,
    opening_axis: np.ndarray | None = None,
    part_axis_top: float | None = None,
    pocket_tier_depth_mm: float | None = None,
    edge_index: np.ndarray | None = None,
    edge_attr: np.ndarray | None = None,
) -> tuple[bool, str]:
    pocket_claimed = pocket_claimed or set()
    for idx in group_ids:
        if _is_central_stack_endcap(idx, by_index[idx], graph, hole_claimed,
                                    by_index, config):
            return False, (
                f"central-stack end cap (face {idx}: "
                f"{by_index[idx].area:.1f} mm², adjacent to hole wall)"
            )
        if (
            pool_plane_ids is not None
            and opening_axis is not None
            and part_axis_top is not None
            and pocket_tier_depth_mm is not None
            and edge_index is not None
            and edge_attr is not None
            and _is_deep_contour_plane(
                idx,
                pool_plane_ids=pool_plane_ids,
                by_index=by_index,
                graph=graph,
                pocket_claimed=pocket_claimed,
                hole_claimed=hole_claimed,
                opening_axis=opening_axis,
                part_axis_top=part_axis_top,
                pocket_tier_depth_mm=pocket_tier_depth_mm,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pocket_floor_absorbed=pocket_floor_absorbed or set(),
            )
        ):
            return False, (
                f"deep contour plane (face {idx}: "
                f"below pocket tier, isolated from pocket/opening skin)"
            )
    total_area = sum(float(by_index[i].area) for i in group_ids)
    if total_area < config.min_flat_area_mm2:
        return False, f"area {total_area:.1f} mm² below min_flat_area"
    return True, "standalone seating plane"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_flats(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    hole_claimed_faces: set[int] | None = None,
    pocket_claimed_faces: set[int] | None = None,
    pocket_floor_absorbed_faces: set[int] | None = None,
    hub_flat_faces: set[int] | None = None,
    opening_axis: Sequence[float] | None = None,
    occ_faces: Sequence[Any] | None = None,
    config: FlatDetectionConfig | None = None,
) -> FlatDetectionResult:
    """Run the flats pass over `candidate_faces` (default: all faces).

    Parameters
    ----------
    faces:
        FaceGeom records from feature_params.analyze_step.
    edge_index, edge_attr:
        Face-adjacency tensors from graph.npz.
    candidate_faces:
        Still-unclaimed face indices (pocket+hole residual in the cascade).
    hole_claimed_faces:
        Faces claimed by the hole pass — used to defer central-stack end caps.
    pocket_claimed_faces:
        Faces claimed by the pocket pass — used for deep-contour exclusion.
    hub_flat_faces:
        Hub seating plane(s) from coaxial_stack_detection — pre-claimed as flat.
    opening_axis:
        Unit opening axis from the pocket pass (auto-detected if omitted).
    occ_faces:
        Optional OCC faces for boundary-accurate depth (from load_step_faces).
    """
    config = config or FlatDetectionConfig()
    config = replace(
        config,
        min_flat_area_mm2=resolve_scaled_mm2(
            config.min_flat_area_mm2, config.min_flat_area_frac, faces),
        endcap_max_area_mm2=resolve_scaled_mm2(
            config.endcap_max_area_mm2, config.endcap_max_area_frac, faces),
    )
    n_faces = len(faces)
    by_index = {int(f.index): f for f in faces}
    pool = set(range(n_faces)) if candidate_faces is None else {
        i for i in candidate_faces if 0 <= i < n_faces
    }
    hole_claimed = hole_claimed_faces or set()
    pocket_claimed = pocket_claimed_faces or set()
    pocket_floor_absorbed = pocket_floor_absorbed_faces or set()
    hub_flat = {int(i) for i in (hub_flat_faces or set()) if int(i) in pool}

    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)

    if opening_axis is not None:
        axis = _unit(np.asarray(opening_axis, dtype=np.float64))
    else:
        axes = [
            np.asarray(f.axis, dtype=np.float64)
            for f in faces
            if getattr(f, "surface_type", None) in ("cylinder", "cone")
            and getattr(f, "axis", None) is not None
        ]
        axis = _unit(np.mean(axes, axis=0)) if axes else np.array([0.0, 1.0, 0.0])

    occ_map = {i: occ_faces[i] for i in range(len(occ_faces))} if occ_faces else None
    part_top = _part_axis_top(faces, axis, occ_map)
    pocket_tier_depth = _deepest_pocket_plane_depth_mm(
        pocket_claimed, by_index, axis, part_top,
    )

    plane_ids = sorted(i for i in pool if by_index[i].surface_type == "plane")
    groups_map = _group_planes(plane_ids, by_index, config)
    plane_groups = [sorted(idxs) for idxs in groups_map.values()]

    features: list[FlatFeature] = []
    claimed: set[int] = set()
    deferred: list[int] = []
    deferred_reasons: dict[int, str] = {}
    next_id = 0

    # Hub seating plane(s) from coaxial tier decomposer — bypass deep-contour gate.
    if hub_flat:
        hub_groups_map = _group_planes(sorted(hub_flat), by_index, config)
        for key, group_ids in hub_groups_map.items():
            n_q, d_q = key
            total_area = sum(float(by_index[i].area) for i in group_ids)
            if total_area < config.min_flat_area_mm2:
                for idx in group_ids:
                    deferred.append(idx)
                    deferred_reasons[idx] = (
                        f"hub flat area {total_area:.1f} mm² below min_flat_area"
                    )
                continue
            feat = FlatFeature(
                feature_id=next_id,
                kind="flat",
                face_indices=set(group_ids),
                normal=n_q,
                offset=d_q,
                area=total_area,
            )
            features.append(feat)
            claimed |= feat.face_indices
            next_id += 1
            logger.info(
                "claimed hub flat %d: faces=%s area=%.1f mm² (coaxial tier)",
                feat.feature_id, sorted(group_ids), total_area,
            )

    for key, group_ids in sorted(groups_map.items(), key=lambda kv: -len(kv[1])):
        if all(idx in claimed for idx in group_ids):
            continue
        n_q, d_q = key
        qualify_kwargs: dict[str, Any] = {}
        if pocket_tier_depth is not None:
            qualify_kwargs = dict(
                pool_plane_ids=plane_ids,
                pocket_claimed=pocket_claimed,
                opening_axis=axis,
                part_axis_top=part_top,
                pocket_tier_depth_mm=pocket_tier_depth,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pocket_floor_absorbed=pocket_floor_absorbed,
            )
        unclaimed_group = [idx for idx in group_ids if idx not in claimed]
        if not unclaimed_group:
            continue
        ok, reason = _qualifies_as_standalone_flat(
            unclaimed_group, by_index, graph, hole_claimed, config,
            **qualify_kwargs,
        )
        if ok:
            total_area = sum(float(by_index[i].area) for i in unclaimed_group)
            feat = FlatFeature(
                feature_id=next_id,
                kind="flat",
                face_indices=set(unclaimed_group),
                normal=n_q,
                offset=d_q,
                area=total_area,
            )
            features.append(feat)
            claimed |= feat.face_indices
            next_id += 1
            logger.info(
                "claimed flat %d: faces=%s area=%.1f mm² normal=%s offset=%+.2f mm",
                feat.feature_id, sorted(unclaimed_group), total_area, n_q, d_q,
            )
        else:
            for idx in unclaimed_group:
                deferred.append(idx)
                deferred_reasons[idx] = reason
            logger.info(
                "deferred plane group %s (%d face(s)): %s",
                sorted(unclaimed_group), len(unclaimed_group), reason,
            )

    if deferred:
        logger.info(
            "deferred %d plane face(s) to contour/central-hole pass: %s",
            len(deferred), sorted(deferred),
        )
        for idx in sorted(deferred):
            logger.info("  face %d deferred: %s", idx, deferred_reasons[idx])

    remaining = pool - claimed
    return FlatDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=remaining,
        n_faces=n_faces,
        deferred_faces=sorted(deferred),
        deferred_reasons=deferred_reasons,
        plane_groups=plane_groups,
        units=config.units,
    )


def detect_flats_from_step(
    step_path: str | Path,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    hole_claimed_faces: set[int] | None = None,
    pocket_claimed_faces: set[int] | None = None,
    pocket_floor_absorbed_faces: set[int] | None = None,
    opening_axis: Sequence[float] | None = None,
    config: FlatDetectionConfig | None = None,
) -> FlatDetectionResult:
    from feature_params import analyze_step, load_step_faces, require_occ

    require_occ()
    faces = analyze_step(step_path)
    occ_faces = load_step_faces(step_path)
    return detect_flats(
        faces, edge_index, edge_attr,
        candidate_faces=candidate_faces,
        hole_claimed_faces=hole_claimed_faces,
        pocket_claimed_faces=pocket_claimed_faces,
        pocket_floor_absorbed_faces=pocket_floor_absorbed_faces,
        opening_axis=opening_axis,
        occ_faces=occ_faces,
        config=config,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass
class FlatValidationReport:
    ok: bool
    checks: list[tuple[str, bool, str]]

    def render(self) -> str:
        lines = [f"flat validation: {'PASS' if self.ok else 'FAIL'}"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_flats(
    result: FlatDetectionResult,
    *,
    expected_flats: int = 2,
    expected_faces: Sequence[int] = (97, 273),
    deferred_faces: Sequence[int] = (),
    deferred_reason_substr: str = "central-stack",
) -> FlatValidationReport:
    """Reference-part contract: two standalone flats (opening + hub seating)."""
    checks: list[tuple[str, bool, str]] = []

    n = len(result.features)
    checks.append((f"exactly {expected_flats} flats", n == expected_flats, f"got {n}"))

    claimed_sorted = sorted(result.claimed_faces)
    exp_set = set(expected_faces)
    checks.append((
        f"claimed faces are {list(expected_faces)}",
        set(claimed_sorted) == exp_set,
        f"got {claimed_sorted}",
    ))

    for idx in deferred_faces:
        ok = idx not in result.claimed_faces and idx in result.deferred_faces
        reason = result.deferred_reasons.get(idx, "not in deferred list")
        if ok and deferred_reason_substr not in reason:
            ok = False
            reason = f"expected '{deferred_reason_substr}' in reason; got: {reason}"
        checks.append((
            f"face {idx} deferred (not claimed as flat)",
            ok,
            reason,
        ))

    ok = all(c[1] for c in checks)
    return FlatValidationReport(ok=ok, checks=checks)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_table(result: FlatDetectionResult) -> str:
    header = f"{'id':>3}  {'#f':>3} {'area (mm²)':>12}  {'offset (mm)':>12}  normal"
    lines = [header, "-" * len(header)]
    for f in sorted(result.features, key=lambda x: -x.area):
        n = f.normal
        lines.append(
            f"{f.feature_id:>3}  {f.n_faces:>3} {f.area:>12.1f}  {f.offset:>+12.2f}  "
            f"[{n[0]:+.3f} {n[1]:+.3f} {n[2]:+.3f}]  faces={sorted(f.face_indices)}"
        )
    if result.deferred_faces:
        lines.append("")
        lines.append(f"deferred to contour pass: {result.deferred_faces}")
    lines.append("")
    lines.append(result.summary())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_STEP = "96260B_REAR_XR004_PCD PLATE.stp copy"
DEFAULT_GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Flat-detection pass (cascade stage 3)")
    ap.add_argument("step", nargs="?", default=DEFAULT_STEP)
    ap.add_argument("--graph-npz", type=Path, default=Path(DEFAULT_GRAPH_NPZ))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    step_path = Path(args.step)
    if not step_path.is_file():
        print(f"STEP not found: {step_path}")
        return 2

    from run_cascade import _load_edges, run_cascade

    edge_index, edge_attr = _load_edges(args.graph_npz, step_path)
    _, pk, hl, _, _, _ = run_cascade(step_path, edge_index, edge_attr)
    result = detect_flats(
        __import__("feature_params").analyze_step(step_path),
        edge_index, edge_attr,
        candidate_faces=hl.remaining_faces,
        hole_claimed_faces=hl.claimed_faces,
        pocket_claimed_faces=pk.claimed_faces,
        pocket_floor_absorbed_faces=pk.floor_absorbed_faces,
        opening_axis=pk.opening_axis,
    )

    print(f"STEP: {step_path}")
    print(f"input pool: {len(hl.remaining_faces)} faces (hole pass residual)")
    print()
    print(render_table(result))
    print()
    print(validate_flats(result).render())
    return 0 if validate_flats(result).ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
