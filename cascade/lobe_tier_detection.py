"""
lobe_tier_detection.py — Toolpath filleted lobe boundary + open/closed tier split.

Each filleted radial lobe on the front plate is TWO Toolpath features:
  * filleted_open_pocket  — mouth step band (shallow axial tier)
  * filleted_pocket       — deep floor band (deep axial tier)

This module discovers lobe boundaries without face-ID tables or hardcoded depths:

  1. Cluster global axial step planes into shallow / deep tiers (gap-based).
  2. Pair each mouth step to its nearest deep step in UV (one lobe = one pair).
  3. Grow a bounded lobe pool from the pair through fillet topology; stop at
     other lobes' step anchors (convex inter-lobe seams).
  4. Split the pool into open vs closed tiers from the paired anchors.

Diagnostic entry point: probe_pocket_tier_split.py
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np

from cascade.hole_detection import FaceGraph
from cascade.pocket_detection import (
    FILLET_TYPES,
    PocketDetectionConfig,
    SCULPTED_FLOOR_TYPES,
    WALL_SURF_TYPES,
    _axial_y,
    _depth_tiers_from_gaps,
    _is_axial_plane,
    _project_uv,
    _torus_borders_pocket_walls,
    _wall_interior_and_axis,
)

logger = logging.getLogger("lobe_tier_detection")

MM_PER_IN = 25.4

# Clearance tripwire only (2× → 0.10 mm minimum clearance to attempt tier split).
NUMERIC_FLOOR_MM = 0.05
# Minimum reliable mouth-step boundary tol for axial centroid projection (>= 2× numeric floor).
TOLERANCE_FLOOR_MM = 0.10
TOLERANCE_EPSILON_MM = 0.001
# Upper bound on span-derived tol. Measured min clearance 1.379 mm (96260B_front) /
# 1.430 mm (96260B_rear) on mouth↔deep span 5.08 mm → cap 0.254 mm ≪ half-min 0.69 mm.
DEFAULT_MOUTH_TIER_SPAN_FRACTION = 0.05


class LobeTierBoundaryError(ValueError):
    """Mouth/deep tier boundary cannot be resolved reliably for this lobe."""


@dataclass
class LobeTierConfig:
    """Geometry-derived thresholds (shared with pocket wall band)."""

    wall_dia_min_mm: float = 3.0
    wall_dia_max_mm: float = 100.0
    step_plane_normal_tol_deg: float = 10.0
    require_interior_wall: bool = True
    min_step_tiers: int = 2
    min_steps_per_tier: int = 6
    # dia_family_tol_in is the diameter-match tolerance (inches) used when testing
    # a wall against a BoreFamily role. The former hard-coded 96260B diameters
    # (2.867/3.453/0.800/0.500 in) are now DERIVED per part by build_bore_family()
    # from concentric-bore relationships (primary vs duplicate by size ratio;
    # mouth vs deep by axial depth). Byte-identical on 96260B; recalibrate/extend
    # as new lobed parts arrive (see memory cadcam-roadmap).
    dia_family_tol_in: float = 0.05
    # A wall is a "primary" bore if it is at least this many times larger than a
    # concave/smooth-adjacent wall (which is then its "duplicate"), and vice versa.
    # 96260B: primaries ~2.867/3.453 in, duplicates ~0.800/0.500 in (ratio ~3.6).
    bore_primary_duplicate_ratio: float = 2.0
    mouth_tier_span_fraction: float = DEFAULT_MOUTH_TIER_SPAN_FRACTION
    mouth_numeric_floor_mm: float = NUMERIC_FLOOR_MM
    mouth_tolerance_floor_mm: float = TOLERANCE_FLOOR_MM
    # Prototype: annex sphere fillet caps sharing convex edges with tier-annexed tori
    # when sphere R ≈ adjacent torus minor R (relative tolerance).
    annex_fillet_sphere_caps: bool = True
    fillet_sphere_cap_radius_rel_tol: float = 0.05
    # Cap area / matched torus area must be ≤ this ratio (part-agnostic sculpt-sphere filter).
    # 96260B: real split-export caps ≈ 0.149; nearest sculpt-sphere FP ≈ 0.206.
    fillet_sphere_cap_max_torus_area_ratio: float | None = 0.18
    # Sculpt cap bsplines convex to a cap sphere above this UV v stay closed-tier
    # (96260B rear crown caps at v≈64; mouth-band caps are below ≈32 or negative v).
    sculpt_cap_crown_uv_v_mm: float = 55.0


FILLET_SPHERE_CAP_EDGE_KINDS = frozenset({"convex", "concave"})
SCULPT_CAP_BSPLINE_EDGE_KIND = "convex"


@dataclass
class FilletedLobeTier:
    lobe_id: int
    mouth_step_face: int
    deep_step_face: int
    mouth_axial: float
    deep_axial: float
    pool_faces: set[int]
    open_faces: set[int]
    closed_faces: set[int]
    unassigned_faces: set[int]
    centroid_uv: tuple[float, float]
    fillet_sphere_cap_annex: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_open(self) -> int:
        return len(self.open_faces)

    @property
    def n_closed(self) -> int:
        return len(self.closed_faces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lobe_id": self.lobe_id,
            "mouth_step_face": self.mouth_step_face,
            "deep_step_face": self.deep_step_face,
            "mouth_axial_mm": round(self.mouth_axial, 4),
            "deep_axial_mm": round(self.deep_axial, 4),
            "pool_faces": sorted(self.pool_faces),
            "open_faces": sorted(self.open_faces),
            "closed_faces": sorted(self.closed_faces),
            "open_class": "filleted_open_pocket",
            "closed_class": "filleted_pocket",
            "n_open": self.n_open,
            "n_closed": self.n_closed,
            "fillet_sphere_cap_annex": self.fillet_sphere_cap_annex,
        }


@dataclass
class LobeTierDetectionResult:
    lobes: list[FilletedLobeTier]
    mouth_step_faces: list[int]
    deep_step_faces: list[int]
    opening_axis: tuple[float, float, float]
    tier_axial_bins: list[tuple[float, list[int]]]
    n_faces: int

    def all_open_faces(self) -> set[int]:
        out: set[int] = set()
        for lob in self.lobes:
            out |= lob.open_faces
        return out

    def all_closed_faces(self) -> set[int]:
        out: set[int] = set()
        for lob in self.lobes:
            out |= lob.closed_faces
        return out

    def summary(self) -> str:
        return (
            f"{len(self.lobes)} filleted lobes; "
            f"open={len(self.all_open_faces())} closed={len(self.all_closed_faces())} faces"
        )


def _pocket_cfg(cfg: LobeTierConfig) -> PocketDetectionConfig:
    return PocketDetectionConfig(
        step_plane_normal_tol_deg=cfg.step_plane_normal_tol_deg,
        wall_dia_min_mm=cfg.wall_dia_min_mm,
        wall_dia_max_mm=cfg.wall_dia_max_mm,
        require_interior_wall=cfg.require_interior_wall,
    )


def _near_dia_in(d_mm: float | None, target_in: float, tol_in: float) -> bool:
    if d_mm is None:
        return False
    return abs(d_mm / MM_PER_IN - target_in) <= tol_in


@dataclass
class BoreFamily:
    """Per-part concentric-bore role table, derived from wall relationships.

    Replaces the former hard-coded 96260B diameters. Roles are assigned by
    RELATIONSHIP, not absolute size: an interior bore is a *primary* if it has a
    much-smaller concave/smooth-adjacent *duplicate* bore (size ratio >=
    ``bore_primary_duplicate_ratio``); of the primaries the axially shallower one
    is the mouth (open-tier) bore and the deeper one is the deep (closed-tier)
    bore; remaining small bores are duplicates ranked largest-first. The central
    hub bore (no small duplicate neighbour) is excluded. Matching still uses the
    ``dia_family_tol_in`` tolerance, so on 96260B the derived targets reproduce
    the old 2.867/3.453/0.800/0.500 in matches byte-for-byte.
    """

    mouth_primary_mm: float | None
    deep_primary_mm: float | None
    duplicates_mm: tuple[float, ...]  # descending diameter
    tol_in: float

    def _near(self, d_mm: float | None, target_mm: float | None) -> bool:
        if d_mm is None or target_mm is None:
            return False
        return abs(float(d_mm) / MM_PER_IN - float(target_mm) / MM_PER_IN) <= self.tol_in

    def is_mouth_primary(self, d_mm: float | None) -> bool:
        return self._near(d_mm, self.mouth_primary_mm)

    def is_deep_primary(self, d_mm: float | None) -> bool:
        return self._near(d_mm, self.deep_primary_mm)

    def is_duplicate(self, d_mm: float | None, rank: int | None = None) -> bool:
        """True if d matches a duplicate bore; if ``rank`` given, that specific
        size rank (0 = largest duplicate)."""
        if rank is None:
            return any(self._near(d_mm, x) for x in self.duplicates_mm)
        if rank < 0 or rank >= len(self.duplicates_mm):
            return False
        return self._near(d_mm, self.duplicates_mm[rank])


def _cluster_diameters(diams: Sequence[float], tol_in: float) -> list[float]:
    """Collapse near-equal diameters (within tol) to representatives, descending."""
    reps: list[list[float]] = []
    for d in sorted(diams, reverse=True):
        for c in reps:
            if abs(d - c[0]) / MM_PER_IN <= tol_in:
                c.append(d)
                break
        else:
            reps.append([d])
    return [float(np.mean(c)) for c in reps]


def build_bore_family(
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    axis: np.ndarray,
    cfg: LobeTierConfig,
) -> BoreFamily:
    """Derive the concentric-bore role table from wall relationships (see BoreFamily)."""
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    a = a / n if n > 1e-12 else a

    walls: dict[int, tuple[float, float]] = {}  # fid -> (diameter_mm, axial)
    for fid, fg in by_index.items():
        if fg.surface_type not in WALL_SURF_TYPES:
            continue
        info = _wall_interior_and_axis(fg, occ_map.get(fid) if occ_map else None)
        if info is None:
            continue
        interior, _, dia = info
        if not interior or dia is None:
            continue
        if dia < cfg.wall_dia_min_mm or dia > cfg.wall_dia_max_mm:
            continue
        axial = float(np.dot(np.asarray(fg.centroid, dtype=np.float64), a))
        walls[fid] = (float(dia), axial)

    ratio = cfg.bore_primary_duplicate_ratio
    primaries: list[tuple[float, float]] = []  # (diameter, axial)
    duplicates: list[float] = []
    for fid, (dia, axial) in walls.items():
        is_primary = is_duplicate = False
        for nb in graph.neighbors.get(fid, ()):
            if graph.edge_kind(fid, nb) not in ("concave", "smooth"):
                continue
            if nb not in walls:
                continue
            nd = walls[nb][0]
            if dia >= ratio * nd:
                is_primary = True
            if nd >= ratio * dia:
                is_duplicate = True
        if is_primary:
            primaries.append((dia, axial))
        if is_duplicate:
            duplicates.append(dia)

    # Cluster primary diameters; assign mouth (shallowest = max axial along +axis,
    # which points to the part top) and deep (deepest = min axial).
    prim_clusters: dict[float, list[float]] = {}
    for dia, axial in primaries:
        key = next(
            (k for k in prim_clusters if abs(dia - k) / MM_PER_IN <= cfg.dia_family_tol_in),
            None,
        )
        prim_clusters.setdefault(dia if key is None else key, []).append(axial)

    mouth_primary_mm: float | None = None
    deep_primary_mm: float | None = None
    if prim_clusters:
        ordered = sorted(prim_clusters.items(), key=lambda kv: float(np.mean(kv[1])))
        deep_primary_mm = ordered[0][0]
        mouth_primary_mm = ordered[-1][0]

    return BoreFamily(
        mouth_primary_mm=mouth_primary_mm,
        deep_primary_mm=deep_primary_mm,
        duplicates_mm=tuple(_cluster_diameters(duplicates, cfg.dia_family_tol_in)),
        tol_in=cfg.dia_family_tol_in,
    )


def _ensure_bore_family(
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    axis: np.ndarray,
    cfg: LobeTierConfig,
) -> BoreFamily:
    """Build and stash the BoreFamily on the graph once; idempotent.

    Called at every entry that has the opening axis (the main detection entry and
    the tolerance/trace helpers that tests invoke directly on a fresh graph), so
    the axis-free diameter helpers can fetch it via _bore_family."""
    fam = getattr(graph, "bore_family", None)
    if fam is None:
        fam = build_bore_family(by_index, occ_map, graph, axis, cfg)
        graph.bore_family = fam
    return fam


def _bore_family(graph: FaceGraph) -> BoreFamily:
    """Fetch the BoreFamily stashed on the graph (see _ensure_bore_family)."""
    fam = getattr(graph, "bore_family", None)
    if fam is None:
        raise RuntimeError(
            "bore family not built; call _ensure_bore_family at the axis-carrying entry"
        )
    return fam


def _opening_blend(
    fg: Any,
    occ_face: Any | None,
    cfg: LobeTierConfig,
) -> bool:
    if fg.surface_type not in WALL_SURF_TYPES:
        return False
    info = _wall_interior_and_axis(fg, occ_face)
    if info is None:
        return False
    is_interior, _, diameter = info
    if cfg.require_interior_wall and not is_interior:
        return False
    d = 0.0 if diameter is None else float(diameter)
    return d > cfg.wall_dia_max_mm


def _wall_band(
    fg: Any,
    occ_face: Any | None,
    cfg: LobeTierConfig,
) -> bool:
    if fg.surface_type not in WALL_SURF_TYPES:
        return False
    info = _wall_interior_and_axis(fg, occ_face)
    if info is None:
        return False
    is_interior, _, diameter = info
    if cfg.require_interior_wall and not is_interior:
        return False
    d = 0.0 if diameter is None else float(diameter)
    if cfg.wall_dia_min_mm is not None and d < cfg.wall_dia_min_mm:
        return False
    if cfg.wall_dia_max_mm is not None and d > cfg.wall_dia_max_mm:
        return False
    return True


def _lobe_traversable(
    face_idx: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> bool:
    fg = by_index[face_idx]
    st = fg.surface_type
    if st in FILLET_TYPES or st in SCULPTED_FLOOR_TYPES:
        return True
    if st == "plane":
        return _is_axial_plane(fg, opening_axis, _pocket_cfg(cfg))
    if st in WALL_SURF_TYPES:
        occ = occ_map.get(face_idx) if occ_map else None
        return _wall_band(fg, occ, cfg) or _opening_blend(fg, occ, cfg)
    return False


def _lobe_edge_ok(graph: FaceGraph, u: int, v: int, growing: set[int]) -> bool:
    kind = graph.edge_kind(u, v)
    if kind in ("smooth", "concave"):
        return True
    if kind == "convex":
        return u in growing or v in growing
    return False


def _collect_axial_step_planes(
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for f, fg in by_index.items():
        if fg.surface_type != "plane":
            continue
        if not _is_axial_plane(fg, opening_axis, _pocket_cfg(cfg)):
            continue
        out.append((f, _axial_y(by_index, f, opening_axis)))
    return out


def discover_step_tier_bins(
    step_planes: list[tuple[int, float]],
    cfg: LobeTierConfig,
) -> list[tuple[float, list[int]]]:
    """Group step planes into axial bins; return bins sorted shallow → deep."""
    bins: dict[float, list[int]] = defaultdict(list)
    for f, ax in step_planes:
        key = round(ax, 1)
        bins[key].append(f)
    return sorted(bins.items(), key=lambda kv: -kv[0])


def discover_mouth_and_deep_steps(
    step_planes: list[tuple[int, float]],
    cfg: LobeTierConfig,
    *,
    n_lobes_hint: int | None = None,
) -> tuple[list[int], list[int], list[tuple[float, list[int]]]]:
    """Pick mouth (shallow) and deep step face lists from axial bin histogram."""
    bins = discover_step_tier_bins(step_planes, cfg)
    if len(bins) < cfg.min_step_tiers:
        return [], [], bins

    min_count = cfg.min_steps_per_tier
    if n_lobes_hint is not None:
        min_count = max(min_count, int(n_lobes_hint * 0.85))

    tier_bins = [b for b in bins if len(b[1]) >= min_count]
    if len(tier_bins) < cfg.min_step_tiers:
        tier_bins = [b for b in bins if len(b[1]) >= max(2, min_count // 2)]
    if len(tier_bins) < cfg.min_step_tiers:
        return [], [], bins

    mouth = tier_bins[0][1]
    deep = tier_bins[1][1]
    return mouth, deep, bins


def pair_mouth_deep_steps(
    mouth_steps: Sequence[int],
    deep_steps: Sequence[int],
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
) -> list[tuple[int, int]]:
    """One (mouth, deep) pair per mouth step — nearest deep step in UV."""
    pairs: list[tuple[int, int]] = []
    used_deep: set[int] = set()
    deep_uv = {
        d: _project_uv(np.asarray(by_index[d].centroid, dtype=np.float64), opening_axis)
        for d in deep_steps
    }
    for m in sorted(mouth_steps):
        m_uv = _project_uv(np.asarray(by_index[m].centroid, dtype=np.float64), opening_axis)
        candidates = [d for d in deep_steps if d not in used_deep] or list(deep_steps)
        best_d = min(candidates, key=lambda d: float(np.linalg.norm(m_uv - deep_uv[d])))
        pairs.append((m, best_d))
        used_deep.add(best_d)
    return pairs


def build_anchor_registry(
    pairs: Sequence[tuple[int, int]],
) -> dict[int, int]:
    reg: dict[int, int] = {}
    for lid, (mouth, deep) in enumerate(pairs):
        reg[mouth] = lid
        reg[deep] = lid
    return reg


def _in_pocket_axial_band(
    axial: float,
    mouth_axial: float,
    deep_axial: float,
    *,
    below_deep_mm: float = 12.0,
    above_mouth_mm: float = 10.5,
) -> bool:
    """Pocket-relevant band: deep floor through mouth and opening bore extension."""
    shallow = max(mouth_axial, deep_axial)
    deep = min(mouth_axial, deep_axial)
    return (deep - below_deep_mm) <= axial <= (shallow + above_mouth_mm)


def grow_lobe_pool_from_anchors(
    mouth_step: int,
    deep_step: int,
    lobe_id: int,
    anchor_registry: dict[int, int],
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
    *,
    mouth_axial: float,
    deep_axial: float,
) -> set[int]:
    """Grow one lobe from its mouth/deep pair; stop at other lobes' step anchors."""
    seeds = {mouth_step, deep_step}
    growing = set(seeds)
    q: deque[int] = deque(sorted(seeds))
    while q:
        u = q.popleft()
        for v in graph.neighbors.get(u, ()):
            if v in growing:
                continue
            owner = anchor_registry.get(v)
            if owner is not None and owner != lobe_id:
                continue
            if not _lobe_traversable(v, by_index, occ_map, opening_axis, cfg):
                continue
            ax = _axial_y(by_index, v, opening_axis)
            if not _in_pocket_axial_band(ax, mouth_axial, deep_axial):
                continue
            if not _lobe_edge_ok(graph, u, v, growing):
                continue
            growing.add(v)
            q.append(v)
    return growing


def assign_lobe_faces_angular(
    pairs: Sequence[tuple[int, int]],
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
    graph: FaceGraph,
    *,
    mouth_axial: float,
    deep_axial: float,
) -> list[set[int]]:
    """Partition pocket faces by nearest mouth-step angle around the hub."""
    mouths = [m for m, _ in pairs]
    mouth_to_lobe = {mouth: lid for lid, (mouth, _d) in enumerate(pairs)}
    mouth_uv = {
        m: _project_uv(np.asarray(by_index[m].centroid, dtype=np.float64), opening_axis)
        for m in mouths
    }
    center = np.mean(np.array(list(mouth_uv.values())), axis=0)
    mouth_angle = {
        m: float(math.atan2(uv[1] - center[1], uv[0] - center[0]))
        for m, uv in mouth_uv.items()
    }
    pools: list[set[int]] = [set() for _ in pairs]

    def _ang_dist(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 2.0 * math.pi - d)

    for f, fg in by_index.items():
        if not _lobe_traversable(f, by_index, occ_map, opening_axis, cfg):
            continue
        ax = _axial_y(by_index, f, opening_axis)
        if not _in_pocket_axial_band(ax, mouth_axial, deep_axial):
            continue
        if _is_exterior_contour_fillet(f, graph, by_index):
            continue
        uv = _project_uv(np.asarray(fg.centroid, dtype=np.float64), opening_axis)
        theta = float(math.atan2(uv[1] - center[1], uv[0] - center[0]))
        best_m = min(mouths, key=lambda m: _ang_dist(theta, mouth_angle[m]))
        pools[mouth_to_lobe[best_m]].add(f)
    return pools


def assign_lobe_faces_voronoi(
    pairs: Sequence[tuple[int, int]],
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
    graph: FaceGraph,
    *,
    mouth_axial: float,
    deep_axial: float,
) -> list[set[int]]:
    """Partition traversable pocket faces by nearest mouth-step UV (one lobe per sector)."""
    mouth_to_lobe = {mouth: lid for lid, (mouth, _deep) in enumerate(pairs)}
    mouth_uv = {
        m: _project_uv(np.asarray(by_index[m].centroid, dtype=np.float64), opening_axis)
        for m, _ in pairs
    }
    pools: list[set[int]] = [set() for _ in pairs]
    for f, fg in by_index.items():
        if not _lobe_traversable(f, by_index, occ_map, opening_axis, cfg):
            continue
        ax = _axial_y(by_index, f, opening_axis)
        if not _in_pocket_axial_band(ax, mouth_axial, deep_axial):
            continue
        if _is_exterior_contour_fillet(f, graph, by_index):
            continue
        uv = _project_uv(np.asarray(fg.centroid, dtype=np.float64), opening_axis)
        best_m = min(
            mouth_uv.keys(),
            key=lambda m: float(np.linalg.norm(uv - mouth_uv[m])),
        )
        pools[mouth_to_lobe[best_m]].add(f)
    return pools


def refine_pool_by_fillet_connectivity(
    pool: set[int],
    mouth_step: int,
    deep_step: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> set[int]:
    """Keep only faces in `pool` reachable from {mouth, deep} without leaving pool."""
    seeds = {mouth_step, deep_step}
    reachable = set(seeds)
    q: deque[int] = deque(sorted(seeds))
    while q:
        u = q.popleft()
        for v in graph.neighbors.get(u, ()):
            if v not in pool or v in reachable:
                continue
            if not _lobe_traversable(v, by_index, occ_map, opening_axis, cfg):
                continue
            if not _lobe_edge_ok(graph, u, v, reachable):
                continue
            reachable.add(v)
            q.append(v)
    return reachable


def _deep_wall_pair_member(
    face_idx: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    cfg: LobeTierConfig,
) -> bool:
    fg = by_index[face_idx]
    if fg.surface_type not in WALL_SURF_TYPES:
        return False
    occ = occ_map.get(face_idx) if occ_map else None
    info = _wall_interior_and_axis(fg, occ)
    if info is None:
        return False
    _, _, diameter = info
    for nb in graph.neighbors.get(face_idx, ()):
        if graph.edge_kind(face_idx, nb) not in ("concave", "smooth"):
            continue
        nfg = by_index[nb]
        if nfg.surface_type not in WALL_SURF_TYPES:
            continue
        ninfo = _wall_interior_and_axis(nfg, occ_map.get(nb) if occ_map else None)
        if ninfo is None:
            continue
        d_a = diameter or 0.0
        d_b = ninfo[2] or 0.0
        fam = _bore_family(graph)
        # Deep primary bore concave/smooth-paired with its (largest) duplicate.
        if (fam.is_deep_primary(d_a) and fam.is_duplicate(d_b, 0)) or (
            fam.is_duplicate(d_a, 0) and fam.is_deep_primary(d_b)
        ):
            return True
    return False


def _composition_tier(
    face_idx: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    cfg: LobeTierConfig,
) -> Literal["open", "closed", "either"]:
    fg = by_index[face_idx]
    st = fg.surface_type
    if st == "sphere":
        return "closed"
    if st in WALL_SURF_TYPES:
        occ = occ_map.get(face_idx) if occ_map else None
        info = _wall_interior_and_axis(fg, occ)
        d = None if info is None else info[2]
        if _opening_blend(fg, occ, cfg):
            return "open"
        if _wall_band(fg, occ, cfg):
            fam = _bore_family(graph)
            if fam.is_mouth_primary(d):
                return "open"
            if _deep_wall_pair_member(face_idx, by_index, occ_map, graph, cfg):
                return "closed"
            if fam.is_deep_primary(d):
                return "closed"
            return "either"
    return "either"


def _adjacent_shallow_bore_cylinder(
    face_idx: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    cfg: LobeTierConfig,
) -> bool:
    """True when a duplicate bore is the shallow one (concave to the mouth primary)."""
    fg = by_index[face_idx]
    if fg.surface_type not in WALL_SURF_TYPES:
        return False
    fam = _bore_family(graph)
    occ = occ_map.get(face_idx) if occ_map else None
    info = _wall_interior_and_axis(fg, occ)
    if info is None or not fam.is_duplicate(info[2], 0):
        return False
    for nb in graph.neighbors.get(face_idx, ()):
        if graph.edge_kind(face_idx, nb) != "concave":
            continue
        nfg = by_index[nb]
        if nfg.surface_type not in WALL_SURF_TYPES:
            continue
        ninfo = _wall_interior_and_axis(nfg, occ_map.get(nb) if occ_map else None)
        if ninfo and fam.is_mouth_primary(ninfo[2]):
            return True
    return False


def _duplicate_bore_transition_cylinder(
    face_idx: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    cfg: LobeTierConfig,
) -> bool:
    """Deep-primary barrel linking the duplicate bores above the mouth step."""
    fg = by_index[face_idx]
    if fg.surface_type not in WALL_SURF_TYPES:
        return False
    occ = occ_map.get(face_idx) if occ_map else None
    info = _wall_interior_and_axis(fg, occ)
    return info is not None and _bore_family(graph).is_deep_primary(info[2])


def _links_duplicate_bore_transition(
    face_idx: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    cfg: LobeTierConfig,
    *,
    edge_kinds: frozenset[str] = frozenset({"convex", "concave"}),
) -> bool:
    for nb in graph.neighbors.get(face_idx, ()):
        if graph.edge_kind(face_idx, nb) not in edge_kinds:
            continue
        if _duplicate_bore_transition_cylinder(nb, by_index, occ_map, graph, cfg):
            return True
    return False


def _closed_tier_opening_extension_wall(
    face_idx: int,
    mouth_step: int,
    mouth_boundary_tol_mm: float,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> bool:
    """Duplicate-bore wall above the mouth step that Toolpath counts as closed-tier pocket."""
    fg = by_index[face_idx]
    if fg.surface_type not in WALL_SURF_TYPES:
        return False
    occ = occ_map.get(face_idx) if occ_map else None
    if not _wall_band(fg, occ, cfg):
        return False
    if _deep_wall_pair_member(face_idx, by_index, occ_map, graph, cfg):
        return False
    if _adjacent_shallow_bore_cylinder(face_idx, by_index, occ_map, graph, cfg):
        return False
    if not _opening_side_lobe_wall(
        face_idx, mouth_step, mouth_boundary_tol_mm,
        by_index, occ_map, graph, opening_axis, cfg,
    ):
        return False
    info = _wall_interior_and_axis(fg, occ)
    if info is None:
        return False
    diameter = info[2]
    fam = _bore_family(graph)
    # Largest duplicate links via convex only; smaller duplicate via convex+concave.
    if fam.is_duplicate(diameter, 0):
        return _links_duplicate_bore_transition(
            face_idx, by_index, occ_map, graph, cfg, edge_kinds=frozenset({"convex"}),
        )
    if fam.is_duplicate(diameter, 1):
        return _links_duplicate_bore_transition(
            face_idx, by_index, occ_map, graph, cfg,
        )
    return False


def _torus_touches_deep_step(
    face_idx: int,
    graph: FaceGraph,
    deep_step_faces: set[int],
) -> bool:
    for nb in graph.neighbors.get(face_idx, ()):
        if nb not in deep_step_faces:
            continue
        if graph.edge_kind(face_idx, nb) in ("concave", "smooth"):
            return True
    return False


def _duplicate_floor_sphere(
    face_idx: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    growing: set[int],
) -> bool:
    fg = by_index[face_idx]
    if fg.surface_type != "sphere":
        return False
    for nb in graph.neighbors.get(face_idx, ()):
        if graph.edge_kind(face_idx, nb) != "concave":
            continue
        if by_index[nb].surface_type == "sphere" and nb in growing:
            return True
    return False


def _convex_only_to_open_bore(
    face_idx: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    growing: set[int],
    cfg: LobeTierConfig,
) -> bool:
    """True when a fillet torus only reaches the local duplicate bore via convex edges."""
    fam = _bore_family(graph)
    for nb in graph.neighbors.get(face_idx, ()):
        if nb not in growing:
            continue
        nfg = by_index[nb]
        if nfg.surface_type not in WALL_SURF_TYPES:
            continue
        ninfo = _wall_interior_and_axis(nfg, occ_map.get(nb) if occ_map else None)
        if ninfo is None or not fam.is_duplicate(ninfo[2], 0):
            continue
        kind = graph.edge_kind(face_idx, nb)
        if kind == "convex":
            return True
        if kind == "smooth":
            return False
    return False


def _resolve_deep_step_reference(
    deep_step_faces: set[int],
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
) -> int:
    """Deep-most step anchor (minimum axial toward pocket interior along opening axis)."""
    if not deep_step_faces:
        raise LobeTierBoundaryError(
            "lobe tier split: empty deep_step_faces — cannot compute mouth/deep span",
        )
    return min(
        deep_step_faces,
        key=lambda f: _axial_y(by_index, f, opening_axis),
    )


def _opening_side_lobe_wall(
    face_idx: int,
    mouth_step: int,
    mouth_boundary_tol_mm: float,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> bool:
    """Interior wall-band cylinder on the opening side of the detected mouth step."""
    fg = by_index[face_idx]
    if fg.surface_type not in WALL_SURF_TYPES:
        return False
    occ = occ_map.get(face_idx) if occ_map else None
    if not _wall_band(fg, occ, cfg):
        return False
    if _deep_wall_pair_member(face_idx, by_index, occ_map, graph, cfg):
        return False
    if _adjacent_shallow_bore_cylinder(face_idx, by_index, occ_map, graph, cfg):
        return False
    mouth_ax = _axial_y(by_index, mouth_step, opening_axis)
    face_ax = _axial_y(by_index, face_idx, opening_axis)
    return face_ax >= mouth_ax - mouth_boundary_tol_mm


def _deep_side_lobe_wall(
    face_idx: int,
    mouth_step: int,
    mouth_boundary_tol_mm: float,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> bool:
    """Interior wall-band cylinder on the deep side of the detected mouth step."""
    fg = by_index[face_idx]
    if fg.surface_type not in WALL_SURF_TYPES:
        return False
    occ = occ_map.get(face_idx) if occ_map else None
    if not _wall_band(fg, occ, cfg):
        return False
    if _deep_wall_pair_member(face_idx, by_index, occ_map, graph, cfg):
        return False
    mouth_ax = _axial_y(by_index, mouth_step, opening_axis)
    face_ax = _axial_y(by_index, face_idx, opening_axis)
    return face_ax < mouth_ax - mouth_boundary_tol_mm


def _tolerance_governed_for_boundary(
    face_idx: int,
    mouth_axial: float,
    deep_axial: float,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> bool:
    """Faces near the mouth step whose tier the boundary tolerance governs."""
    if not _lobe_traversable(face_idx, by_index, occ_map, opening_axis, cfg):
        return False
    if _is_exterior_contour_fillet(face_idx, graph, by_index):
        return False
    ax = _axial_y(by_index, face_idx, opening_axis)
    return _in_pocket_axial_band(ax, mouth_axial, deep_axial)


def _mouth_boundary_clearance_mm(
    pool: set[int],
    mouth_step: int,
    deep_step: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> float:
    """Minimum axial clearance from mouth step to nearest opening-side wall or deep-side feature."""
    mouth_ax = _axial_y(by_index, mouth_step, opening_axis)
    deep_ax = _axial_y(by_index, deep_step, opening_axis)

    gap_above: float | None = None
    for fid in pool:
        if not _opening_side_lobe_wall(
            fid, mouth_step, 0.0, by_index, occ_map, graph, opening_axis, cfg,
        ):
            continue
        ax = _axial_y(by_index, fid, opening_axis)
        if ax < mouth_ax:
            continue
        gap = ax - mouth_ax
        gap_above = gap if gap_above is None else min(gap_above, gap)

    gap_below: float | None = None
    for fid in by_index:
        if not _tolerance_governed_for_boundary(
            fid, mouth_ax, deep_ax, by_index, occ_map, graph, opening_axis, cfg,
        ):
            continue
        ax = _axial_y(by_index, fid, opening_axis)
        if ax >= mouth_ax:
            continue
        gap = mouth_ax - ax
        gap_below = gap if gap_below is None else min(gap_below, gap)

    if gap_above is None and gap_below is None:
        raise LobeTierBoundaryError(
            f"lobe tier split: no measurable clearance around mouth step {mouth_step}",
        )
    return min(g for g in (gap_above, gap_below) if g is not None)


def _mouth_tier_boundary_tol_mm(
    pool: set[int],
    mouth_step: int,
    deep_step: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> float:
    _ensure_bore_family(graph, by_index, occ_map, opening_axis, cfg)
    clearance = _mouth_boundary_clearance_mm(
        pool, mouth_step, deep_step, by_index, occ_map, graph, opening_axis, cfg,
    )
    if clearance < 2.0 * cfg.mouth_numeric_floor_mm:
        raise LobeTierBoundaryError(
            f"lobe tier split: clearance {clearance:.4f} mm too tight to tier-split "
            f"(minimum {2.0 * cfg.mouth_numeric_floor_mm:.4f} mm)",
        )
    mouth_ax = _axial_y(by_index, mouth_step, opening_axis)
    deep_ax = _axial_y(by_index, deep_step, opening_axis)
    span = abs(mouth_ax - deep_ax)
    span_tol = cfg.mouth_tier_span_fraction * span
    tol_candidate = min(span_tol, 0.5 * clearance - TOLERANCE_EPSILON_MM)
    if tol_candidate < cfg.mouth_tolerance_floor_mm:
        raise LobeTierBoundaryError(
            f"lobe tier split: mouth boundary tolerance {tol_candidate:.4f} mm below "
            f"measurement precision floor {cfg.mouth_tolerance_floor_mm:.4f} mm "
            f"(clearance={clearance:.4f} mm span={span:.4f} mm)",
        )
    return tol_candidate


def _axial_side_of_mouth(
    face_idx: int,
    mouth_step: int,
    mouth_boundary_tol_mm: float,
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
) -> Literal["open", "closed"]:
    """Opening vs deep side of mouth step for non-wall faces (fillet/sphere orphans)."""
    mouth_ax = _axial_y(by_index, mouth_step, opening_axis)
    face_ax = _axial_y(by_index, face_idx, opening_axis)
    if face_ax >= mouth_ax - mouth_boundary_tol_mm:
        return "open"
    return "closed"


def _skip_closed_tier_region_grow(
    face_idx: int,
    mouth_step: int,
    mouth_boundary_tol_mm: float,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> bool:
    """True when closed-tier grow must not claim an opening-side pool face."""
    if _axial_side_of_mouth(
        face_idx, mouth_step, mouth_boundary_tol_mm, by_index, opening_axis,
    ) != "open":
        return False
    if _closed_tier_opening_extension_wall(
        face_idx, mouth_step, mouth_boundary_tol_mm,
        by_index, occ_map, graph, opening_axis, cfg,
    ):
        return False
    fg = by_index[face_idx]
    if fg.surface_type in WALL_SURF_TYPES:
        return not _deep_side_lobe_wall(
            face_idx, mouth_step, mouth_boundary_tol_mm,
            by_index, occ_map, graph, opening_axis, cfg,
        )
    if fg.surface_type in ("sphere", "bspline") or fg.surface_type in FILLET_TYPES:
        return True
    return False


def _region_grow_tier(
    seeds: set[int],
    pool: set[int],
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
    *,
    tier_hint: Literal["open", "closed"],
    mouth_step: int,
    mouth_boundary_tol_mm: float,
    deep_step_faces: set[int] | None = None,
) -> set[int]:
    growing = set(seeds) & pool
    q: deque[int] = deque(sorted(growing))
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
            if tier_hint == "closed" and vfg.surface_type in FILLET_TYPES:
                if _fillet_above_mouth_step(v, mouth_step, by_index, opening_axis):
                    continue
                if _fillet_has_concave_cone_seam(v, graph, by_index):
                    continue
            if kind == "convex" and tier_hint == "closed":
                if vfg.surface_type in WALL_SURF_TYPES:
                    continue
                if (
                    vfg.surface_type == "sphere"
                    and ufg.surface_type in WALL_SURF_TYPES
                ):
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
                if (
                    ufg.surface_type in FILLET_TYPES
                    and vfg.surface_type in WALL_SURF_TYPES
                ):
                    vinfo = _wall_interior_and_axis(vfg, occ_map.get(v) if occ_map else None)
                    if vinfo and _bore_family(graph).is_duplicate(vinfo[2], 0):
                        continue
            if kind == "concave" and tier_hint == "open":
                if (
                    ufg.surface_type in WALL_SURF_TYPES
                    and vfg.surface_type in WALL_SURF_TYPES
                ):
                    uinfo = _wall_interior_and_axis(ufg, occ_map.get(u) if occ_map else None)
                    vinfo = _wall_interior_and_axis(vfg, occ_map.get(v) if occ_map else None)
                    if uinfo and vinfo:
                        ud = uinfo[2] or 0.0
                        vd = vinfo[2] or 0.0
                        fam = _bore_family(graph)
                        if fam.is_mouth_primary(ud) and fam.is_duplicate(vd, 0):
                            continue
            if kind == "convex" and tier_hint == "open":
                if ufg.surface_type == "plane" and _is_axial_plane(
                    ufg, opening_axis, _pocket_cfg(cfg),
                ):
                    if comp == "closed":
                        continue
            if not _lobe_edge_ok(graph, u, v, growing):
                continue
            if comp == "open" and tier_hint == "closed":
                continue
            if comp == "closed" and tier_hint == "open":
                continue
            if tier_hint == "open" and _adjacent_shallow_bore_cylinder(
                v, by_index, occ_map, graph, cfg,
            ):
                continue
            if tier_hint == "open" and _closed_tier_opening_extension_wall(
                v, mouth_step, mouth_boundary_tol_mm,
                by_index, occ_map, graph, opening_axis, cfg,
            ):
                continue
            if tier_hint == "open" and vfg.surface_type in WALL_SURF_TYPES:
                if comp == "either" and not _opening_side_lobe_wall(
                    v, mouth_step, mouth_boundary_tol_mm,
                    by_index, occ_map, graph, opening_axis, cfg,
                ):
                    continue
            if tier_hint == "open" and vfg.surface_type in FILLET_TYPES:
                if _fillet_has_concave_cone_seam(v, graph, by_index):
                    continue
                if deep_step_faces and _torus_touches_deep_step(
                    v, graph, deep_step_faces,
                ):
                    continue
                if _convex_only_to_open_bore(
                    v, by_index, occ_map, graph, growing, cfg,
                ):
                    continue
                if any(
                    _adjacent_shallow_bore_cylinder(nb, by_index, occ_map, graph, cfg)
                    for nb in graph.neighbors.get(v, ())
                    if graph.edge_kind(v, nb) == "convex"
                ):
                    continue
            if tier_hint == "closed" and vfg.surface_type == "sphere":
                if (
                    ufg.surface_type in FILLET_TYPES
                    and _duplicate_floor_sphere(v, graph, by_index, growing)
                ):
                    continue
            if tier_hint == "closed" and _skip_closed_tier_region_grow(
                v, mouth_step, mouth_boundary_tol_mm,
                by_index, occ_map, graph, opening_axis, cfg,
            ):
                continue
            growing.add(v)
            q.append(v)
    return growing


def _sphere_radius_mm(
    face_idx: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
) -> float | None:
    fg = by_index[face_idx]
    if fg.surface_type != "sphere":
        return None
    if fg.radius is not None:
        return float(fg.radius)
    if occ_map is None:
        return None
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Sphere

    surf = BRepAdaptor_Surface(occ_map[face_idx], True)
    if surf.GetType() == GeomAbs_Sphere:
        return float(surf.Sphere().Radius())
    return None


def _torus_minor_radius_mm(fg: Any) -> float | None:
    if fg.surface_type != "torus" or fg.torus_minor_r is None:
        return None
    return float(fg.torus_minor_r)


def _relative_radius_match(a: float, b: float, rel_tol: float) -> bool:
    if a <= 0.0 or b <= 0.0:
        return False
    return abs(a - b) / max(a, b) <= rel_tol


def _annex_fillet_sphere_caps(
    annex: set[int],
    pool: set[int],
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    cfg: LobeTierConfig,
    *,
    walls: set[int],
) -> tuple[set[int], list[dict[str, Any]]]:
    """Annex sphere caps that convex-adjoin an already-annexed torus with matching minor R."""
    if not cfg.annex_fillet_sphere_caps:
        return annex, []

    rel_tol = cfg.fillet_sphere_cap_radius_rel_tol
    out = set(annex)
    annexed_tori = {
        f for f in annex if by_index[f].surface_type == "torus"
    }
    logs: list[dict[str, Any]] = []

    for f in sorted(pool - out):
        if by_index[f].surface_type != "sphere":
            continue
        sphere_r = _sphere_radius_mm(f, by_index, occ_map)
        if sphere_r is None:
            continue
        sphere_area = float(by_index[f].area)
        matched_torus: int | None = None
        matched_torus_r: float | None = None
        matched_edge_kind: str | None = None
        for nb in graph.neighbors.get(f, ()):
            edge_kind = graph.edge_kind(f, nb)
            if edge_kind not in FILLET_SPHERE_CAP_EDGE_KINDS:
                continue
            if nb not in annexed_tori:
                continue
            borders = _torus_borders_pocket_walls(nb, graph, by_index, walls)
            if not borders and edge_kind != "concave":
                if edge_kind != "convex":
                    continue
                torus_area_probe = float(by_index[nb].area)
                if (
                    cfg.fillet_sphere_cap_max_torus_area_ratio is None
                    or sphere_area
                    > cfg.fillet_sphere_cap_max_torus_area_ratio * torus_area_probe
                ):
                    continue
            torus_r = _torus_minor_radius_mm(by_index[nb])
            if torus_r is None:
                continue
            if not _relative_radius_match(sphere_r, torus_r, rel_tol):
                continue
            torus_area = float(by_index[nb].area)
            if (
                cfg.fillet_sphere_cap_max_torus_area_ratio is not None
                and sphere_area
                > cfg.fillet_sphere_cap_max_torus_area_ratio * torus_area
            ):
                continue
            matched_torus = nb
            matched_torus_r = torus_r
            matched_edge_kind = edge_kind
            break
        if matched_torus is None or matched_torus_r is None or matched_edge_kind is None:
            continue
        torus_area = float(by_index[matched_torus].area)
        area_ratio = sphere_area / torus_area if torus_area > 0.0 else None
        out.add(f)
        logs.append({
            "sphere_face": f,
            "torus_face": matched_torus,
            "sphere_radius_mm": round(sphere_r, 6),
            "torus_minor_radius_mm": round(matched_torus_r, 6),
            "radius_rel_tol": rel_tol,
            "sphere_area_mm2": round(sphere_area, 4),
            "torus_area_mm2": round(torus_area, 4),
            "cap_torus_area_ratio": round(area_ratio, 4) if area_ratio is not None else None,
            "max_torus_area_ratio": cfg.fillet_sphere_cap_max_torus_area_ratio,
            "edge_kind": matched_edge_kind,
            "torus_borders_walls": _torus_borders_pocket_walls(
                matched_torus, graph, by_index, walls,
            ),
        })

    if logs:
        logger.info(
            "lobe tier split: annexed %d fillet sphere cap(s): %s",
            len(logs),
            logs,
        )
    return out, logs


def _convex_cap_spheres(
    face_idx: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    pool: set[int],
) -> list[int]:
    return [
        nb for nb in graph.neighbors.get(face_idx, ())
        if nb in pool
        and by_index[nb].surface_type == "sphere"
        and graph.edge_kind(face_idx, nb) == SCULPT_CAP_BSPLINE_EDGE_KIND
    ]


def _crown_sculpt_cap_sphere(
    cap_face: int,
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> bool:
    uv = _project_uv(np.asarray(by_index[cap_face].centroid, dtype=np.float64), opening_axis)
    return float(uv[1]) > cfg.sculpt_cap_crown_uv_v_mm


def _sculpt_cap_bspline_tier_open(
    face_idx: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    pool: set[int],
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
    *,
    concave_annexed_cap_spheres: set[int],
    open_tier_faces: set[int] | None = None,
) -> bool:
    cap_nbs = _convex_cap_spheres(face_idx, graph, by_index, pool)
    if not cap_nbs:
        return False
    if any(nb in concave_annexed_cap_spheres for nb in cap_nbs):
        return True
    if open_tier_faces and any(nb in open_tier_faces for nb in cap_nbs):
        # Severed-bridge sculpt cap: convex bridge sphere already in open tier
        # (e.g. fillet_sphere_cap annex) but bspline stranded in closed tier.
        return True
    return any(
        not _crown_sculpt_cap_sphere(nb, by_index, opening_axis, cfg)
        for nb in cap_nbs
    )


def _annex_sculpt_cap_bsplines(
    annex: set[int],
    pool: set[int],
    graph: FaceGraph,
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
    *,
    mouth_step: int,
    mouth_boundary_tol_mm: float,
    concave_annexed_cap_spheres: set[int],
) -> set[int]:
    """Grow open tier into mouth-band bspline sculpt rings around non-crown cap spheres."""
    out = set(annex)
    for fid in sorted(pool - out):
        if by_index[fid].surface_type != "bspline":
            continue
        if _axial_side_of_mouth(
            fid, mouth_step, mouth_boundary_tol_mm, by_index, opening_axis,
        ) != "open":
            continue
        if not _sculpt_cap_bspline_tier_open(
            fid, graph, by_index, pool, opening_axis, cfg,
            concave_annexed_cap_spheres=concave_annexed_cap_spheres,
            open_tier_faces=out,
        ):
            continue
        out.add(fid)
    return out


def _annex_fillets(
    tier_faces: set[int],
    pool: set[int],
    graph: FaceGraph,
    by_index: dict[int, Any],
    *,
    occ_map: dict[int, Any] | None = None,
    cfg: LobeTierConfig | None = None,
    skip: Callable[[int], bool] | None = None,
) -> tuple[set[int], list[dict[str, Any]]]:
    walls = {
        i for i in tier_faces if by_index[i].surface_type in WALL_SURF_TYPES
    }
    annex = set(tier_faces)
    for f in sorted(pool - annex):
        if skip is not None and skip(f):
            continue
        if by_index[f].surface_type not in FILLET_TYPES:
            continue
        if _torus_borders_pocket_walls(f, graph, by_index, walls):
            annex.add(f)
    if cfg is not None:
        annex, logs = _annex_fillet_sphere_caps(
            annex, pool, graph, by_index, occ_map, cfg, walls=walls,
        )
        return annex, logs
    return annex, []


def _fillet_has_concave_cone_seam(
    face_idx: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
) -> bool:
    for nb in graph.neighbors.get(face_idx, ()):
        if graph.edge_kind(face_idx, nb) != "concave":
            continue
        if by_index[nb].surface_type == "cone":
            return True
    return False


def _is_exterior_contour_fillet(
    face_idx: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
) -> bool:
    """Opening-side contour blend — fillet with a concave seam to an exterior cone."""
    if by_index[face_idx].surface_type not in FILLET_TYPES:
        return False
    return _fillet_has_concave_cone_seam(face_idx, graph, by_index)


def _fillet_has_convex_shallow_bore(
    face_idx: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    cfg: LobeTierConfig,
) -> bool:
    for nb in graph.neighbors.get(face_idx, ()):
        if graph.edge_kind(face_idx, nb) != "convex":
            continue
        if _adjacent_shallow_bore_cylinder(nb, by_index, occ_map, graph, cfg):
            return True
    return False


def _is_convex_duplicate_open_fillet(
    face_idx: int,
    tier_faces: set[int],
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    cfg: LobeTierConfig,
) -> bool:
    """Drop open fillets that convex-only mirror another fillet's smooth bore link."""
    for nb in graph.neighbors.get(face_idx, ()):
        if graph.edge_kind(face_idx, nb) != "concave":
            continue
        if by_index[nb].surface_type == "sphere" and nb in tier_faces:
            return False
    for nb in graph.neighbors.get(face_idx, ()):
        if graph.edge_kind(face_idx, nb) != "convex":
            continue
        nfg = by_index[nb]
        if nfg.surface_type not in WALL_SURF_TYPES:
            continue
        ninfo = _wall_interior_and_axis(nfg, occ_map.get(nb) if occ_map else None)
        if ninfo is None or not _bore_family(graph).is_duplicate(ninfo[2], 0):
            continue
        for other in tier_faces:
            if other == face_idx or by_index[other].surface_type not in FILLET_TYPES:
                continue
            if graph.edge_kind(other, nb) == "smooth":
                return True
    return False


def _open_fillet_should_drop(
    face_idx: int,
    tier_faces: set[int],
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    cfg: LobeTierConfig,
    deep_step_faces: set[int],
) -> bool:
    fg = by_index[face_idx]
    if fg.surface_type not in FILLET_TYPES:
        return False
    if _fillet_has_concave_cone_seam(face_idx, graph, by_index):
        return True
    if _fillet_has_convex_shallow_bore(face_idx, graph, by_index, occ_map, cfg):
        return True
    if deep_step_faces and _torus_touches_deep_step(face_idx, graph, deep_step_faces):
        return True
    if _is_convex_duplicate_open_fillet(
        face_idx, tier_faces, graph, by_index, occ_map, cfg,
    ):
        return True
    return False


def _prune_open_tier_fillets(
    open_faces: set[int],
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    cfg: LobeTierConfig,
    deep_step_faces: set[int],
) -> set[int]:
    out = set(open_faces)
    changed = True
    while changed:
        changed = False
        for f in sorted(out):
            if f not in out:
                continue
            if _open_fillet_should_drop(
                f, out, graph, by_index, occ_map, cfg, deep_step_faces,
            ):
                out.discard(f)
                changed = True
    return out


def _prune_deep_step_interface_tori(
    tier_faces: set[int],
    deep_step: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
) -> set[int]:
    """Drop shallow interface tori sandwiched concave between deep step and floor fillet."""
    out = set(tier_faces)
    deep_steps = {deep_step}

    def _concave_to_deep(f: int) -> bool:
        for nb in graph.neighbors.get(f, ()):
            if nb not in deep_steps:
                continue
            if graph.edge_kind(f, nb) == "concave":
                return True
        return False

    for f in sorted(
        out,
        key=lambda i: -_axial_y(by_index, i, opening_axis),
    ):
        if f not in out or by_index[f].surface_type not in FILLET_TYPES:
            continue
        if not _concave_to_deep(f):
            continue
        ax_f = _axial_y(by_index, f, opening_axis)
        for nb in graph.neighbors.get(f, ()):
            if nb not in out or by_index[nb].surface_type not in FILLET_TYPES:
                continue
            if graph.edge_kind(f, nb) != "concave":
                continue
            if not _concave_to_deep(nb) and nb != f:
                out.discard(f)
                break
            ax_nb = _axial_y(by_index, nb, opening_axis)
            if ax_f > ax_nb:
                if _concave_to_deep(nb):
                    continue
                out.discard(f)
                break
    return out


def _fillet_above_mouth_step(
    face_idx: int,
    mouth_step: int,
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    *,
    tol_mm: float = 0.1,
) -> bool:
    fg = by_index[face_idx]
    if fg.surface_type not in FILLET_TYPES:
        return False
    mouth_ax = _axial_y(by_index, mouth_step, opening_axis)
    return _axial_y(by_index, face_idx, opening_axis) > mouth_ax + tol_mm


def _prune_closed_tier_fillets(
    closed_faces: set[int],
    mouth_step: int,
    deep_step: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    cfg: LobeTierConfig,
    opening_axis: np.ndarray,
) -> set[int]:
    out = set(closed_faces)
    for f in sorted(out):
        if by_index[f].surface_type not in FILLET_TYPES:
            continue
        if _fillet_above_mouth_step(f, mouth_step, by_index, opening_axis):
            out.discard(f)
            continue
        if _fillet_has_concave_cone_seam(f, graph, by_index):
            out.discard(f)
            continue
        if _fillet_has_convex_shallow_bore(f, graph, by_index, occ_map, cfg):
            out.discard(f)
    out = _prune_deep_step_interface_tori(out, deep_step, graph, by_index, opening_axis)
    return out


def _resolve_overlap(
    open_faces: set[int],
    closed_faces: set[int],
    open_anchors: set[int],
    closed_anchors: set[int],
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
    *,
    mouth_step: int,
    mouth_boundary_tol_mm: float,
    deep_step_faces: set[int] | None = None,
) -> tuple[set[int], set[int]]:
    overlap = open_faces & closed_faces
    open_out = set(open_faces)
    closed_out = set(closed_faces)
    for f in sorted(overlap):
        if f in open_anchors and f not in closed_anchors:
            closed_out.discard(f)
            continue
        if f in closed_anchors and f not in open_anchors:
            open_out.discard(f)
            continue
        if (
            deep_step_faces
            and by_index[f].surface_type in FILLET_TYPES
            and _torus_touches_deep_step(f, graph, deep_step_faces)
        ):
            open_out.discard(f)
            continue
        comp = _composition_tier(f, by_index, occ_map, graph, cfg)
        if comp == "open":
            closed_out.discard(f)
        elif comp == "closed":
            open_out.discard(f)
        elif _closed_tier_opening_extension_wall(
            f, mouth_step, mouth_boundary_tol_mm,
            by_index, occ_map, graph, opening_axis, cfg,
        ):
            open_out.discard(f)
        elif _opening_side_lobe_wall(
            f, mouth_step, mouth_boundary_tol_mm,
            by_index, occ_map, graph, opening_axis, cfg,
        ):
            closed_out.discard(f)
        elif _deep_side_lobe_wall(
            f, mouth_step, mouth_boundary_tol_mm,
            by_index, occ_map, graph, opening_axis, cfg,
        ):
            open_out.discard(f)
        elif _axial_side_of_mouth(
            f, mouth_step, mouth_boundary_tol_mm, by_index, opening_axis,
        ) == "open":
            closed_out.discard(f)
        else:
            open_out.discard(f)
    return open_out, closed_out


def _prune_paired_floor_spheres(
    tier_faces: set[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    opening_axis: np.ndarray,
) -> set[int]:
    """Drop the deeper sphere when two sphere floors share a concave seam."""
    out = set(tier_faces)
    spheres = [f for f in tier_faces if by_index[f].surface_type == "sphere"]
    for s in spheres:
        for nb in graph.neighbors.get(s, ()):
            if nb not in tier_faces:
                continue
            if graph.edge_kind(s, nb) != "concave":
                continue
            if by_index[nb].surface_type != "sphere":
                continue
            ax_s = _axial_y(by_index, s, opening_axis)
            ax_nb = _axial_y(by_index, nb, opening_axis)
            if ax_s < ax_nb:
                out.discard(s)
            else:
                out.discard(nb)
    return out


def _reassign_orphan_lobe_faces(
    pool: set[int],
    open_faces: set[int],
    closed_faces: set[int],
    mouth_step: int,
    deep_step: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
    *,
    mouth_boundary_tol_mm: float,
    concave_annexed_cap_spheres: set[int] | None = None,
) -> tuple[set[int], set[int]]:
    """Re-home pool faces that tier grow/prune left unassigned.

    Exterior contour fillets stay unassigned (released to later cascade passes).
    Opening-side wall-band cylinders should already be claimed by region_grow; this
    pass remains a safety net for fillets, spheres, and edge cases.
    """
    open_out = set(open_faces)
    closed_out = set(closed_faces)
    orphans = sorted(pool - open_out - closed_out)
    if not orphans:
        return open_out, closed_out

    def _tier_walls(tier: set[int]) -> set[int]:
        return {i for i in tier if by_index[i].surface_type in WALL_SURF_TYPES}

    concave_caps = set(concave_annexed_cap_spheres or ())

    remaining: list[int] = []
    for fid in orphans:
        if _is_exterior_contour_fillet(fid, graph, by_index):
            continue
        comp = _composition_tier(fid, by_index, occ_map, graph, cfg)
        if comp == "open":
            open_out.add(fid)
        elif comp == "closed":
            closed_out.add(fid)
        elif _closed_tier_opening_extension_wall(
            fid, mouth_step, mouth_boundary_tol_mm,
            by_index, occ_map, graph, opening_axis, cfg,
        ):
            closed_out.add(fid)
        elif _opening_side_lobe_wall(
            fid, mouth_step, mouth_boundary_tol_mm,
            by_index, occ_map, graph, opening_axis, cfg,
        ):
            open_out.add(fid)
        elif _deep_side_lobe_wall(
            fid, mouth_step, mouth_boundary_tol_mm,
            by_index, occ_map, graph, opening_axis, cfg,
        ):
            closed_out.add(fid)
        else:
            remaining.append(fid)

    still: list[int] = []
    for fid in remaining:
        if by_index[fid].surface_type in FILLET_TYPES:
            if _axial_side_of_mouth(
                fid, mouth_step, mouth_boundary_tol_mm, by_index, opening_axis,
            ) == "open":
                open_out.add(fid)
                continue
            borders_open = _torus_borders_pocket_walls(
                fid, graph, by_index, _tier_walls(open_out),
            )
            borders_closed = _torus_borders_pocket_walls(
                fid, graph, by_index, _tier_walls(closed_out),
            )
            if borders_open and not borders_closed:
                open_out.add(fid)
                continue
            if borders_closed and not borders_open:
                closed_out.add(fid)
                continue
        still.append(fid)

    for fid in still:
        if by_index[fid].surface_type == "bspline":
            if _sculpt_cap_bspline_tier_open(
                fid, graph, by_index, pool, opening_axis, cfg,
                concave_annexed_cap_spheres=concave_caps,
                open_tier_faces=open_out,
            ):
                open_out.add(fid)
            else:
                closed_out.add(fid)
            continue
        if _axial_side_of_mouth(
            fid, mouth_step, mouth_boundary_tol_mm, by_index, opening_axis,
        ) == "closed":
            closed_out.add(fid)
        else:
            open_out.add(fid)

    reassigned = [f for f in orphans if f in open_out or f in closed_out]
    if reassigned:
        logger.info(
            "lobe orphan reassignment: re-homed %d pool orphan(s) "
            "(open=%d closed=%d): %s",
            len(reassigned),
            sum(1 for f in reassigned if f in open_out),
            sum(1 for f in reassigned if f in closed_out),
            sorted(reassigned),
        )
    return open_out, closed_out


def _migrate_sculpt_cap_bsplines_follow_open_bridge(
    pool: set[int],
    open_faces: set[int],
    closed_faces: set[int],
    graph: FaceGraph,
    by_index: dict[int, Any],
) -> tuple[set[int], set[int]]:
    """Move closed-tier sculpt-cap bsplines that convex-bridge into open tier."""
    open_out = set(open_faces)
    closed_out = set(closed_faces)
    migrated: list[int] = []
    for fid in sorted(closed_out):
        if by_index[fid].surface_type != "bspline":
            continue
        cap_nbs = _convex_cap_spheres(fid, graph, by_index, pool)
        if not cap_nbs or not any(nb in open_out for nb in cap_nbs):
            continue
        closed_out.discard(fid)
        open_out.add(fid)
        migrated.append(fid)
    if migrated:
        logger.info(
            "lobe tier split: migrated %d sculpt-cap bspline(s) to open tier "
            "(convex bridge into open): %s",
            len(migrated),
            migrated,
        )
    return open_out, closed_out


def _migrate_closed_tier_opening_extension_walls(
    pool: set[int],
    open_faces: set[int],
    closed_faces: set[int],
    mouth_step: int,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    graph: FaceGraph,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
    *,
    mouth_boundary_tol_mm: float,
) -> tuple[set[int], set[int]]:
    """Move duplicate-bore opening-extension walls from open tier to closed tier."""
    open_out = set(open_faces)
    closed_out = set(closed_faces)
    migrated: list[int] = []
    for fid in sorted(pool):
        if not _closed_tier_opening_extension_wall(
            fid, mouth_step, mouth_boundary_tol_mm,
            by_index, occ_map, graph, opening_axis, cfg,
        ):
            continue
        if fid in open_out:
            open_out.discard(fid)
        closed_out.add(fid)
        migrated.append(fid)
    if migrated:
        logger.info(
            "lobe tier split: migrated %d opening-extension wall(s) to closed tier: %s",
            len(migrated),
            migrated,
        )
    return open_out, closed_out


def split_lobe_pool(
    pool: set[int],
    mouth_step: int,
    deep_step: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
    cfg: LobeTierConfig,
) -> tuple[set[int], set[int], list[dict[str, Any]]]:
    open_seeds, closed_seeds = {mouth_step}, {deep_step}
    deep_steps = {deep_step}
    deep_ref = _resolve_deep_step_reference(deep_steps, by_index, opening_axis)
    mouth_boundary_tol = _mouth_tier_boundary_tol_mm(
        pool, mouth_step, deep_ref, by_index, occ_map, graph, opening_axis, cfg,
    )
    open_faces = _region_grow_tier(
        open_seeds, pool, graph, by_index, occ_map, opening_axis, cfg,
        tier_hint="open", mouth_step=mouth_step,
        mouth_boundary_tol_mm=mouth_boundary_tol, deep_step_faces=deep_steps,
    )
    closed_faces = _region_grow_tier(
        closed_seeds, pool, graph, by_index, occ_map, opening_axis, cfg,
        tier_hint="closed", mouth_step=mouth_step,
        mouth_boundary_tol_mm=mouth_boundary_tol, deep_step_faces=deep_steps,
    )
    open_faces, sphere_cap_logs = _annex_fillets(
        open_faces, pool, graph, by_index,
        occ_map=occ_map,
        cfg=cfg,
        skip=lambda f: _open_fillet_should_drop(
            f, open_faces, graph, by_index, occ_map, cfg, deep_steps,
        ),
    )
    concave_caps = {
        int(entry["sphere_face"])
        for entry in sphere_cap_logs
        if entry.get("edge_kind") == "concave"
    }
    open_faces = _annex_sculpt_cap_bsplines(
        open_faces, pool, graph, by_index, opening_axis, cfg,
        mouth_step=mouth_step,
        mouth_boundary_tol_mm=mouth_boundary_tol,
        concave_annexed_cap_spheres=concave_caps,
    )
    closed_faces, _ = _annex_fillets(
        closed_faces, pool, graph, by_index,
        occ_map=occ_map,
        cfg=cfg,
        skip=lambda f: _axial_side_of_mouth(
            f, mouth_step, mouth_boundary_tol, by_index, opening_axis,
        ) == "open",
    )
    if sphere_cap_logs:
        closed_faces -= {entry["sphere_face"] for entry in sphere_cap_logs}
    open_faces = _prune_open_tier_fillets(
        open_faces, graph, by_index, occ_map, cfg, deep_steps,
    )
    closed_faces = _prune_paired_floor_spheres(
        closed_faces, by_index, graph, opening_axis,
    )
    closed_faces = _prune_closed_tier_fillets(
        closed_faces, mouth_step, deep_step, graph, by_index, occ_map, cfg, opening_axis,
    )
    open_faces, closed_faces = _resolve_overlap(
        open_faces, closed_faces, open_seeds, closed_seeds,
        by_index, occ_map, graph, opening_axis, cfg,
        mouth_step=mouth_step,
        mouth_boundary_tol_mm=mouth_boundary_tol,
        deep_step_faces=deep_steps,
    )
    open_faces, closed_faces = _migrate_closed_tier_opening_extension_walls(
        pool, open_faces, closed_faces, mouth_step,
        by_index, occ_map, graph, opening_axis, cfg,
        mouth_boundary_tol_mm=mouth_boundary_tol,
    )
    open_faces, closed_faces = _reassign_orphan_lobe_faces(
        pool, open_faces, closed_faces, mouth_step, deep_step,
        by_index, occ_map, graph, opening_axis, cfg,
        mouth_boundary_tol_mm=mouth_boundary_tol,
        concave_annexed_cap_spheres=concave_caps,
    )
    open_faces, closed_faces = _migrate_sculpt_cap_bsplines_follow_open_bridge(
        pool, open_faces, closed_faces, graph, by_index,
    )
    return open_faces, closed_faces, sphere_cap_logs


def detect_filleted_lobe_tiers(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    occ_faces: Sequence[Any] | None = None,
    opening_axis: Sequence[float] | None = None,
    n_lobes_hint: int | None = None,
    config: LobeTierConfig | None = None,
) -> LobeTierDetectionResult:
    """Discover filleted lobe boundaries and open/closed tier face sets."""
    config = config or LobeTierConfig()
    n_faces = len(faces)
    by_index = {int(f.index): f for f in faces}
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)
    occ_map = None
    if occ_faces is not None:
        occ_map = {i: occ_faces[i] for i in range(n_faces)}

    if opening_axis is None:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        axis = np.asarray(opening_axis, dtype=np.float64)

    # Derive the concentric-bore role table once and stash it on the graph so the
    # downstream tier helpers can query roles instead of hard-coded diameters.
    _ensure_bore_family(graph, by_index, occ_map, axis, config)

    step_planes = _collect_axial_step_planes(by_index, axis, config)
    mouth_steps, deep_steps, bins = discover_mouth_and_deep_steps(
        step_planes, config, n_lobes_hint=n_lobes_hint,
    )
    if not mouth_steps or not deep_steps:
        logger.warning(
            "could not discover mouth/deep step tiers (bins=%s)",
            [(ax, len(ids)) for ax, ids in bins],
        )
        return LobeTierDetectionResult(
            lobes=[], mouth_step_faces=[], deep_step_faces=[],
            opening_axis=tuple(float(x) for x in axis),
            tier_axial_bins=bins, n_faces=n_faces,
        )

    pairs = pair_mouth_deep_steps(mouth_steps, deep_steps, by_index, axis)
    mouth_ax = max(_axial_y(by_index, m, axis) for m in mouth_steps)
    deep_ax = min(_axial_y(by_index, d, axis) for d in deep_steps)
    voronoi_pools = assign_lobe_faces_angular(
        pairs, by_index, occ_map, axis, config, graph,
        mouth_axial=mouth_ax, deep_axial=deep_ax,
    )
    lobes: list[FilletedLobeTier] = []

    for lid, (mouth, deep) in enumerate(pairs):
        pool = refine_pool_by_fillet_connectivity(
            voronoi_pools[lid], mouth, deep, graph, by_index, occ_map, axis, config,
        )
        open_faces, closed_faces, sphere_cap_logs = split_lobe_pool(
            pool, mouth, deep, graph, by_index, occ_map, axis, config,
        )
        uvs = np.array([
            _project_uv(np.asarray(by_index[i].centroid, dtype=np.float64), axis)
            for i in pool
        ])
        centroid_uv = tuple(float(x) for x in uvs.mean(axis=0)) if len(uvs) else (0.0, 0.0)
        lobes.append(FilletedLobeTier(
            lobe_id=lid,
            mouth_step_face=mouth,
            deep_step_face=deep,
            mouth_axial=_axial_y(by_index, mouth, axis),
            deep_axial=_axial_y(by_index, deep, axis),
            pool_faces=pool,
            open_faces=open_faces,
            closed_faces=closed_faces,
            unassigned_faces=pool - open_faces - closed_faces,
            centroid_uv=centroid_uv,
            fillet_sphere_cap_annex=sphere_cap_logs,
        ))

    return LobeTierDetectionResult(
        lobes=lobes,
        mouth_step_faces=sorted(mouth_steps),
        deep_step_faces=sorted(deep_steps),
        opening_axis=tuple(float(x) for x in axis),
        tier_axial_bins=bins,
        n_faces=n_faces,
    )
