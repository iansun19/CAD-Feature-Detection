"""
coaxial_stack_detection.py — Cascade stage 3: central coaxial hub features.

After holes, claims geometry sharing the central bore axis before flats:
  * flat             — deepest coaxial horizontal seating plane (Toolpath FACE)
  * open_pocket      — shallower stepped floor + vertical step walls
  * contour_surface  — hub step-rim torus + deep cone/torus contour blends

Toolpath opening-tier outer_fillet is deferred by the hole pass and claimed by
outer_fillet_detection (stack-boundary Pass A). Hub-perimeter torus pairs are
claimed in outer_fillet_detection Pass C using ``hub_perimeter_context`` from
this pass.

Runs before flats so deep hub seating planes are tagged for the flat pass and
not lumped into open_pocket or residual/contour.
"""
from __future__ import annotations

import argparse
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hole_detection import FaceGraph, _unit
from part_scale import resolve_scaled_mm

logger = logging.getLogger("coaxial_stack_detection")

MM_PER_IN = 25.4
WALL_TYPES = frozenset({"cylinder", "cone"})
BLEND_TYPES = frozenset({"torus"})
HUB_BLEND_TYPES = frozenset({"torus", "cone"})

# Reference-part regression (96260B_front hub) — oracles only, not algorithm input.
REFERENCE_HUB_FLAT_FACES_FRONT = (273,)
REFERENCE_OPEN_POCKET_FACES_FRONT = (277, 281, 282, 283)
REFERENCE_CONTOUR_FACES_FRONT = (274, 275, 276)

# 96260B_rear hub (symmetric geometry, different face indices).
REFERENCE_HUB_FLAT_FACES_REAR = (322,)
REFERENCE_OPEN_POCKET_FACES_REAR = (326, 330, 331, 332)
REFERENCE_CONTOUR_FACES_REAR = (323, 324, 325)


@dataclass
class CoaxialStackDetectionConfig:
    """Relational hub contract — depths/radii derived from hole stack on part."""

    min_floor_depth_mm: float = 10.0
    min_floor_area_mm2: float = 500.0
    # Radial grab margin beyond the hole-stack radius. Absolute (mm) if set;
    # when None it is derived as ``coaxial_radial_margin_frac * part scale`` so
    # the bound tracks part size instead of 96260B's ~85 mm plate radius.
    coaxial_radial_margin_mm: float | None = None
    coaxial_radial_margin_frac: float = 0.379  # 85.0 mm / 224.6 mm ref plate
    pocket_axial_band_mm: float = 5.0
    bfs_max_hops: int = 8
    horizontal_normal_tol_deg: float = 15.0
    hub_contour_radial_margin_mm: float = 8.0
    units: str = "mm"


@dataclass
class HubPerimeterFilletContext:
    """Context for outer_fillet_detection Pass C (hub-perimeter torus band)."""

    axis_point: tuple[float, float, float]
    axis_direction: tuple[float, float, float]
    open_pocket_floor_radial_mm: float
    perimeter_radial_margin_mm: float = 5.0


@dataclass
class CoaxialStackFeature:
    feature_id: int
    kind: str  # open_pocket | contour_surface
    face_indices: set[int]
    toolpath_class: str

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "toolpath_class": self.toolpath_class,
            "face_indices": sorted(self.face_indices),
            "n_faces": self.n_faces,
        }


@dataclass
class CoaxialStackDetectionResult:
    features: list[CoaxialStackFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int
    hub_flat_faces: set[int] = field(default_factory=set)
    hub_perimeter_context: HubPerimeterFilletContext | None = None
    units: str = "mm"

    def summary(self) -> str:
        kinds = ", ".join(
            f"{sum(1 for f in self.features if f.kind == k)} {k}"
            for k in ("open_pocket", "contour_surface")
        )
        flat_note = f", {len(self.hub_flat_faces)} hub flat(s) deferred to flats pass" if self.hub_flat_faces else ""
        return (
            f"{len(self.features)} coaxial hub feature(s) ({kinds}{flat_note}); "
            f"claimed {len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain for flats pass"
        )


def _radial_dist(
    centroid: Sequence[float],
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
) -> float:
    c = np.asarray(centroid, dtype=float) - axis_point
    d = _unit(axis_dir)
    return float(np.linalg.norm(c - np.dot(c, d) * d))


def _face_depth_below_top(
    face_idx: int,
    by_index: dict[int, Any],
    opening_axis: np.ndarray,
    part_axis_top: float,
) -> float:
    axis = _unit(opening_axis)
    c = np.asarray(by_index[face_idx].centroid, dtype=np.float64)
    return part_axis_top - float(np.dot(c, axis))


def _stack_axis_from_holes(hole_features: Sequence[Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (axis_point, axis_direction) from the largest-diameter hole feature."""
    best: Any | None = None
    best_d = -1.0
    for feat in hole_features:
        d = float(getattr(feat, "nominal_diameter", 0.0) or 0.0)
        if d > best_d:
            best_d = d
            best = feat
    if best is None or not getattr(best, "axis", None):
        return None
    ax = best.axis
    point = np.asarray(ax.point, dtype=float)
    direction = _unit(np.asarray(ax.direction, dtype=float))
    return point, direction


def _coaxial_pool(
    pool: set[int],
    hole_claimed: set[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    config: CoaxialStackDetectionConfig,
    radial_margin_mm: float,
) -> set[int]:
    """Faces in pool reachable from hole walls within radial bound of the stack axis."""
    hole_walls = {
        i for i in hole_claimed
        if by_index[i].surface_type in WALL_TYPES
    }
    if not hole_walls:
        return set()

    max_rad = max(
        _radial_dist(by_index[i].centroid, axis_point, axis_dir)
        for i in hole_claimed
    ) + radial_margin_mm

    frontier: set[int] = set()
    for wall in hole_walls:
        for nb in graph.neighbors.get(wall, set()):
            if nb in pool:
                frontier.add(nb)

    out: set[int] = set()
    q: deque[tuple[int, int]] = deque((i, 0) for i in frontier)
    seen = set(frontier)
    while q:
        fid, depth = q.popleft()
        if _radial_dist(by_index[fid].centroid, axis_point, axis_dir) > max_rad:
            continue
        out.add(fid)
        if depth >= config.bfs_max_hops:
            continue
        for nb in graph.neighbors.get(fid, set()):
            if nb in seen or nb not in pool:
                continue
            seen.add(nb)
            q.append((nb, depth + 1))
    return out


def _is_horizontal_plane(
    fg: Any,
    opening_axis: np.ndarray,
    tol_deg: float,
) -> bool:
    if fg.surface_type != "plane":
        return False
    normal = getattr(fg, "normal", None)
    if normal is None:
        return False
    n = _unit(np.asarray(normal, dtype=float))
    a = _unit(opening_axis)
    return abs(float(np.dot(n, a))) >= math.cos(math.radians(tol_deg))


def _depth_tiers_from_gaps(
    face_ids: Sequence[int],
    depth_fn,
) -> list[list[int]]:
    """Cluster face ids by depth using the largest natural gaps (shallow → deep)."""
    if not face_ids:
        return []
    items = sorted((depth_fn(i), i) for i in face_ids)
    if len(items) == 1:
        return [[items[0][1]]]
    gaps = [
        (items[k + 1][0] - items[k][0], k)
        for k in range(len(items) - 1)
        if items[k + 1][0] - items[k][0] > 1e-6
    ]
    if not gaps:
        return [[i for _, i in items]]
    gaps.sort(key=lambda x: -x[0])
    split_indices = sorted(idx for _, idx in gaps[: min(2, len(gaps))])
    tiers: list[list[int]] = []
    start = 0
    for split_idx in split_indices:
        chunk = items[start: split_idx + 1]
        tiers.append([i for _, i in chunk])
        start = split_idx + 1
    tail = items[start:]
    tiers.append([i for _, i in tail])
    return tiers


def _has_concave_hub_blend_neighbor(
    face_idx: int,
    graph: FaceGraph,
    by_index: dict[int, Any],
) -> bool:
    for nb in graph.neighbors.get(face_idx, set()):
        if graph.edge_kind(face_idx, nb) != "concave":
            continue
        if by_index[nb].surface_type in HUB_BLEND_TYPES:
            return True
    return False


def _hole_wall_adjacent(
    face_idx: int,
    hole_claimed: set[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
) -> bool:
    return any(
        nb in hole_claimed and by_index[nb].surface_type in WALL_TYPES
        for nb in graph.neighbors.get(face_idx, set())
    )


def _is_interior_recess_floor(face_idx: int, graph: FaceGraph) -> bool:
    """A genuine open-pocket floor is an interior recess: it meets at least one
    neighbour across a CONCAVE edge (material sits inside the pocket). An
    exterior prism cap is bounded entirely by convex edges — no concave edge —
    so it fails this test and cannot seed an open pocket. Pure topology: no face
    lists, no part gate, no size constant.
    """
    return any(
        graph.edge_kind(face_idx, nb) == "concave"
        for nb in graph.neighbors.get(face_idx, set())
    )


def _grow_open_pocket(
    seed: int,
    coaxial: set[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    axial_band_mm: float,
    *,
    exclude: set[int] | None = None,
) -> set[int]:
    """BFS pocket interior: planes + cylinders within axial band; stop at torus/cone."""
    exclude = exclude or set()
    seed_y = float(by_index[seed].centroid[1])
    claimed: set[int] = {seed}
    q: deque[int] = deque([seed])
    while q:
        fid = q.popleft()
        for nb in graph.neighbors.get(fid, set()):
            if nb not in coaxial or nb in claimed or nb in exclude:
                continue
            st = by_index[nb].surface_type
            if st in BLEND_TYPES or st == "cone":
                continue
            if st not in {"plane", "cylinder"}:
                continue
            if abs(float(by_index[nb].centroid[1]) - seed_y) > axial_band_mm:
                continue
            claimed.add(nb)
            q.append(nb)
    return claimed


def _decompose_hub_stack(
    coaxial: set[int],
    hole_claimed: set[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    part_axis_top: float,
    config: CoaxialStackDetectionConfig,
) -> tuple[set[int], set[int]]:
    """Return (hub_flat_faces, open_pocket_faces) via depth-tier roles.

    Multi-tier hub (Toolpath pattern):
      * deepest horizontal tier, largest area  → flat (deferred to flats pass)
      * shallower horizontal, concave-bounded  → open_pocket floor + step-wall grow

    Single-tier fallback preserves legacy open_pocket-from-largest-plane behavior.
    """
    def depth_fn(i: int) -> float:
        return _face_depth_below_top(i, by_index, opening_axis, part_axis_top)

    horizontal: list[int] = []
    for fid in coaxial:
        fg = by_index[fid]
        if not _is_horizontal_plane(fg, opening_axis, config.horizontal_normal_tol_deg):
            continue
        if depth_fn(fid) < config.min_floor_depth_mm:
            continue
        if float(fg.area) < config.min_floor_area_mm2:
            continue
        horizontal.append(fid)

    if not horizontal:
        return set(), set()

    tiers = _depth_tiers_from_gaps(horizontal, depth_fn)
    hub_flat: set[int] = set()
    pocket_seed: int | None = None

    if len(tiers) >= 2:
        deepest_tier = tiers[-1]
        shallow_tiers = tiers[:-1]
        hub_flat.add(max(deepest_tier, key=lambda i: float(by_index[i].area)))

        for tier in shallow_tiers:
            for fid in tier:
                if _has_concave_hub_blend_neighbor(fid, graph, by_index):
                    pocket_seed = fid
                    break
            if pocket_seed is not None:
                break
        if pocket_seed is None:
            pocket_seed = max(
                [i for tier in shallow_tiers for i in tier],
                key=lambda i: float(by_index[i].area),
            )
    else:
        seeds = [
            fid for fid in horizontal
            if _hole_wall_adjacent(fid, hole_claimed, by_index, graph)
        ]
        if not seeds:
            return set(), set()
        pocket_seed = max(seeds, key=lambda i: float(by_index[i].area))

    if pocket_seed is None:
        return hub_flat, set()

    # Predicate A: the floor seed must be an interior recess (>=1 concave edge).
    # Rejects exterior prism caps that the coaxial pool grabbed via hole walls —
    # those are convex-bounded and carry no open pocket. No hub flat is deferred
    # when there is no genuine pocket; the faces flow to the downstream passes.
    if not _is_interior_recess_floor(pocket_seed, graph):
        logger.info(
            "coaxial: floor seed %d has no concave boundary (convex-bounded "
            "prism cap) — no open pocket", pocket_seed,
        )
        return set(), set()

    open_faces = _grow_open_pocket(
        pocket_seed,
        coaxial,
        by_index,
        graph,
        config.pocket_axial_band_mm,
        exclude=hub_flat,
    )
    return hub_flat, open_faces


def _open_pocket_floor_radial_mm(
    open_pocket: set[int],
    by_index: dict[int, Any],
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
) -> float:
    floors = [i for i in open_pocket if by_index[i].surface_type == "plane"]
    if not floors:
        return 0.0
    main_floor = max(floors, key=lambda i: float(by_index[i].area))
    return _radial_dist(by_index[main_floor].centroid, axis_point, axis_dir)


def _hub_contour_tori(
    open_pocket: set[int],
    coaxial: set[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    hub_radial_margin_mm: float = 8.0,
) -> set[int]:
    """Torus step-rims near the pocket floor axis, not perimeter blend torus."""
    floors = [i for i in open_pocket if by_index[i].surface_type == "plane"]
    if not floors:
        return set()
    main_floor = max(floors, key=lambda i: float(by_index[i].area))
    floor_rad = _radial_dist(by_index[main_floor].centroid, axis_point, axis_dir)

    out: set[int] = set()
    for fid in coaxial:
        if by_index[fid].surface_type != "torus":
            continue
        if _radial_dist(by_index[fid].centroid, axis_point, axis_dir) > floor_rad + hub_radial_margin_mm:
            continue
        if not any(nb in open_pocket for nb in graph.neighbors.get(fid, set())):
            continue
        if any(
            nb in coaxial
            and by_index[nb].surface_type == "cone"
            and graph.edge_kind(fid, nb) == "convex"
            for nb in graph.neighbors.get(fid, set())
        ):
            continue
        out.add(fid)
    return out


def _hub_contour_cone_torus(
    open_pocket: set[int],
    contour: set[int],
    coaxial: set[int],
    by_index: dict[int, Any],
    graph: FaceGraph,
) -> set[int]:
    """Cone+torus contour blend concave to pocket/step-rim, convex to each other."""
    anchor = open_pocket | contour
    cones = [
        i for i in coaxial
        if by_index[i].surface_type == "cone"
        and any(nb in anchor for nb in graph.neighbors.get(i, set()))
    ]
    claimed: set[int] = set()
    for cone in cones:
        partners = {
            nb for nb in graph.neighbors.get(cone, set())
            if nb in coaxial
            and by_index[nb].surface_type == "torus"
            and graph.edge_kind(cone, nb) == "convex"
        }
        if partners:
            claimed.add(cone)
            claimed |= partners
    return claimed


def detect_coaxial_stack(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | None = None,
    hole_claimed_faces: set[int] | None = None,
    hole_features: Sequence[Any] | None = None,
    opening_axis: Sequence[float] | None = None,
    part_axis_top: float | None = None,
    config: CoaxialStackDetectionConfig | None = None,
) -> CoaxialStackDetectionResult:
    """Claim central coaxial hub features from the hole-pass residual."""
    config = config or CoaxialStackDetectionConfig()
    n_faces = len(faces)
    by_index = {int(f.index): f for f in faces}
    pool = set(range(n_faces)) if candidate_faces is None else {
        i for i in candidate_faces if 0 <= i < n_faces
    }
    hole_claimed = hole_claimed_faces or set()
    hole_features = hole_features or []
    axis_info = _stack_axis_from_holes(hole_features)
    opening_axis = (
        _unit(np.asarray(opening_axis, dtype=float))
        if opening_axis is not None
        else (axis_info[1] if axis_info else np.array([0.0, 1.0, 0.0]))
    )
    if part_axis_top is None:
        part_axis_top = max(float(f.centroid[1]) for f in faces)

    if axis_info is None:
        logger.info("no hole axis — skip coaxial stack pass")
        return CoaxialStackDetectionResult(
            features=[], claimed_faces=set(), remaining_faces=pool,
            n_faces=n_faces, units=config.units,
        )

    axis_point, axis_dir = axis_info
    radial_margin_mm = resolve_scaled_mm(
        config.coaxial_radial_margin_mm, config.coaxial_radial_margin_frac, faces,
    )
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)
    coaxial = _coaxial_pool(
        pool, hole_claimed, by_index, graph, axis_point, axis_dir, config,
        radial_margin_mm,
    )
    if not coaxial:
        logger.info("empty coaxial pool — skip")
        return CoaxialStackDetectionResult(
            features=[], claimed_faces=set(), remaining_faces=pool,
            n_faces=n_faces, units=config.units,
        )

    hub_flat_faces, open_faces = _decompose_hub_stack(
        coaxial, hole_claimed, by_index, graph, opening_axis, part_axis_top, config,
    )
    contour_faces = (
        _hub_contour_tori(
            open_faces, coaxial, by_index, graph, axis_point, axis_dir,
            hub_radial_margin_mm=config.hub_contour_radial_margin_mm,
        )
        if open_faces else set()
    )
    if open_faces:
        contour_faces |= _hub_contour_cone_torus(
            open_faces, contour_faces, coaxial, by_index, graph,
        )

    hub_perimeter_context: HubPerimeterFilletContext | None = None
    if open_faces:
        floor_rad = _open_pocket_floor_radial_mm(open_faces, by_index, axis_point, axis_dir)
        hub_perimeter_context = HubPerimeterFilletContext(
            axis_point=tuple(float(x) for x in axis_point),
            axis_direction=tuple(float(x) for x in axis_dir),
            open_pocket_floor_radial_mm=floor_rad,
            perimeter_radial_margin_mm=config.hub_contour_radial_margin_mm,
        )

    features: list[CoaxialStackFeature] = []
    claimed: set[int] = set()
    next_id = 0

    def _emit(kind: str, tp_class: str, face_set: set[int]) -> None:
        nonlocal next_id, claimed
        if not face_set:
            return
        features.append(CoaxialStackFeature(
            feature_id=next_id,
            kind=kind,
            face_indices=set(face_set),
            toolpath_class=tp_class,
        ))
        claimed |= face_set
        next_id += 1
        logger.info("%s %d: faces=%s", kind, next_id - 1, sorted(face_set))

    _emit("open_pocket", "open_pocket", open_faces)
    _emit("contour_surface", "contour_surface", contour_faces - claimed)

    remaining = pool - claimed
    return CoaxialStackDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=remaining,
        n_faces=n_faces,
        hub_flat_faces=hub_flat_faces,
        hub_perimeter_context=hub_perimeter_context,
        units=config.units,
    )


@dataclass
class CoaxialStackValidationReport:
    ok: bool
    checks: list[tuple[str, bool, str]]

    def render(self) -> str:
        lines = [f"coaxial stack validation: {'PASS' if self.ok else 'FAIL'}"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_coaxial_stack(
    result: CoaxialStackDetectionResult,
    *,
    expected_open_pocket_faces: Sequence[int] | None = None,
    expected_contour_faces: Sequence[int] | None = None,
    expected_hub_flat_faces: Sequence[int] | None = None,
    forbidden_faces: Sequence[int] | None = None,
) -> CoaxialStackValidationReport:
    checks: list[tuple[str, bool, str]] = []
    by_kind = {f.kind: f for f in result.features}

    if expected_hub_flat_faces is not None:
        exp = set(expected_hub_flat_faces)
        got = set(result.hub_flat_faces)
        checks.append((
            f"hub flat faces {sorted(expected_hub_flat_faces)}",
            got == exp,
            f"got {sorted(got)}",
        ))

    if expected_open_pocket_faces is not None:
        got = by_kind.get("open_pocket", CoaxialStackFeature(0, "open_pocket", set(), "open_pocket")).face_indices
        exp = set(expected_open_pocket_faces)
        checks.append((
            f"open_pocket faces {sorted(expected_open_pocket_faces)}",
            got == exp,
            f"got {sorted(got)}",
        ))

    if expected_contour_faces is not None:
        got = by_kind.get(
            "contour_surface",
            CoaxialStackFeature(0, "contour_surface", set(), "contour_surface"),
        ).face_indices
        exp = set(expected_contour_faces)
        checks.append((
            f"contour_surface faces {sorted(expected_contour_faces)}",
            got == exp,
            f"got {sorted(got)}",
        ))

    if forbidden_faces is not None:
        bad = set(forbidden_faces) & result.claimed_faces
        checks.append((
            f"forbidden faces not claimed {sorted(forbidden_faces)}",
            not bad,
            f"overlap {sorted(bad)}",
        ))

    checks.append((
        "remaining pool excludes claimed faces",
        not (result.claimed_faces & result.remaining_faces),
        f"overlap {sorted(result.claimed_faces & result.remaining_faces)}",
    ))

    ok = all(c[1] for c in checks)
    return CoaxialStackValidationReport(ok=ok, checks=checks)


def render_table(result: CoaxialStackDetectionResult) -> str:
    header = f"{'id':>3}  {'kind':>16}  {'#f':>3}  faces"
    lines = [header, "-" * len(header)]
    for f in result.features:
        lines.append(
            f"{f.feature_id:>3}  {f.toolpath_class:>16}  {f.n_faces:>3}  "
            f"{sorted(f.face_indices)}"
        )
    if result.hub_flat_faces:
        lines.append(f"  {'flat (deferred)':>16}  {len(result.hub_flat_faces):>3}  "
                     f"{sorted(result.hub_flat_faces)}")
    lines.append("")
    lines.append(result.summary())
    return "\n".join(lines)


DEFAULT_STEP = "96260B_FRONT_XR004_PCD PLATE.stp copy"
DEFAULT_GRAPH_NPZ = "pipeline_out/96260B_front/graph.npz"


# ---------------------------------------------------------------------------
# OCC-free self-tests for Predicate A (interior-recess floor gate)
# ---------------------------------------------------------------------------
def _mk_face(index, stype, area, centroid, normal=(0, 0, 0), radius=None, axis=None):
    from feature_params import FaceGeom

    return FaceGeom(
        index=index,
        surface_type=stype,
        area=float(area),
        centroid=np.asarray(centroid, dtype=float),
        normal=np.asarray(normal, dtype=float),
        radius=radius,
        axis=None if axis is None else np.asarray(axis, dtype=float),
    )


def _mk_graph(n, edges):
    """edges: list of (u, v, kind) with kind in concave|convex|smooth."""
    kind_col = {"concave": 0, "convex": 1, "smooth": 2}
    ei = np.array([[u for u, _v, _k in edges], [v for _u, v, _k in edges]], dtype=np.int64)
    ea = np.zeros((len(edges), 4), dtype=np.float32)
    for row, (_u, _v, k) in enumerate(edges):
        ea[row, kind_col[k]] = 1.0
    return FaceGraph.from_edge_tensors(ei, ea, n)


def _selftest() -> int:
    """Exercise Predicate A directly: an interior-recess floor (>=1 concave
    boundary edge) seeds an open pocket; a convex-only prism cap does not.
    Independent of the (legacy) axial-band grow axis. numpy only, no OCC."""
    cfg = CoaxialStackDetectionConfig()
    opening_axis = np.array([0.0, 0.0, 1.0])
    part_axis_top = 0.0
    fails: list[str] = []

    # Shared face layout: one hole wall (cylinder), one horizontal floor plane
    # (hole-wall-adjacent, depth 15mm, area 1500), one vertical step wall plane.
    def build(floor_to_wall_kind: str):
        faces = [
            _mk_face(0, "cylinder", 200.0, (5, 0, -10), radius=3.0, axis=(0, 0, 1)),
            _mk_face(1, "plane", 1500.0, (0, 0, -15), normal=(0, 0, 1)),
            _mk_face(2, "plane", 300.0, (10, 0, -15), normal=(1, 0, 0)),
        ]
        by_index = {int(f.index): f for f in faces}
        graph = _mk_graph(3, [
            (0, 1, "convex"),            # hole wall -> floor (exterior contact)
            (1, 2, floor_to_wall_kind),  # floor -> step wall
        ])
        return by_index, graph

    # 1. look-alike open pocket ACCEPTED: floor meets step wall concavely.
    by_index, graph = build("concave")
    assert _is_interior_recess_floor(1, graph), "concave floor should be recess"
    _hub, open_faces = _decompose_hub_stack(
        {1, 2}, {0}, by_index, graph, opening_axis, part_axis_top, cfg)
    if not open_faces:
        fails.append("interior-recess floor was NOT claimed as open pocket")
    elif 1 not in open_faces:
        fails.append(f"floor seed missing from open pocket: {sorted(open_faces)}")

    # 2. convex-only prism cap REJECTED: floor meets step wall convexly.
    by_index, graph = build("convex")
    if _is_interior_recess_floor(1, graph):
        fails.append("convex-only cap wrongly flagged as interior recess")
    _hub, open_faces = _decompose_hub_stack(
        {1, 2}, {0}, by_index, graph, opening_axis, part_axis_top, cfg)
    if open_faces:
        fails.append(f"convex-only cap wrongly claimed open pocket: {sorted(open_faces)}")

    if fails:
        print("coaxial_stack Predicate-A selftest: FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("coaxial_stack Predicate-A selftest: PASS "
          "(interior-recess floor accepted; convex-only prism cap rejected)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Coaxial hub stack pass")
    ap.add_argument("step", nargs="?", default=DEFAULT_STEP)
    ap.add_argument("--graph-npz", type=Path, default=Path(DEFAULT_GRAPH_NPZ))
    ap.add_argument("--selftest", action="store_true",
                    help="run OCC-free Predicate-A unit checks and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from run_cascade import _load_edges, run_cascade

    step_path = Path(args.step)
    edge_index, edge_attr = _load_edges(args.graph_npz, step_path)
    _, pk, hl, cx, _, _, _, _, _ = run_cascade(step_path, edge_index, edge_attr)
    result = cx

    print(f"STEP: {step_path}")
    print(render_table(result))
    print()
    if "FRONT" in step_path.name.upper():
        report = validate_coaxial_stack(
            result,
            expected_hub_flat_faces=REFERENCE_HUB_FLAT_FACES_FRONT,
            expected_open_pocket_faces=REFERENCE_OPEN_POCKET_FACES_FRONT,
            expected_contour_faces=REFERENCE_CONTOUR_FACES_FRONT,
            forbidden_faces=(97, 280, 298),
        )
    else:
        report = validate_coaxial_stack(
            result,
            expected_hub_flat_faces=REFERENCE_HUB_FLAT_FACES_REAR,
            expected_open_pocket_faces=REFERENCE_OPEN_POCKET_FACES_REAR,
            expected_contour_faces=REFERENCE_CONTOUR_FACES_REAR,
        )
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
