"""
exterior_boundary.py — shared exterior lobe boundary geometry for wall + profile passes.

Toolpath-aligned split on the reference plate:
  * wall    — one vertical OD notch segment per seed (max-R, max-D cylinder sliver)
  * profile — one merged lobe-exterior sculpt chain per part (C10): inboard
                cone/torus/cylinder/blends grown convex/smooth from each wall seed

Units are mm. numpy only.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from brep.feature_params import FaceGeom
from cascade.hole_detection import FaceGraph, _unit
from brep.part_scale import resolve_scaled_mm

VERTICAL_WALL_TYPES = frozenset({"cylinder", "cone", "plane"})
LOBE_GROW_TYPES = frozenset({"cylinder", "cone", "plane", "torus", "bspline", "bezier"})


@dataclass
class ExteriorBoundaryConfig:
    vertical_wall_min_deg: float = 60.0
    exterior_radial_margin_mm: float = 2.0
    exterior_diameter_rel_tol: float = 0.03
    opening_axis_parallel_tol_deg: float = 15.0
    # Inboard radial floor for lobe sculpt grow (mm below R_exterior). Absolute
    # if set; when None derived as ``lobe_grow_radial_drop_frac * part scale``
    # so the drop tracks part size (was pinned to 96260B's ~12 mm).
    lobe_grow_radial_drop_mm: float | None = None
    lobe_grow_radial_drop_frac: float = 0.05343  # 12.0 mm / 224.6 mm ref plate
    grow_surface_types: frozenset[str] = LOBE_GROW_TYPES
    units: str = "mm"


def radial_mm(centroid: np.ndarray, opening_axis: np.ndarray) -> float:
    a = _unit(opening_axis)
    c = np.asarray(centroid, dtype=np.float64)
    perp = c - np.dot(c, a) * a
    return float(np.linalg.norm(perp))


def axial_mm(centroid: np.ndarray, opening_axis: np.ndarray) -> float:
    a = _unit(opening_axis)
    c = np.asarray(centroid, dtype=np.float64)
    return float(np.dot(c, a))


def normal_angle_to_opening_deg(
    normal: np.ndarray | None,
    opening_axis: np.ndarray,
) -> float:
    if normal is None:
        return 0.0
    n = _unit(np.asarray(normal, dtype=float))
    a = _unit(opening_axis)
    c = abs(float(np.dot(n, a)))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def axis_parallel(
    axis: np.ndarray | None,
    opening_axis: np.ndarray,
    tol_deg: float,
) -> bool:
    if axis is None:
        return False
    d = _unit(np.asarray(axis, dtype=float))
    a = _unit(opening_axis)
    return abs(float(np.dot(d, a))) >= math.cos(math.radians(tol_deg))


def face_diameter_mm(f: FaceGeom) -> float | None:
    if f.radius is not None:
        return 2.0 * float(f.radius)
    return None


def is_vertical_wall(
    f: FaceGeom,
    opening_axis: np.ndarray,
    config: ExteriorBoundaryConfig,
) -> bool:
    if f.surface_type not in VERTICAL_WALL_TYPES:
        return False
    if normal_angle_to_opening_deg(f.normal, opening_axis) >= config.vertical_wall_min_deg:
        return True
    if f.surface_type == "cylinder" and axis_parallel(
        f.axis, opening_axis, config.opening_axis_parallel_tol_deg,
    ):
        return True
    return False


def interior_claimed(
    fid: int,
    *,
    pocket_claimed: set[int],
    hole_claimed: set[int],
    hub_open_pocket_faces: set[int],
) -> bool:
    return fid in pocket_claimed or fid in hole_claimed or fid in hub_open_pocket_faces


def concave_into_interior(
    fid: int,
    graph: FaceGraph,
    interior: set[int],
) -> bool:
    for nb in graph.neighbors.get(fid, ()):
        if nb in interior and graph.edge_kind(fid, nb) == "concave":
            return True
    return False


def can_grow_face(
    fid: int,
    *,
    chain_so_far: set[int],
    graph: FaceGraph,
    interior: set[int],
) -> bool:
    if not concave_into_interior(fid, graph, interior):
        return True
    for nb in graph.neighbors.get(fid, ()):
        if nb in chain_so_far and graph.edge_kind(fid, nb) in ("convex", "smooth"):
            return True
    return False


def compute_exterior_envelope(
    pool: set[int],
    by_index: dict[int, FaceGeom],
    opening_axis: np.ndarray,
    config: ExteriorBoundaryConfig,
) -> tuple[float, float | None]:
    radials: list[float] = []
    for fid in pool:
        f = by_index[fid]
        if not is_vertical_wall(f, opening_axis, config):
            continue
        radials.append(radial_mm(f.centroid, opening_axis))
    if not radials:
        return 0.0, None
    r_ext = max(radials)
    band = [
        fid for fid in pool
        if radial_mm(by_index[fid].centroid, opening_axis)
        >= r_ext - config.exterior_radial_margin_mm
    ]
    band_diams = [
        face_diameter_mm(by_index[fid])
        for fid in band
        if by_index[fid].surface_type == "cylinder"
        and face_diameter_mm(by_index[fid]) is not None
    ]
    d_ext = max(band_diams) if band_diams else None
    return r_ext, d_ext


def is_exterior_wall_seed(
    fid: int,
    *,
    by_index: dict[int, FaceGeom],
    opening_axis: np.ndarray,
    r_exterior: float,
    d_exterior: float | None,
    interior: set[int],
    graph: FaceGraph,
    config: ExteriorBoundaryConfig,
) -> bool:
    f = by_index[fid]
    if radial_mm(f.centroid, opening_axis) < r_exterior - config.exterior_radial_margin_mm:
        return False
    if not is_vertical_wall(f, opening_axis, config):
        return False
    if fid in interior:
        return False
    if concave_into_interior(fid, graph, interior):
        return False
    if f.surface_type != "cylinder" or f.radius is None:
        return False
    if d_exterior is None:
        return True
    d = face_diameter_mm(f)
    assert d is not None
    return abs(d - d_exterior) / max(d_exterior, 1e-6) <= config.exterior_diameter_rel_tol


def find_exterior_wall_seeds(
    pool: set[int],
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    interior: set[int],
    config: ExteriorBoundaryConfig,
) -> tuple[set[int], float, float | None]:
    r_ext, d_ext = compute_exterior_envelope(pool, by_index, opening_axis, config)
    seeds = {
        fid for fid in pool
        if is_exterior_wall_seed(
            fid,
            by_index=by_index,
            opening_axis=opening_axis,
            r_exterior=r_ext,
            d_exterior=d_ext,
            interior=interior,
            graph=graph,
            config=config,
        )
    }
    return seeds, r_ext, d_ext


def grow_lobe_exterior_chain(
    seed: int,
    *,
    n_faces: int,
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    r_exterior: float,
    interior: set[int],
    forbidden: set[int],
    config: ExteriorBoundaryConfig,
) -> set[int]:
    """Convex/smooth BFS from one exterior wall seed through inboard lobe sculpt."""
    radial_drop = resolve_scaled_mm(
        config.lobe_grow_radial_drop_mm,
        config.lobe_grow_radial_drop_frac,
        list(by_index.values()),
    )
    radial_floor = max(0.0, r_exterior - radial_drop)

    def _allowed(fid: int) -> bool:
        if fid in interior or fid in forbidden:
            return False
        if radial_mm(by_index[fid].centroid, opening_axis) < radial_floor:
            return False
        return by_index[fid].surface_type in config.grow_surface_types

    claimed: set[int] = {seed}
    queue: deque[int] = deque([seed])
    while queue:
        u = queue.popleft()
        for v in graph.neighbors.get(u, ()):
            if v in claimed or not _allowed(v):
                continue
            kind = graph.edge_kind(u, v)
            if kind not in ("convex", "smooth"):
                continue
            if not can_grow_face(v, chain_so_far=claimed, graph=graph, interior=interior):
                continue
            claimed.add(v)
            queue.append(v)
    return claimed


def merged_lobe_profile_faces(
    wall_seeds: set[int],
    *,
    n_faces: int,
    by_index: dict[int, FaceGeom],
    graph: FaceGraph,
    opening_axis: np.ndarray,
    r_exterior: float,
    interior: set[int],
    wall_claimed: set[int],
    config: ExteriorBoundaryConfig,
) -> set[int]:
    """C10 profile sculpt: union of lobe chains minus wall-owned faces."""
    sculpt: set[int] = set()
    for seed in sorted(wall_seeds):
        chain = grow_lobe_exterior_chain(
            seed,
            n_faces=n_faces,
            by_index=by_index,
            graph=graph,
            opening_axis=opening_axis,
            r_exterior=r_exterior,
            interior=interior,
            forbidden=wall_claimed,
            config=config,
        )
        sculpt |= chain - wall_claimed
    return sculpt
