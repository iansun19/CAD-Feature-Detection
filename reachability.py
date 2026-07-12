"""Verified 3-axis tool reachability (Stage 3, step 4a).

Upgrades the step-2 approach vectors ([[approach_vectors]]) from pure geometric
*candidacy* to *verified* reachability by a swept-tool-cylinder collision test
against the real part solid. It resolves the two seams step 2 admitted: the
through-hole sign guess (test both +/-Z, report the actually-unoccluded sides)
and the null fillet/contour features (give them a real per-direction test).

Read-only post-pass over already-built feature-graph nodes; never feeds back
into partitioning (regression-safe — the golden compares only class_name +
face_ids). Additive: attaches ``node['approach']['reachability']`` and bumps the
graph ``schema_version`` to 4.

Swept-cylinder model (sampled): the tool is modelled as a bundle of parallel
rays covering its circular cross-section (centre + a ring at the tool radius),
cast outward along the approach axis from just above the feature mouth. If any
ray meets solid material in the outward corridor, the tool *body* is blocked
from that direction. This captures tool-width occlusion (not just a centreline
ray). Depth-of-reach is tool-length-only; holder/gauge collision is out of scope
(no holder model exists in the data). See [[cadcam-roadmap]].
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from brep_extents import collect_boundary_points

# Ray offset past the surface so the feature's own face is not counted as an
# obstruction, and ring sampling density for the swept-tool cross-section.
MOUTH_CLEARANCE_MM = 0.5
N_RING_SAMPLES = 6
# How many on-surface target points to probe per feature (subsampled from the
# feature's boundary points). Multi-point exposed-fraction is what lets a deep
# tier reached through an open cavity above read as reachable, while a genuine
# undercut reads as occluded.
N_TARGET_SAMPLES = 12
# A target point is "clear" if its centre ray plus at least this fraction of its
# tool-radius ring is unobstructed; a direction is reachable if at least this
# fraction of target points are clear. The exposed threshold is intentionally
# low: on a real machined part every recognized feature was in fact cut, so any
# nontrivial exposure along a setup axis counts as reachable. Thin sculpted
# contour bands legitimately expose only a small fraction of their sampled
# points. Walls are exempt entirely (their access is lateral, not axial).
RING_CLEAR_FRACTION = 0.5
EXPOSED_FRACTION_THRESHOLD = 0.15
# Swept-cylinder radius = a real shop cutter, not the feature size. The 3/8"
# endmill is the shop's large tool (wall_geometry.SHOP_LARGE_WALL_TOOL_DIA_MM);
# using it asks "can the largest routine tool descend here", the right permissive
# accessibility test. A tool wider than the feature is a lateral-fit question,
# handled separately from approach occlusion.
SWEPT_TOOL_RADIUS_MM = 0.5 * 0.375 * 25.4  # 4.7625 mm


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _orthonormal_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to d."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(d, ref))
    v = _unit(np.cross(d, u))
    return u, v


class _Corridor:
    """Wraps an OCC shape intersector for outward-corridor collision queries."""

    def __init__(self, shape, tol: float = 1e-6):
        from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector

        self._inter = IntCurvesFace_ShapeIntersector()
        self._inter.Load(shape, tol)

    def blocked(self, start: np.ndarray, direction: np.ndarray, length: float,
                eps: float) -> bool:
        """True if solid material lies along [start+eps, start+length] * direction.

        A ray grazing an edge/vertex can raise a sporadic OCC ``Standard_NullObject``;
        jitter the start slightly and retry, then treat a persistently degenerate
        ray as non-blocking (multi-point voting absorbs the odd dropped ray)."""
        from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt

        d = gp_Dir(float(direction[0]), float(direction[1]), float(direction[2]))
        for jitter in (0.0, 1e-3, -1e-3):
            s = start + jitter
            lin = gp_Lin(gp_Pnt(float(s[0]), float(s[1]), float(s[2])), d)
            try:
                self._inter.Perform(lin, eps, length)
            except RuntimeError:
                continue
            if not self._inter.IsDone():
                return False
            return self._inter.NbPnt() > 0
        return False


def _feature_boundary_points(occ_by_index: dict[int, Any], face_ids: Sequence[int]) -> np.ndarray:
    pts: list[np.ndarray] = []
    for fid in face_ids:
        occ = occ_by_index.get(int(fid))
        if occ is not None:
            pts.append(collect_boundary_points(occ))
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(pts, axis=0)


def _subsample(pts: np.ndarray, n: int) -> np.ndarray:
    if pts.shape[0] <= n:
        return pts
    idx = np.linspace(0, pts.shape[0] - 1, n).astype(int)
    return pts[idx]


def _point_clear(corridor: _Corridor, p: np.ndarray, d: np.ndarray,
                 ring: list[np.ndarray], ray_length: float) -> bool:
    """A target point is clear if its centre ray and >= half its tool-radius
    ring escape outward without meeting solid (sampled swept-tool cylinder)."""
    if corridor.blocked(p, d, ray_length, MOUTH_CLEARANCE_MM):
        return False
    ring_clear = sum(
        not corridor.blocked(p + off, d, ray_length, MOUTH_CLEARANCE_MM) for off in ring
    )
    return ring_clear >= RING_CLEAR_FRACTION * len(ring)


def _direction_result(
    corridor: _Corridor,
    all_pts: np.ndarray,
    d: np.ndarray,
    ray_length: float,
    node: dict[str, Any],
) -> dict[str, Any]:
    """Occlusion + depth for one candidate direction d (unit, points outward).

    Probes many real on-surface points across the feature; a direction is
    reachable if enough of them expose a clear swept-tool corridor outward.
    """
    proj = all_pts @ d
    span = float(proj.max() - proj.min())

    r = SWEPT_TOOL_RADIUS_MM
    u, v = _orthonormal_basis(d)
    ring = [
        r * (math.cos(2.0 * math.pi * k / N_RING_SAMPLES) * u
             + math.sin(2.0 * math.pi * k / N_RING_SAMPLES) * v)
        for k in range(N_RING_SAMPLES)
    ]

    targets = _subsample(all_pts, N_TARGET_SAMPLES)
    clear = sum(_point_clear(corridor, p, d, ring, ray_length) for p in targets)
    exposed = clear / len(targets) if len(targets) else 0.0

    return {
        "occluded": bool(exposed < EXPOSED_FRACTION_THRESHOLD),
        "exposed_fraction": round(float(exposed), 3),
        "n_targets": int(len(targets)),
        "required_depth_mm": round(span, 6),
        "effective_tool_radius_mm": round(float(r), 6),
    }


def make_axis_reachability_probe(
    shape: Any,
    occ_faces: Sequence[Any],
    axis: Sequence[float],
):
    """Build a reusable ``probe(face_ids) -> list[str]`` for one setup axis.

    Returns the subset of ``["+Z", "-Z"]`` (interpreted as ``+axis`` / ``-axis``)
    from which the given face group exposes a clear swept-tool corridor -- the
    identical occlusion test :func:`annotate_reachability` runs per node, exposed
    as a standalone callable so an in-pipeline pass (e.g. the chamfer recognizer,
    which must gate *before* the graph/node reachability annotation exists) can
    read the same signal. The OCC ``_Corridor`` and ray length are built once and
    closed over, so repeated probe calls are cheap.
    """
    z = _unit(np.asarray(axis, dtype=np.float64))
    directions = {"+Z": z, "-Z": -z}
    occ_by_index = {int(getattr(f, "index", i)): f for i, f in enumerate(occ_faces)}

    all_face_pts = _feature_boundary_points(occ_by_index, list(occ_by_index))
    if all_face_pts.shape[0]:
        diag = float(np.linalg.norm(all_face_pts.max(0) - all_face_pts.min(0)))
    else:
        diag = 1000.0
    ray_length = 2.0 * diag + 10.0
    corridor = _Corridor(shape)

    def probe(face_ids: Sequence[int]) -> list[str]:
        pts = _feature_boundary_points(occ_by_index, list(face_ids))
        if pts.shape[0] == 0:
            return []
        dirs: list[str] = []
        for name, d in directions.items():
            if not _direction_result(corridor, pts, d, ray_length, {})["occluded"]:
                dirs.append(name)
        return dirs

    return probe


def annotate_reachability(
    nodes: Sequence[dict[str, Any]],
    *,
    occ_faces: Sequence[Any],
    shape: Any,
    opening_axis: Sequence[float],
) -> dict[str, Any]:
    """Attach ``node['approach']['reachability']`` in place; return a summary.

    Requires the step-2 ``approach`` annotation to already be present on nodes.
    """
    z = _unit(np.asarray(opening_axis, dtype=np.float64))
    directions = {"+Z": z, "-Z": -z}
    occ_by_index = {int(getattr(f, "index", i)): f for i, f in enumerate(occ_faces)}

    # Ray length: comfortably past the part along any direction.
    all_face_pts = _feature_boundary_points(occ_by_index, list(occ_by_index))
    if all_face_pts.shape[0]:
        diag = float(np.linalg.norm(all_face_pts.max(0) - all_face_pts.min(0)))
    else:
        diag = 1000.0
    ray_length = 2.0 * diag + 10.0

    corridor = _Corridor(shape)
    n_verified = 0
    for node in nodes:
        approach = node.get("approach")
        if approach is None:
            continue
        # Walls are approached laterally (tool moves along the face, not plunging
        # onto it along the setup axis); the axial-corridor test does not model
        # that, so leave step-2's candidate direction untouched and mark exempt.
        if node.get("class_name") == "wall":
            approach["reachability"] = {
                "verified": False,
                "exempt": True,
                "reachable_dirs": [d for d in (approach.get("setup_dir"),) if d],
                "reason": "lateral_access_wall",
            }
            continue
        pts = _feature_boundary_points(occ_by_index, node.get("face_ids", []))
        if pts.shape[0] == 0:
            approach["reachability"] = {
                "verified": False,
                "reachable_dirs": [],
                "occluded": True,
                "reason": "no_boundary_geometry",
            }
            approach["reachable_3axis"] = False
            approach["setup_dir"] = None
            continue

        per_dir = {name: _direction_result(corridor, pts, d, ray_length, node)
                   for name, d in directions.items()}
        reachable_dirs = [name for name, r in per_dir.items() if not r["occluded"]]

        # Reconcile step-2's candidate setup_dir with what verification found.
        prior = approach.get("setup_dir")
        if prior in reachable_dirs:
            setup_dir = prior
        elif reachable_dirs:
            setup_dir = reachable_dirs[0]
        else:
            setup_dir = None

        primary = setup_dir or (reachable_dirs[0] if reachable_dirs else "+Z")
        approach["reachability"] = {
            "verified": True,
            "reachable_dirs": reachable_dirs,
            "occluded": not reachable_dirs,
            "required_depth_mm": per_dir[primary]["required_depth_mm"],
            "effective_tool_radius_mm": per_dir[primary]["effective_tool_radius_mm"],
            "per_direction": per_dir,
            "corrected_from": prior if setup_dir != prior else None,
        }
        approach["setup_dir"] = setup_dir
        approach["reachable_3axis"] = bool(reachable_dirs)
        n_verified += 1

    return {
        "n_verified": n_verified,
        "ray_length_mm": round(ray_length, 3),
    }
