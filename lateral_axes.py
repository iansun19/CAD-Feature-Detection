"""Lateral (±X/±Y/±Z) approach candidacy + reachability — PROVISIONAL.

⚠️  UNVALIDATED CODE PATH.  The only real part in this repo (96260B) is a flat
plate whose every feature opens along a single ±Z opening axis; it has NO
side-access features. Nothing here has been checked against a real part that
actually needs a lateral setup. Treat every result as a hypothesis, clearly
separate from the calibrated Z-only path in [[approach_vectors]] / [[reachability]].

Design — reuse, do not rewrite (see the lateral-axis task brief):
    The calibrated Z-only *collision primitive* (``reachability._direction_result``)
    was already parameterised by an arbitrary outward direction ``d``; only the
    *direction enumeration* in ``annotate_reachability`` was hardcoded to
    ``{+Z, -Z}`` along the opening axis. So extending to six cardinal directions
    needs NO change to the collision math: we enumerate all six cardinals and call
    the unmodified primitive once per direction. Rays are cast in the real part
    frame along the real cardinal vector, so the part solid, the feature points,
    and (if ever supplied) a stock/fixture solid are all queried in the ONE part
    frame — the "rotated feature vs. stale global-frame stock" bug is structurally
    impossible here because nothing is rotated independently.

    ``axis_frames`` provides the exact rigid reframe used by the pure-geometry
    candidacy below and proved equivalent to the direct call in the tests
    (``tests/test_lateral_axes.py``): reframing feature points to +Z and testing a
    +Z corridor against a reframed part is identical to testing the cardinal
    corridor directly, because the transform is a rigid rotation.

Additive only: attaches ``node['approach']['lateral']`` (reachability) and
``node['approach']['lateral_candidates']`` (candidacy). It never touches the
calibrated ``approach`` / ``reachability`` fields, and never feeds partitioning.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from approach_vectors import _candidate_axis, _unit
from axis_frames import CARDINAL_LABELS, CARDINAL_VECTORS, nearest_cardinal
from reachability import (
    _Corridor,
    _direction_result,
    _feature_boundary_points,
)

# A feature's intrinsic axis must lie within this angle of a cardinal direction
# to be a lateral candidate for it. Matches the Z-only parallel tolerance
# (approach_vectors.APPROACH_PARALLEL_TOL_DEG) so lateral candidacy is no looser
# than the calibrated axial one — provisional, not a retune of the Z path.
LATERAL_PARALLEL_TOL_DEG = 15.0

_PROVISIONAL_NOTE = "lateral ±X/±Y/±Z path — provisional, no real-part validation"


def lateral_candidates_for_node(
    node: dict[str, Any],
    *,
    z: np.ndarray,
    normals: dict[int, np.ndarray],
    areas: dict[int, float],
    tol_deg: float = LATERAL_PARALLEL_TOL_DEG,
) -> dict[str, Any]:
    """Which cardinal directions the feature's intrinsic axis could be approached
    from, by pure geometry (no collision test). Returns a summary dict."""
    a, source = _candidate_axis(node, z, normals, areas)
    if a is None:
        return {"provisional": True, "source": source, "axis": None, "candidates": []}

    a = _unit(a)
    # Directional sources (opening axis, flat outward normal, aggregate normal)
    # carry a meaningful sign — approach from the side the axis points to. A bore
    # axis has ambiguous sign, so a hole is a candidate from BOTH ends; the
    # collision pass decides which end is actually open.
    bidirectional = source == "hole_axis"
    label, angle = nearest_cardinal(a)
    candidates: list[dict[str, Any]] = []
    for lbl in CARDINAL_LABELS:
        cos = float(np.dot(a, CARDINAL_VECTORS[lbl]))
        signed = abs(cos) if bidirectional else cos
        if signed <= 0.0:
            continue
        ang = float(np.degrees(np.arccos(max(-1.0, min(1.0, signed)))))
        if ang <= tol_deg:
            candidates.append({"dir": lbl, "angle_deg": round(ang, 3)})
    return {
        "provisional": True,
        "source": source,
        "axis": [round(float(x), 6) for x in _unit(a)],
        "nearest_cardinal": label,
        "nearest_angle_deg": round(angle, 3),
        "candidates": candidates,
    }


def annotate_lateral_candidates(
    nodes: Sequence[dict[str, Any]],
    *,
    faces: Sequence[Any],
    opening_axis: Sequence[float],
    tol_deg: float = LATERAL_PARALLEL_TOL_DEG,
) -> dict[str, Any]:
    """Attach ``node['approach']['lateral_candidates']`` in place; return a summary.

    Pure geometry, no OCC. Safe to run anywhere numpy is available.
    """
    z = _unit(np.asarray(opening_axis, dtype=np.float64))
    normals = {
        int(f.index): np.asarray(f.normal, dtype=np.float64)
        for f in faces
        if getattr(f, "normal", None) is not None
    }
    areas = {int(f.index): float(getattr(f, "area", 1.0) or 1.0) for f in faces}

    n_with_lateral = 0
    for node in nodes:
        summary = lateral_candidates_for_node(
            node, z=z, normals=normals, areas=areas, tol_deg=tol_deg
        )
        approach = node.setdefault("approach", {})
        approach["lateral_candidates"] = summary
        # "lateral" = any cardinal candidate that is not the ±opening-axis pair.
        axial = {_cardinal_of_axis(z), _cardinal_of_axis(-z)}
        if any(c["dir"] not in axial for c in summary["candidates"]):
            n_with_lateral += 1

    return {
        "provisional": True,
        "note": _PROVISIONAL_NOTE,
        "tol_deg": tol_deg,
        "n_features_with_lateral_candidate": n_with_lateral,
    }


def approach_vector_for_setup(setup: Any) -> np.ndarray:
    """Machine-frame unit approach vector for a setup (CamPlan Setup / SetupContext).

    Ops carry no axis of their own; their approach direction *is* their setup's
    orientation. This resolves that to a concrete outward machine-frame vector,
    preferring the general cardinal ``orientation`` (lateral path) over the
    calibrated ``opening_axis`` label — so op geometry is expressed relative to the
    setup approach rather than hardcoded to +Z. The tool travels along ``-vector``.
    """
    label = getattr(setup, "orientation", None) or getattr(setup, "opening_axis", None)
    if label not in CARDINAL_VECTORS:
        raise ValueError(f"setup orientation/opening_axis {label!r} is not a cardinal label")
    return CARDINAL_VECTORS[label].copy()


def _cardinal_of_axis(v: np.ndarray) -> str:
    try:
        return nearest_cardinal(v)[0]
    except ValueError:
        return ""


def annotate_lateral_reachability(
    nodes: Sequence[dict[str, Any]],
    *,
    occ_faces: Sequence[Any],
    shape: Any,
    cardinals: Sequence[str] = CARDINAL_LABELS,
) -> dict[str, Any]:
    """Attach ``node['approach']['lateral']`` = verified reachability over the six
    cardinal directions, in place; return a summary.

    Reuses the calibrated Z-only collision primitive verbatim, once per cardinal
    direction. Requires OCC (real 3D part solid). Does NOT modify the calibrated
    ``reachability`` field.
    """
    occ_by_index = {int(getattr(f, "index", i)): f for i, f in enumerate(occ_faces)}

    all_face_pts = _feature_boundary_points(occ_by_index, list(occ_by_index))
    if all_face_pts.shape[0]:
        diag = float(np.linalg.norm(all_face_pts.max(0) - all_face_pts.min(0)))
    else:
        diag = 1000.0
    ray_length = 2.0 * diag + 10.0

    directions = {lbl: CARDINAL_VECTORS[lbl] for lbl in cardinals}
    corridor = _Corridor(shape)
    n_verified = 0
    for node in nodes:
        pts = _feature_boundary_points(occ_by_index, node.get("face_ids", []))
        approach = node.setdefault("approach", {})
        if pts.shape[0] == 0:
            approach["lateral"] = {
                "provisional": True,
                "note": _PROVISIONAL_NOTE,
                "verified": False,
                "reachable_dirs": [],
                "reason": "no_boundary_geometry",
            }
            continue

        per_dir = {
            lbl: _direction_result(corridor, pts, d, ray_length, node)
            for lbl, d in directions.items()
        }
        reachable_dirs = [lbl for lbl, r in per_dir.items() if not r["occluded"]]
        approach["lateral"] = {
            "provisional": True,
            "note": _PROVISIONAL_NOTE,
            "verified": True,
            "method": "cardinal_direction_enumeration",
            "reachable_dirs": reachable_dirs,
            "per_direction": per_dir,
        }
        n_verified += 1

    return {
        "provisional": True,
        "note": _PROVISIONAL_NOTE,
        "n_verified": n_verified,
        "cardinals": list(cardinals),
        "ray_length_mm": round(ray_length, 3),
    }
