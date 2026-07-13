"""Per-feature slope profile (Stage 3, Track B) — steep vs shallow face split.

Read-only post-pass over an already-built cascade feature graph: for each feature
node, measure the slope of every member face from horizontal (relative to the setup
opening axis Z), area-weight the faces, and summarise the steep/shallow split. The
planner consumes this to pick a slope-aware surface finish (``steep_shallow``) when a
surface spans both bands, instead of a single raster/scallop pass.

Slope-from-horizontal is the angle between a face normal and Z:
  normal ∥ Z  -> horizontal face -> slope 0°   (shallow / floor-like)
  normal ⊥ Z  -> vertical  face  -> slope 90°  (steep / wall-like, e.g. a bore wall)

Like approach_vectors, this never re-partitions faces; it only attaches
``node['slope_profile']`` in place. One representative normal per face is used (the
persisted face normal), so tightly-curved faces are approximated, not integrated.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# Faces at or above this slope-from-horizontal are "steep" (waterline territory);
# below it they are "shallow" (raster/parallel territory). 45° is the usual
# waterline/raster changeover.
STEEP_SLOPE_DEG = 45.0

# A surface is "mixed" (-> steep_shallow) only when BOTH bands hold at least this
# fraction of its measured area; otherwise it is predominantly one band.
MIXED_MIN_FRACTION = 0.15


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def annotate_slope_profiles(
    nodes: Sequence[dict[str, Any]],
    *,
    faces: Sequence[Any],
    opening_axis: Sequence[float],
    steep_deg: float = STEEP_SLOPE_DEG,
    mixed_min_fraction: float = MIXED_MIN_FRACTION,
) -> dict[str, Any]:
    """Attach ``node['slope_profile']`` to every node in place; return a summary.

    ``node['slope_profile']`` = ``{steep_fraction, shallow_fraction,
    n_faces_measured, mixed, steep_deg}`` (area-weighted over member faces). Nodes
    with no measurable faces get zero fractions and ``mixed=False``.
    """
    z = _unit(np.asarray(opening_axis, dtype=np.float64))
    normals = {
        int(f.index): _unit(np.asarray(f.normal, dtype=np.float64))
        for f in faces
        if getattr(f, "normal", None) is not None
    }
    areas = {int(f.index): float(getattr(f, "area", 0.0) or 0.0) for f in faces}
    cos_thresh = math.cos(math.radians(steep_deg))  # |n·z| <= this  => steep

    mixed_count = 0
    for node in nodes:
        steep_area = 0.0
        shallow_area = 0.0
        measured = 0
        for fid in node.get("face_ids", []):
            n = normals.get(int(fid))
            if n is None:
                continue
            area = areas.get(int(fid), 0.0)
            if area <= 0.0:
                area = 1.0  # count the face even without a usable area
            measured += 1
            if abs(float(np.dot(n, z))) <= cos_thresh:
                steep_area += area
            else:
                shallow_area += area

        total = steep_area + shallow_area
        if total <= 0.0:
            node["slope_profile"] = {
                "steep_fraction": 0.0,
                "shallow_fraction": 0.0,
                "n_faces_measured": 0,
                "mixed": False,
                "steep_deg": steep_deg,
            }
            continue

        steep_fraction = steep_area / total
        shallow_fraction = shallow_area / total
        mixed = (
            steep_fraction >= mixed_min_fraction
            and shallow_fraction >= mixed_min_fraction
        )
        if mixed:
            mixed_count += 1
        node["slope_profile"] = {
            "steep_fraction": round(steep_fraction, 4),
            "shallow_fraction": round(shallow_fraction, 4),
            "n_faces_measured": measured,
            "mixed": mixed,
            "steep_deg": steep_deg,
        }

    return {
        "steep_deg": steep_deg,
        "mixed_min_fraction": mixed_min_fraction,
        "mixed_features": mixed_count,
    }
