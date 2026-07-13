"""
part_scale.py — characteristic part length for de-scaling cascade thresholds.

Several detectors historically pinned absolute-millimetre thresholds (radial
grab margins, minimum feature areas, radial drops) that were tuned to the
96260B reference plate's physical size. Those constants silently assume a
part of that scale and over/under-annex on differently-sized parts.

This module exposes a single geometry-driven characteristic length so those
constants can be expressed as a *fraction of part scale* and track part size.

REFERENCE_PART_SCALE_MM is the 96260B calibration point: fractions are chosen
so ``frac * REFERENCE_PART_SCALE_MM`` reproduces the old absolute value on the
reference part, then scale naturally on new parts.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Mean centroid-bbox diagonal of the 96260B front (221.4 mm) and rear
# (227.9 mm) plates. Used only to calibrate default fractions so behaviour on
# the reference part is preserved; not consumed by detection logic.
REFERENCE_PART_SCALE_MM = 224.6


def characteristic_scale(faces: Sequence[Any]) -> float:
    """Characteristic part length (mm): diagonal of the face-centroid bbox.

    Monotonic with part size and always available (every FaceGeom carries a
    centroid). Returns 0.0 for an empty part.
    """
    if not faces:
        return 0.0
    c = np.stack([np.asarray(f.centroid, dtype=np.float64) for f in faces], axis=0)
    dims = c.max(axis=0) - c.min(axis=0)
    return float(np.linalg.norm(dims))


def resolve_scaled_mm(
    absolute_mm: float | None,
    frac: float,
    faces: Sequence[Any],
) -> float:
    """Resolve a threshold that may be pinned absolute or derived from scale.

    If ``absolute_mm`` is not None it wins (explicit override / legacy pin).
    Otherwise return ``frac * characteristic_scale(faces)`` so the threshold
    tracks part size. Falls back to ``frac * REFERENCE_PART_SCALE_MM`` if the
    part has no usable geometry.
    """
    if absolute_mm is not None:
        return float(absolute_mm)
    scale = characteristic_scale(faces)
    if scale <= 0.0:
        scale = REFERENCE_PART_SCALE_MM
    return frac * scale


def resolve_scaled_mm2(
    absolute_mm2: float | None,
    frac: float,
    faces: Sequence[Any],
) -> float:
    """Area analogue of :func:`resolve_scaled_mm` (scales with part scale**2).

    If ``absolute_mm2`` is not None it wins. Otherwise return
    ``frac * characteristic_scale(faces) ** 2`` so an area gate tracks part
    size quadratically.
    """
    if absolute_mm2 is not None:
        return float(absolute_mm2)
    scale = characteristic_scale(faces)
    if scale <= 0.0:
        scale = REFERENCE_PART_SCALE_MM
    return frac * scale * scale
