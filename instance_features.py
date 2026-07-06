"""
instance_features.py — shared per-instance geometry summary for cascade passes.

Training and inference share this schema: surface-type histogram, internal-edge
convexity mix, total area, bbox extent, and distinct analytic axis count.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from feature_params import FaceGeom, _unit, bbox_params

# Toolpath "3D Surface: Yes" ↔ any bspline/bezier/sculpted-floor or sphere blend.
SURFACE_3D_TYPES = frozenset({"bspline", "bezier", "sphere"})


def _surface_type_histogram(faces: Sequence[FaceGeom]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for f in faces:
        hist[f.surface_type] = hist.get(f.surface_type, 0) + 1
    return hist


def _distinct_axis_count(faces: Sequence[FaceGeom], *, round_decimals: int = 2) -> int:
    keys: set[tuple[float, ...]] = set()
    for f in faces:
        if f.axis is None:
            continue
        a = _unit(np.asarray(f.axis, dtype=np.float64))
        if float(np.linalg.norm(a)) < 1e-12:
            continue
        if a[int(np.argmax(np.abs(a)))] < 0:
            a = -a
        keys.add(tuple(float(round(float(x), round_decimals)) for x in a))
    return len(keys)


def _internal_edge_convexity_mix(
    face_indices: set[int],
    graph: Any,
) -> dict[str, int]:
    """Count concave/convex/smooth edges with both endpoints in `face_indices`."""
    counts = {"concave": 0, "convex": 0, "smooth": 0}
    for f in face_indices:
        for nb in graph.neighbors.get(f, set()):
            if nb not in face_indices or nb <= f:
                continue
            kind = graph.edge_kind(f, nb)
            if kind in counts:
                counts[kind] += 1
    return counts


def instance_features(
    faces: Sequence[FaceGeom],
    *,
    graph: Any | None = None,
    face_indices: set[int] | None = None,
) -> dict[str, Any]:
    """Build the shared instance-feature dict for a face group.

    Parameters
    ----------
    faces:
        FaceGeom records belonging to this instance (any order).
    graph:
        Optional FaceGraph (or compatible adapter) for internal-edge convexity.
    face_indices:
        When ``graph`` is supplied, the face-index set used to filter internal
        edges (defaults to ``{f.index for f in faces}``).
    """
    idx_set = face_indices if face_indices is not None else {f.index for f in faces}
    total_area = sum(float(f.area) for f in faces)
    out: dict[str, Any] = {
        "face_indices": sorted(idx_set),
        "n_faces": len(idx_set),
        "surface_type_histogram": _surface_type_histogram(faces),
        "3D_surface": any(f.surface_type in SURFACE_3D_TYPES for f in faces),
        "total_area": round(total_area, 3),
        "n_distinct_axes": _distinct_axis_count(faces),
    }
    out.update(bbox_params([f.centroid for f in faces]))
    if graph is not None:
        out["internal_edge_convexity"] = _internal_edge_convexity_mix(idx_set, graph)
    return out
