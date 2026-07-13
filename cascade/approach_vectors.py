"""Per-feature 3-axis tool-approach directions (Stage 3, step 2 — Z-only).

Purely geometric candidacy: for each feature node in the cascade feature graph,
derive the direction from which a 3-axis tool would approach it, then snap it to
the part's opening axis (the only setup axis we can verify today). This does NOT
do collision / occlusion / depth-of-reach checking — real reachability is a
later step. It is a read-only post-pass over already-built nodes; it never feeds
back into how faces are partitioned into features.

Frame (Z-only): ``Z = opening_axis``, oriented so ``+Z`` points toward the part
top (the max-projection end, which is how ``opening_axis`` is already signed).
Lateral ``+/-X`` / ``+/-Y`` setup directions are deliberately omitted until a
part with genuine side-access features exists to calibrate against.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# A feature's candidate axis must lie within this angle of +/-Z to be reachable
# in one of the two Z setups; anything more oblique needs >3 axes (or a fixture).
APPROACH_PARALLEL_TOL_DEG = 15.0


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _candidate_axis(
    node: dict[str, Any],
    z: np.ndarray,
    normals: dict[int, np.ndarray],
    areas: dict[int, float],
) -> tuple[np.ndarray | None, str]:
    """Return (unit candidate axis, source) for one feature node.

    Detection is by the geometry actually persisted in ``params`` rather than by
    class name, so it stays robust to class-name churn:
      * pockets      -> ``params.opening_axis``
      * holes        -> ``params.axis.direction``
      * flats        -> ``params.normal``
      * fillets/etc. -> area-weighted mean of member face normals
      * everything else opening-axis (wall / profile / coaxial open_pocket) -> Z
    """
    params = node.get("params") or {}

    opening = params.get("opening_axis")
    if opening is not None:
        return _unit(np.asarray(opening, dtype=np.float64)), "opening_axis"

    axis = params.get("axis")
    if isinstance(axis, dict) and axis.get("direction") is not None:
        return _unit(np.asarray(axis["direction"], dtype=np.float64)), "hole_axis"

    normal = params.get("normal")
    if normal is not None:
        return _unit(np.asarray(normal, dtype=np.float64)), "flat_normal"

    if node.get("class_name") in ("outer_fillet", "contour_surface"):
        acc = np.zeros(3, dtype=np.float64)
        for fid in node.get("face_ids", []):
            n = normals.get(int(fid))
            if n is None:
                continue
            acc += float(areas.get(int(fid), 1.0)) * _unit(n)
        if float(np.linalg.norm(acc)) < 1e-9:
            # Member normals cancel (a fillet wrapping around) -> no single
            # approach direction. Honest null; refine in the reachability step.
            return None, "aggregate_normal"
        return _unit(acc), "aggregate_normal"

    # wall / profile / coaxial open_pocket and any other opening-axis feature.
    return _unit(z), "opening_axis"


def annotate_approach_vectors(
    nodes: Sequence[dict[str, Any]],
    *,
    faces: Sequence[Any],
    opening_axis: Sequence[float],
    tol_deg: float = APPROACH_PARALLEL_TOL_DEG,
) -> dict[str, Any]:
    """Attach ``node['approach']`` to every node in place; return the frame dict.

    ``node['approach']`` = ``{axis, source, setup_dir, reachable_3axis}`` where
    ``axis`` is the outward-pointing unit approach vector (the tool travels along
    ``-axis`` into the feature), ``setup_dir`` is ``"+Z"`` / ``"-Z"`` / ``None``,
    and ``reachable_3axis`` is ``True`` iff a Z setup reaches it.
    """
    z = _unit(np.asarray(opening_axis, dtype=np.float64))
    cos_tol = math.cos(math.radians(tol_deg))

    centroids = {int(f.index): np.asarray(f.centroid, dtype=np.float64) for f in faces}
    normals = {
        int(f.index): np.asarray(f.normal, dtype=np.float64)
        for f in faces
        if getattr(f, "normal", None) is not None
    }
    areas = {int(f.index): float(getattr(f, "area", 1.0) or 1.0) for f in faces}

    projections = [float(np.dot(c, z)) for c in centroids.values()]
    axis_top = max(projections) if projections else 0.0
    axis_bottom = min(projections) if projections else 0.0
    axis_mid = 0.5 * (axis_top + axis_bottom)

    def _feature_axial(node: dict[str, Any]) -> float:
        pts = [centroids[int(f)] for f in node.get("face_ids", []) if int(f) in centroids]
        if not pts:
            return axis_mid
        return float(np.dot(np.mean(pts, axis=0), z))

    for node in nodes:
        a, source = _candidate_axis(node, z, normals, areas)
        if a is None:
            node["approach"] = {
                "axis": None,
                "source": source,
                "setup_dir": None,
                "reachable_3axis": False,
            }
            continue

        cos = float(np.dot(a, z))
        if abs(cos) < cos_tol:
            # Oblique to the opening axis: not reachable in either Z setup.
            out = a if cos >= 0 else -a
            node["approach"] = {
                "axis": [round(float(x), 6) for x in out],
                "source": source,
                "setup_dir": None,
                "reachable_3axis": False,
            }
            continue

        if source == "opening_axis":
            # Opening axis already means "direction the feature opens" == outward.
            sign = 1.0
        else:
            # Hole bore / flat normal: sign is ambiguous, so pick the side of the
            # part the feature sits on (upper half -> +Z, lower half -> -Z).
            sign = 1.0 if _feature_axial(node) >= axis_mid else -1.0

        # Orient the raw candidate axis to the chosen outward hemisphere.
        out = a if (cos >= 0) == (sign >= 0) else -a
        node["approach"] = {
            "axis": [round(float(x), 6) for x in out],
            "source": source,
            "setup_dir": "+Z" if sign >= 0 else "-Z",
            "reachable_3axis": True,
        }

    return {
        "z": [round(float(x), 6) for x in z],
        "part_axis_top": round(axis_top, 6),
        "part_axis_bottom": round(axis_bottom, 6),
    }
