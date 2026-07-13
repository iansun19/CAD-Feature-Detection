"""Cardinal-axis reframing transforms (lateral ±X/±Y/±Z support — PROVISIONAL).

Pure, data-independent geometry. This module exists so the *existing, calibrated*
Z-only approach-vector and reachability logic can be reused verbatim for lateral
setup directions without being rewritten to handle arbitrary ray directions.

The idea (see the lateral-axis task brief): to evaluate a feature from a cardinal
direction ``d`` (one of ±X/±Y/±Z), apply a fixed proper-rotation permutation that
sends ``d`` to local ``+Z``, run the unmodified Z-only logic in that reframed
space, then transform the result back with the inverse rotation. Because every
transform here is a rigid rotation (orthonormal, det = +1), the reuse is *exact*
— no interpolation, no distortion of distances, angles, or containment.

Scope is strictly the six cardinal directions; this module deliberately does NOT
generalize to arbitrary face-normal orientations. See [[cadcam-roadmap]].
"""
from __future__ import annotations

import numpy as np

# Canonical cardinal labels and their unit vectors, in a stable order.
CARDINAL_LABELS: tuple[str, ...] = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
CARDINAL_VECTORS: dict[str, np.ndarray] = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}

_PLUS_Z = np.array([0.0, 0.0, 1.0])


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return v / n


def cardinal_label(vector) -> str:
    """Snap a (near-)axis-aligned unit-ish vector to its cardinal label.

    Raises if the vector is not close to a principal axis — callers that only
    ever pass true cardinals (this module's own tables) never hit that path.
    """
    a = _unit(vector)
    i = int(np.argmax(np.abs(a)))
    if abs(a[i]) < 0.9:  # ~25.8 deg off-axis; not a cardinal
        raise ValueError(f"vector {list(np.asarray(vector))!r} is not axis-aligned")
    return f"{'+' if a[i] >= 0 else '-'}{'XYZ'[i]}"


def nearest_cardinal(vector) -> tuple[str, float]:
    """Return (nearest cardinal label, angle to it in degrees) for any vector."""
    a = _unit(vector)
    i = int(np.argmax(np.abs(a)))
    label = f"{'+' if a[i] >= 0 else '-'}{'XYZ'[i]}"
    cos = float(np.dot(a, CARDINAL_VECTORS[label]))
    angle = float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))
    return label, angle


def rotation_to_plus_z(direction) -> np.ndarray:
    """Proper rotation ``R`` (3x3, orthonormal, det +1) with ``R @ direction ≈ +Z``.

    ``direction`` may be a cardinal label ("+X", "-Y", …) or a unit-ish 3-vector.
    Rows are a right-handed triad (u, v, d): ``R @ d`` = ``+Z`` by construction,
    and ``det = u·(v×d) = 1`` because ``u × v = d``. Deterministic: ``u`` comes
    from a fixed reference so the same input always yields the same matrix.
    """
    if isinstance(direction, str):
        direction = CARDINAL_VECTORS[direction]
    d = _unit(direction)

    # Reference not parallel to d, so cross products are well-conditioned.
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(ref, d))
    v = np.cross(d, u)  # unit and ⟂ both; u × v = d (right-handed)
    R = np.array([u, v, d], dtype=np.float64)

    # Self-check: the whole point is exactness, so assert it rather than trust it.
    assert np.allclose(R @ d, _PLUS_Z, atol=1e-9), "rotation does not send d to +Z"
    assert abs(float(np.linalg.det(R)) - 1.0) < 1e-9, "rotation is not proper (det != 1)"
    return R


def inverse(R: np.ndarray) -> np.ndarray:
    """Inverse of a rotation matrix (its transpose)."""
    return np.asarray(R, dtype=np.float64).T


def transform_points(R: np.ndarray, points) -> np.ndarray:
    """Apply rotation ``R`` to an ``(N, 3)`` (or ``(3,)``) array of row-vector points."""
    P = np.asarray(points, dtype=np.float64)
    if P.ndim == 1:
        return R @ P
    return P @ np.asarray(R, dtype=np.float64).T


def transform_vector(R: np.ndarray, vector) -> np.ndarray:
    """Apply rotation ``R`` to a single direction vector."""
    return np.asarray(R, dtype=np.float64) @ np.asarray(vector, dtype=np.float64)
