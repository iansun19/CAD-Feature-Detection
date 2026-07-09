"""Unit tests for cardinal-axis reframing transforms (axis_frames).

Pure synthetic geometry — no OCC, no real part data. Runs in the repo .venv
(numpy only): ``python -m unittest tests.test_axis_frames``.
"""
from __future__ import annotations

import unittest

import numpy as np

from axis_frames import (
    CARDINAL_LABELS,
    CARDINAL_VECTORS,
    cardinal_label,
    inverse,
    nearest_cardinal,
    rotation_to_plus_z,
    transform_points,
    transform_vector,
)

_PLUS_Z = np.array([0.0, 0.0, 1.0])


class RotationBasicsTests(unittest.TestCase):
    def test_every_cardinal_maps_to_plus_z(self):
        for label in CARDINAL_LABELS:
            R = rotation_to_plus_z(label)
            got = transform_vector(R, CARDINAL_VECTORS[label])
            np.testing.assert_allclose(got, _PLUS_Z, atol=1e-9, err_msg=label)

    def test_rotations_are_proper_orthonormal(self):
        for label in CARDINAL_LABELS:
            R = rotation_to_plus_z(label)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)

    def test_plus_z_axis_fixed(self):
        # +Z need not be the identity (a spin about Z is harmless — the approach
        # corridor is rotationally symmetric about its axis), but it must fix +Z
        # and keep +Z in the z-plane.
        R = rotation_to_plus_z("+Z")
        np.testing.assert_allclose(transform_vector(R, _PLUS_Z), _PLUS_Z, atol=1e-9)
        self.assertAlmostEqual(float(R[2, 2]), 1.0, places=9)

    def test_deterministic(self):
        # Same input must always give the same matrix (no random reference).
        for label in CARDINAL_LABELS:
            np.testing.assert_array_equal(
                rotation_to_plus_z(label), rotation_to_plus_z(label)
            )

    def test_accepts_vector_or_label(self):
        np.testing.assert_allclose(
            rotation_to_plus_z("+X"), rotation_to_plus_z([1.0, 0.0, 0.0]), atol=1e-12
        )


class RoundTripTests(unittest.TestCase):
    def test_inverse_recovers_points(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(20, 3)) * 37.0
        for label in CARDINAL_LABELS:
            R = rotation_to_plus_z(label)
            fwd = transform_points(R, pts)
            back = transform_points(inverse(R), fwd)
            np.testing.assert_allclose(back, pts, atol=1e-9, err_msg=label)

    def test_rigid_preserves_distances_and_angles(self):
        # A rigid rotation cannot distort geometry: pairwise distances and dot
        # products are invariant. This is what makes transform-and-reuse exact.
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(8, 3)) * 12.0
        for label in CARDINAL_LABELS:
            R = rotation_to_plus_z(label)
            tp = transform_points(R, pts)
            d0 = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            d1 = np.linalg.norm(tp[:, None, :] - tp[None, :, :], axis=-1)
            np.testing.assert_allclose(d0, d1, atol=1e-9, err_msg=label)


class StockConsistencyTests(unittest.TestCase):
    """The task's explicit trap: a feature and its stock/fixture must be reframed
    by the SAME transform, or a rotated-feature-vs-stale-stock check silently
    lies. These tests pin the invariant that guards against it."""

    def _inside(self, p, lo, hi):
        return bool(np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9))

    def test_containment_preserved_under_shared_transform(self):
        # Stock box [0,100]^3; a feature point inside it. After reframing BOTH by
        # the same R, the point is still inside the reframed stock box.
        stock = np.array(
            [[x, y, z] for x in (0.0, 100.0) for y in (0.0, 100.0) for z in (0.0, 100.0)]
        )
        feature_pt = np.array([25.0, 60.0, 10.0])
        self.assertTrue(self._inside(feature_pt, stock.min(0), stock.max(0)))

        for label in CARDINAL_LABELS:
            R = rotation_to_plus_z(label)
            stock_r = transform_points(R, stock)
            pt_r = transform_points(R, feature_pt)
            self.assertTrue(
                self._inside(pt_r, stock_r.min(0), stock_r.max(0)),
                f"{label}: containment broke — feature/stock reframed inconsistently",
            )

    def test_stale_stock_would_be_detectably_wrong(self):
        # Negative control: reframe the feature but NOT the stock (the bug the
        # task warns about) and confirm the mismatch is actually observable, so
        # the positive test above has teeth. Use a non-symmetric stock so a
        # rotation genuinely moves its AABB.
        stock = np.array(
            [[x, y, z] for x in (0.0, 200.0) for y in (0.0, 40.0) for z in (0.0, 40.0)]
        )
        feature_pt = np.array([180.0, 20.0, 20.0])  # inside, near the long +X end
        R = rotation_to_plus_z("+X")  # sends +X -> +Z, so the long axis becomes Z
        pt_r = transform_points(R, feature_pt)
        stale_lo, stale_hi = stock.min(0), stock.max(0)  # NOT reframed
        self.assertFalse(
            self._inside(pt_r, stale_lo, stale_hi),
            "expected stale-stock check to be wrong, proving the invariant matters",
        )


class LabelTests(unittest.TestCase):
    def test_cardinal_label_snaps(self):
        self.assertEqual(cardinal_label([0.0, 0.999, 0.02]), "+Y")
        self.assertEqual(cardinal_label([-1.0, 0.05, 0.0]), "-X")

    def test_cardinal_label_rejects_oblique(self):
        with self.assertRaises(ValueError):
            cardinal_label([1.0, 1.0, 0.0])

    def test_nearest_cardinal_reports_angle(self):
        label, angle = nearest_cardinal([0.0, 0.0, 1.0])
        self.assertEqual(label, "+Z")
        self.assertAlmostEqual(angle, 0.0, places=6)
        label, angle = nearest_cardinal([0.1, 0.0, 1.0])
        self.assertEqual(label, "+Z")
        self.assertGreater(angle, 0.0)


if __name__ == "__main__":
    unittest.main()
