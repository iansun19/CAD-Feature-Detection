"""Tests for the chamfer recognizer (closed oblique-bevel ring + reachability gate).

No real part in the repo has a plane+cone oblique-bevel loop about its part axis
(96260B's lip chamfer is a cone+torus band handled elsewhere -- see the module
docstring), so positive detection is proven here on fabricated face/edge geometry.
The negatives lock in the discrimination that keeps the recognizer from firing on
tangent fillet blends, 90-deg steps, 96260B's ~78.5-deg planar lobe facets,
occluded (buried) rings, and lone oblique planes.

The detector was rewritten (commit 1b0addf) from the old edge-angle-between-normals
API to this normals + part-axis + closed-ring + reachability-gate design, and is
wired into run_cascade.py. These tests exercise the current API.
"""
from __future__ import annotations

import math
import os
import sys
import types
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chamfer_detection import (  # noqa: E402
    detect_chamfers,
    derive_part_axis,
    is_oblique_bevel_face,
)

Z = np.array([0.0, 0.0, 1.0])


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / float(np.linalg.norm(v))


def _plane(idx, normal, area=50.0):
    return types.SimpleNamespace(
        index=idx, surface_type="plane", normal=np.asarray(normal, float),
        axis=None, radius=None, semi_angle_rad=None, area=area,
        centroid=np.zeros(3),
    )


def _cone(idx, axis=(0, 0, 1), semi_deg=45.0, area=12.0):
    return types.SimpleNamespace(
        index=idx, surface_type="cone", normal=np.zeros(3),
        axis=np.asarray(axis, float), radius=0.0,
        semi_angle_rad=math.radians(semi_deg), area=area, centroid=np.zeros(3),
    )


def _cap(idx, normal, area):  # large envelope cap -> derives the part axis
    return _plane(idx, normal, area=area)


def _ei(edges):
    if not edges:
        return np.zeros((2, 0), dtype=int)
    a = [e[0] for e in edges] + [e[1] for e in edges]
    b = [e[1] for e in edges] + [e[0] for e in edges]
    return np.array([a, b], dtype=int)


# Four oblique planes (normals 45deg to Z) + four coaxial 45deg cones, wired into
# one closed ring -- the canonical chamfer band around a square step.
_N45 = (_unit([0.0, 1.0, -1.0]), _unit([1.0, 0.0, -1.0]),
        _unit([0.0, -1.0, -1.0]), _unit([-1.0, 0.0, -1.0]))


def _ring_faces():
    return [
        _cap(0, (0, 0, 1), 4000.0), _cap(1, (0, 0, -1), 4000.0),
        _plane(10, _N45[0]), _cone(11), _plane(12, _N45[1]), _cone(13),
        _plane(14, _N45[2]), _cone(15), _plane(16, _N45[3]), _cone(17),
    ]


_RING_EDGES = [(10, 11), (11, 12), (12, 13), (13, 14),
               (14, 15), (15, 16), (16, 17), (17, 10)]
_RING_IDS = [10, 11, 12, 13, 14, 15, 16, 17]

_REACH_ALL = lambda ids: ["+Z", "-Z"]  # noqa: E731
_REACH_NONE = lambda ids: []  # noqa: E731


class TestChamferDetection(unittest.TestCase):
    def test_detects_45deg_ring(self) -> None:
        res = detect_chamfers(
            _ring_faces(), _ei(_RING_EDGES), None,
            candidate_faces=_RING_IDS, reachability_probe=_REACH_ALL,
        )
        self.assertEqual(len(res.features), 1)
        feat = res.features[0]
        self.assertEqual(feat.face_indices_, frozenset(_RING_IDS))
        self.assertAlmostEqual(feat.bevel_angle_deg, 45.0, places=1)
        self.assertEqual(feat.n_planes, 4)
        self.assertEqual(feat.n_cones, 4)
        self.assertEqual(feat.to_dict()["chamfer_angle_deg"], 45.0)

    def test_axis_derived_from_largest_cap(self) -> None:
        res = detect_chamfers(
            _ring_faces(), _ei(_RING_EDGES), None,
            candidate_faces=_RING_IDS, reachability_probe=_REACH_ALL,
        )
        self.assertIsNotNone(res.reference_axis)
        assert res.reference_axis is not None
        self.assertAlmostEqual(abs(res.reference_axis[2]), 1.0, places=6)

    def test_occluded_ring_not_a_chamfer(self) -> None:
        # Fully occluded ring -> rejected by the load-bearing reachability gate.
        res = detect_chamfers(
            _ring_faces(), _ei(_RING_EDGES), None,
            candidate_faces=_RING_IDS, reachability_probe=_REACH_NONE,
        )
        self.assertEqual(res.features, [])

    def test_no_reachability_probe_claims_nothing(self) -> None:
        # Loop candidates exist but no probe supplied -> safe default is empty.
        res = detect_chamfers(
            _ring_faces(), _ei(_RING_EDGES), None, candidate_faces=_RING_IDS,
        )
        self.assertEqual(res.features, [])

    def test_wall_and_floor_not_pulled_into_ring(self) -> None:
        faces = _ring_faces() + [_plane(20, (0, 1, 0), area=200.0),
                                 _plane(21, (0, 0, 1), area=300.0)]
        edges = _RING_EDGES + [(10, 20), (12, 21)]
        res = detect_chamfers(
            faces, _ei(edges), None,
            candidate_faces=_RING_IDS + [20, 21], reachability_probe=_REACH_ALL,
        )
        self.assertEqual(len(res.features), 1)
        self.assertEqual(res.features[0].face_indices_, frozenset(_RING_IDS))

    def test_lone_oblique_plane_insufficient(self) -> None:
        # A single oblique plane is not a closed ring (loop filter).
        faces = [_cap(0, (0, 0, 1), 4000.0), _cap(1, (0, 0, -1), 4000.0),
                 _plane(10, _N45[0])]
        res = detect_chamfers(faces, _ei([]), None, candidate_faces=[10],
                              reachability_probe=_REACH_ALL)
        self.assertEqual(res.features, [])

    def test_excluded_faces_skipped(self) -> None:
        res = detect_chamfers(
            _ring_faces(), _ei(_RING_EDGES), None,
            candidate_faces=_RING_IDS, exclude_faces=frozenset(_RING_IDS),
            reachability_probe=_REACH_ALL,
        )
        self.assertEqual(res.features, [])


class TestObliqueBevelDiscrimination(unittest.TestCase):
    """is_oblique_bevel_face gates the 25..65deg bevel band about the part axis."""

    def test_45deg_plane_is_oblique(self) -> None:
        self.assertTrue(is_oblique_bevel_face(_plane(0, _N45[0]), Z))

    def test_tangent_blend_below_band(self) -> None:
        # A plane whose normal is ~2deg off the axis is an axis-aligned flat /
        # tangent blend, below the 25deg bevel floor -> not oblique.
        self.assertFalse(is_oblique_bevel_face(_plane(0, _unit([0, 0.035, 0.999])), Z))

    def test_step_wall_above_band(self) -> None:
        # 90deg step wall: normal perpendicular to axis -> excluded.
        self.assertFalse(is_oblique_bevel_face(_plane(0, (0, 1, 0)), Z))

    def test_lobe_facet_angle_excluded(self) -> None:
        # 96260B planar lobe facets meet blends at ~78.5deg -> above the band.
        n = _unit([0.0, math.sin(math.radians(78.5)), math.cos(math.radians(78.5))])
        self.assertFalse(is_oblique_bevel_face(_plane(0, n), Z))

    def test_coaxial_45deg_cone_is_oblique(self) -> None:
        self.assertTrue(is_oblique_bevel_face(_cone(0, axis=(0, 0, 1), semi_deg=45.0), Z))

    def test_off_axis_cone_excluded(self) -> None:
        # Cone axis far from the part axis -> not a coaxial bevel.
        self.assertFalse(is_oblique_bevel_face(_cone(0, axis=(1, 0, 0), semi_deg=45.0), Z))


class TestDerivePartAxis(unittest.TestCase):
    def test_largest_planar_face_normal(self) -> None:
        faces = [_plane(0, (0, 0, 1), area=4000.0), _plane(1, (1, 0, 0), area=10.0)]
        axis = derive_part_axis(faces)
        self.assertIsNotNone(axis)
        assert axis is not None
        self.assertAlmostEqual(abs(float(axis[2])), 1.0, places=6)

    def test_no_planar_face_returns_none(self) -> None:
        faces = [_cone(0)]
        self.assertIsNone(derive_part_axis(faces))


if __name__ == "__main__":
    unittest.main()
