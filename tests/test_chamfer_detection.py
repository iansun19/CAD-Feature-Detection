"""Synthetic tests for the chamfer recognizer.

No real part in the repo has chamfers (96260B has none -- see the module docstring),
so positive detection is proven here on fabricated face/edge geometry. The negatives
lock in the discrimination that keeps it from firing on tangent fillet blends, 90-deg
steps, and 96260B's ~78.5-deg planar lobe facets.
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

from chamfer_detection import detect_chamfers  # noqa: E402


def _face(index: int, area: float, surface_type: str = "plane") -> types.SimpleNamespace:
    return types.SimpleNamespace(index=index, surface_type=surface_type, area=area)


def _edges(pairs_angles, convex=True):
    """pairs_angles: list of (i, j, angle_deg). Symmetric edges, convex by default."""
    ei, ea = [], []
    onehot = [0, 1, 0] if convex else [1, 0, 0]  # [concave, convex, smooth]
    for i, j, ang in pairs_angles:
        cos = math.cos(math.radians(ang))
        ei.append((i, j)); ea.append([*onehot, cos])
        ei.append((j, i)); ea.append([*onehot, cos])
    return np.asarray(ei).T, np.asarray(ea, dtype=float)


class TestChamferDetection(unittest.TestCase):
    def _two_big_faces(self):
        # faces 0,1 are large; face 2 is the small bevel between them
        return [_face(0, 500.0), _face(1, 500.0), _face(2, 20.0)]

    def test_detects_45deg_chamfer(self) -> None:
        faces = self._two_big_faces()
        ei, ea = _edges([(2, 0, 45.0), (2, 1, 45.0)])
        res = detect_chamfers(faces, ei, ea)
        self.assertEqual([f.face_index for f in res.features], [2])
        self.assertAlmostEqual(res.features[0].chamfer_angle_deg, 45.0, places=1)
        self.assertEqual(res.features[0].bridged_faces, [0, 1])

    def test_tangent_blend_edges_are_not_chamfers(self) -> None:
        # fillet-like: near-tangent edges (cos ~ 1, ~2 deg) -> below the bevel band
        faces = self._two_big_faces()
        ei, ea = _edges([(2, 0, 2.0), (2, 1, 2.0)])
        self.assertEqual(detect_chamfers(faces, ei, ea).features, [])

    def test_step_edges_are_not_chamfers(self) -> None:
        # 90-deg step walls -> above the bevel band
        faces = self._two_big_faces()
        ei, ea = _edges([(2, 0, 90.0), (2, 1, 90.0)])
        self.assertEqual(detect_chamfers(faces, ei, ea).features, [])

    def test_lobe_facet_angle_is_not_a_chamfer(self) -> None:
        # 96260B's planar lobe facets meet blends at ~78.5 deg -> above the band
        faces = self._two_big_faces()
        ei, ea = _edges([(2, 0, 78.5), (2, 1, 78.5)])
        self.assertEqual(detect_chamfers(faces, ei, ea).features, [])

    def test_concave_bevel_excluded(self) -> None:
        # internal/concave bevels are out of scope (only convex edge-breaks)
        faces = self._two_big_faces()
        ei, ea = _edges([(2, 0, 45.0), (2, 1, 45.0)], convex=False)
        self.assertEqual(detect_chamfers(faces, ei, ea).features, [])

    def test_single_bevel_edge_insufficient(self) -> None:
        # a lone bevel edge (a ramp/draft) does not bevel a corner
        faces = self._two_big_faces()
        ei, ea = _edges([(2, 0, 45.0), (2, 1, 90.0)])
        self.assertEqual(detect_chamfers(faces, ei, ea).features, [])

    def test_face_larger_than_bridged_neighbors_excluded(self) -> None:
        # a chamfer strip is smaller than the faces it bevels
        faces = [_face(0, 10.0), _face(1, 10.0), _face(2, 500.0)]
        ei, ea = _edges([(2, 0, 45.0), (2, 1, 45.0)])
        self.assertEqual(detect_chamfers(faces, ei, ea).features, [])

    def test_non_planar_candidate_excluded(self) -> None:
        faces = [_face(0, 500.0), _face(1, 500.0), _face(2, 20.0, surface_type="cylinder")]
        ei, ea = _edges([(2, 0, 45.0), (2, 1, 45.0)])
        self.assertEqual(detect_chamfers(faces, ei, ea).features, [])

    def test_excluded_faces_skipped(self) -> None:
        faces = self._two_big_faces()
        ei, ea = _edges([(2, 0, 45.0), (2, 1, 45.0)])
        self.assertEqual(detect_chamfers(faces, ei, ea, exclude_faces=frozenset({2})).features, [])


if __name__ == "__main__":
    unittest.main()
