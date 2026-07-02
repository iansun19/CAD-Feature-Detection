"""Unit tests for feature_params pure logic (no pythonocc required)."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from feature_params import (  # noqa: E402
    FaceGeom,
    analytic_surfaces,
    bbox_params,
    classify_planar_roles,
    count_planar_walls,
    count_walls,
    cylindrical_radii,
    pick_floor_plane,
    span_along_direction,
)


def _plane(index, area, xyz, normal):
    return FaceGeom(
        index=index, surface_type="plane", area=area,
        centroid=np.array(xyz, dtype=np.float64),
        normal=np.array(normal, dtype=np.float64),
    )


def _cylinder(index, radius, area, xyz, axis=(0, 0, 1)):
    return FaceGeom(
        index=index, surface_type="cylinder", area=area,
        centroid=np.array(xyz, dtype=np.float64),
        normal=np.array([1, 0, 0], dtype=np.float64),
        radius=radius,
        axis=np.array(axis, dtype=np.float64),
    )


class TestFeatureParams(unittest.TestCase):
    def test_bbox_params(self):
        cents = [np.array([0, 0, 0]), np.array([10, 20, 5])]
        b = bbox_params(cents)
        self.assertAlmostEqual(b["bbox_size_x"], 10.0)
        self.assertAlmostEqual(b["bbox_depth"], 5.0)

    def test_count_planar_walls_legacy(self):
        planes = [_plane(0, 100, [0, 0, 0], [0, 0, 1]),
                  _plane(1, 10, [1, 0, 0], [1, 0, 0]),
                  _plane(2, 10, [0, 1, 0], [0, 1, 0]),
                  _plane(3, 10, [0, 0, 1], [0, 0, 1])]
        self.assertEqual(count_planar_walls(planes), 3)

    def test_count_walls_pocket_with_cylinders(self):
        floor = _plane(0, 50, [0, 0, 0], [0, 0, 1])
        opening = _plane(1, 40, [0, 0, 5], [0, 0, 1])
        cyl_a = _cylinder(2, 2.0, 10, [1, 0, 2])
        cyl_b = _cylinder(3, 2.0, 10, [-1, 0, 2])
        info = count_walls([floor, opening], [cyl_a, cyl_b], floor)
        self.assertEqual(info["n_planar_walls"], 0)
        self.assertEqual(info["n_cylindrical_walls"], 2)
        self.assertEqual(info["n_opening_faces"], 1)
        self.assertEqual(info["n_walls"], 2)
        self.assertEqual(info["floor_face_index"], 0)

    def test_classify_planar_walls(self):
        floor = _plane(0, 100, [0, 0, 0], [0, 0, 1])
        wall_x = _plane(1, 20, [1, 0, 0], [1, 0, 0])
        wall_y = _plane(2, 20, [0, 1, 0], [0, 1, 0])
        opening = _plane(3, 30, [0, 0, 5], [0, 0, 1])
        walls, openings, _ = classify_planar_roles(
            [floor, wall_x, wall_y, opening], floor,
        )
        self.assertEqual(len(walls), 2)
        self.assertEqual(len(openings), 1)

    def test_span_along_direction(self):
        cents = [np.array([0, 0, 0]), np.array([0, 0, 8])]
        self.assertAlmostEqual(span_along_direction(cents, np.array([0, 0, 1])), 8.0)

    def test_cylindrical_radii(self):
        faces = [_cylinder(5, 3.5, 12, [0, 0, 0]), _plane(0, 1, [0, 0, 0], [0, 0, 1])]
        rows = cylindrical_radii(faces)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["radius"], 3.5)
        self.assertAlmostEqual(rows[0]["diameter"], 7.0)
        self.assertEqual(rows[0]["face_index"], 5)

    def test_analytic_surfaces_includes_planes(self):
        faces = [_plane(0, 5, [0, 0, 0], [0, 0, 1])]
        surf = analytic_surfaces(faces)
        self.assertEqual(len(surf), 1)
        self.assertIn("normal", surf[0])


if __name__ == "__main__":
    unittest.main()
