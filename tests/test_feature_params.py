"""Unit tests for feature_params pure logic (no pythonocc required)."""
from __future__ import annotations

import copy
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brep.feature_params import (  # noqa: E402
    FaceGeom,
    REMOVED_HEURISTIC_KEYS,
    analytic_surfaces,
    bbox_params,
    classify_planar_roles,
    count_planar_walls,
    count_walls,
    cylindrical_radii,
    pick_floor_plane,
    span_along_direction,
    split_params_for_cam,
    strip_heuristic_params,
    validate_face_indices,
    DERIVED_DEBUG_KEYS,
    INDEX_EXPORT_PARAM_KEYS,
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


def _cone(index, area, xyz, semi_angle_deg=45.0, axis=(0, 0, 1)):
    return FaceGeom(
        index=index, surface_type="cone", area=area,
        centroid=np.array(xyz, dtype=np.float64),
        normal=np.array([0, 0, 1], dtype=np.float64),
        semi_angle_rad=float(np.radians(semi_angle_deg)),
        axis=np.array(axis, dtype=np.float64),
    )


class TestIndexExport(unittest.TestCase):
    def test_split_keeps_only_direct_occ_reads(self):
        faces = [
            _cone(17, 4.0, [0, 0, 0], 45.0),
            _plane(19, 1.0, [0, 1, 0], [0, 1, 0]),
        ]
        full_params = {
            "analytic_surfaces": analytic_surfaces(faces),
            "surface_counts": {"cone": 1, "plane": 1},
            "total_area": 5.0,
            "cylindrical_radii": [],
            "semi_angle_deg": 45.0,
            "bbox_depth": 1.0,
            "n_walls": 1,
            "radius": 3.5,
            "depth_along_floor_normal": 2.0,
        }
        index_params, derived = split_params_for_cam(full_params, 9, faces)
        self.assertEqual(set(index_params.keys()), INDEX_EXPORT_PARAM_KEYS)
        self.assertNotIn("bbox_depth", index_params)
        self.assertNotIn("semi_angle_deg", index_params)
        self.assertNotIn("radius", index_params)
        self.assertIn("bbox_depth", derived)
        self.assertIn("semi_angle_deg", derived)
        self.assertIn("analytic_surfaces", index_params)
        self.assertTrue(
            any("semi_angle_deg" in s for s in index_params["analytic_surfaces"]),
        )

    def test_derived_debug_keys_routed(self):
        full = {
            "analytic_surfaces": [],
            "surface_counts": {"plane": 1},
            "total_area": 1.0,
            "cylindrical_radii": [],
            "bbox_depth": 1.0,
            "semi_angle_deg": 45.0,
            "step_height": 3.0,
        }
        index_params, derived = split_params_for_cam(full, 9, [])
        for key in ("bbox_depth", "semi_angle_deg", "step_height"):
            self.assertNotIn(key, index_params)
            self.assertIn(key, derived)
            self.assertIn(key, DERIVED_DEBUG_KEYS)
        for key in ("surface_counts", "total_area", "cylindrical_radii"):
            self.assertIn(key, index_params)

    def test_strip_heuristic_params(self):
        params = {
            "analytic_surfaces": [],
            "step_height": 1.0,
            "bbox_depth": 2.0,
            "surface_counts": {"plane": 1},
        }
        stripped = strip_heuristic_params(params)
        self.assertNotIn("step_height", stripped)
        self.assertNotIn("bbox_depth", stripped)
        self.assertIn("surface_counts", stripped)
        for key in REMOVED_HEURISTIC_KEYS:
            self.assertNotIn(key, stripped)

    def test_validate_face_indices_clean(self):
        records = [_plane(i, 1.0, [0, 0, 0], [0, 0, 1]) for i in range(10)]
        out = validate_face_indices([2, 5], len(records), records=records)
        self.assertTrue(out["valid"], out["errors"])

    def test_validate_face_indices_duplicate(self):
        records = [_plane(i, 1.0, [0, 0, 0], [0, 0, 1]) for i in range(10)]
        out = validate_face_indices([2, 2], len(records), records=records)
        self.assertFalse(out["valid"])
        self.assertTrue(any("duplicate" in e for e in out["errors"]))

    def test_validate_face_indices_out_of_range(self):
        records = [_plane(i, 1.0, [0, 0, 0], [0, 0, 1]) for i in range(10)]
        out = validate_face_indices([99], len(records), records=records)
        self.assertFalse(out["valid"])
        self.assertIn(99, out["invalid_indices"])

    def test_validate_face_indices_empty(self):
        out = validate_face_indices([], 10, records=[])
        self.assertFalse(out["valid"])
        self.assertTrue(any("empty" in e for e in out["errors"]))


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


@unittest.skipUnless(
    os.path.isfile(os.path.join(ROOT, "MFCAD++_dataset", "step", "test", "29000.step")),
    "missing 29000.step",
)
class Test29000IndexExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from OCC.Core.STEPControl import STEPControl_Reader  # noqa: F401
            from brep.feature_params import HAS_OCC
        except ImportError:
            raise unittest.SkipTest("pythonocc-core not installed")
        if not HAS_OCC:
            raise unittest.SkipTest("pythonocc-core not installed")

        cls.step = os.path.join(ROOT, "MFCAD++_dataset", "step", "test", "29000.step")
        cls.graph_path = os.path.join(ROOT, "fixtures/graphs/fixtures/graphs/29000_feature_graph.json")
        if not os.path.isfile(cls.graph_path):
            raise unittest.SkipTest("missing fixtures/graphs/fixtures/graphs/29000_feature_graph.json")

        with open(cls.graph_path) as f:
            cls.base_graph = json.load(f)

        from brep.feature_params import analyze_step, enrich_graph_with_params, extract_feature_params

        cls.analyze_step = analyze_step
        cls.enrich_graph_with_params = enrich_graph_with_params
        cls.extract_feature_params = extract_feature_params
        cls.records = analyze_step(cls.step)

        cls.full_surfaces = {}
        for node in cls.base_graph["nodes"]:
            fid = int(node["feature_id"])
            face_ids = [int(i) for i in node["face_ids"]]
            full = extract_feature_params(int(node["class_id"]), face_ids, cls.records)
            cls.full_surfaces[fid] = copy.deepcopy(full.get("analytic_surfaces", []))

        cls.graph = copy.deepcopy(cls.base_graph)
        for node in cls.graph["nodes"]:
            node.pop("params", None)
            node.pop("invalid", None)
            node.pop("face_index_error", None)
        enrich_graph_with_params(cls.graph, cls.step)

    def test_no_removed_heuristic_fields_in_nodes(self):
        for node in self.graph["nodes"]:
            params = node.get("params", {})
            for key in REMOVED_HEURISTIC_KEYS:
                self.assertNotIn(
                    key, params,
                    f"feature_id={node['feature_id']} still has {key}",
                )

    def test_every_node_has_required_fields(self):
        for node in self.graph["nodes"]:
            self.assertIn("class_name", node)
            self.assertTrue(node.get("face_ids"))
            params = node.get("params", {})
            self.assertIn("analytic_surfaces", params)
            face_count = node.get("n_faces") or len(node["face_ids"])
            counts = params.get("surface_counts", {})
            if counts:
                self.assertEqual(sum(counts.values()), face_count)
            self.assertFalse(node.get("invalid"), node.get("face_index_error"))

    def test_analytic_surfaces_unchanged(self):
        for node in self.graph["nodes"]:
            fid = int(node["feature_id"])
            expected = self.full_surfaces[fid]
            actual = node["params"]["analytic_surfaces"]
            self.assertEqual(actual, expected, f"feature_id={fid} analytic_surfaces changed")

    def test_face_index_error_on_bad_indices(self):
        from brep.brep_extents import FaceIndexError, resolve_occ_faces

        with self.assertRaises(FaceIndexError):
            resolve_occ_faces(self.step, [0, 0], expected_n_faces=26)

        out = validate_face_indices([0, 0], 26, records=self.records)
        self.assertFalse(out["valid"])


if __name__ == "__main__":
    unittest.main()
