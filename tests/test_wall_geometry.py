"""Tests for wall_geometry post-pass enrichment."""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from OCC.Core.STEPControl import STEPControl_Reader  # noqa: F401
    HAS_OCC = True
except ImportError:
    HAS_OCC = False

from cascade.wall_geometry import (  # noqa: E402
    WALL_CLEARANCE_MAX_MM,
    WALL_CLEARANCE_MIN_MM,
    WALL_DEPTH_MAX_MM,
    WALL_DEPTH_MIN_MM,
    WALL_PARAM_KEYS,
    compute_wall_geometry,
    enrich_graph_wall_geometry,
    verify_wall_geometry,
)

REAR_GRAPH = Path(ROOT) / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
REAR_STEP = Path(ROOT) / "fixtures/step/96260B_rear.stp"
REAR_NPZ = Path(ROOT) / "pipeline_out" / "96260B_plate" / "graph.npz"


class TestWallGeometryPure(unittest.TestCase):
    def test_wall_param_keys(self) -> None:
        self.assertEqual(
            WALL_PARAM_KEYS,
            ("wall_depth_mm", "wall_min_clearance_mm", "wall_lateral_extent_mm"),
        )

    def test_empty_face_ids_returns_nulls_and_warning(self) -> None:
        result = compute_wall_geometry([], mock.Mock(), feature_id=99)
        self.assertIsNone(result.wall_depth_mm)
        self.assertIsNone(result.wall_min_clearance_mm)
        self.assertIsNone(result.wall_lateral_extent_mm)
        self.assertTrue(any("empty face_ids" in w for w in result.warnings))

    def test_verify_report_shape(self) -> None:
        graph = {
            "nodes": [
                {
                    "feature_id": 1,
                    "class_name": "wall",
                    "face_ids": [3],
                    "params": {
                        "wall_depth_mm": 6.0,
                        "wall_min_clearance_mm": 3.81,
                        "wall_lateral_extent_mm": 2.5,
                    },
                },
            ],
        }
        report = verify_wall_geometry(graph)
        self.assertEqual(report["separation"]["n_walls"], 1)
        self.assertIn("wall_depth_mm", report["stats"])


@unittest.skipUnless(HAS_OCC, "pythonocc-core not installed")
@unittest.skipUnless(REAR_GRAPH.is_file(), "96260B rear cascade graph missing")
@unittest.skipUnless(REAR_STEP.is_file(), "96260B rear STEP missing")
@unittest.skipUnless(REAR_NPZ.is_file(), "96260B plate graph.npz missing")
class TestWallGeometry96260B(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(REAR_GRAPH, encoding="utf-8") as handle:
            cls.original_graph = json.load(handle)
        cls.original_bytes = REAR_GRAPH.read_bytes()

    def test_enrichment_attaches_three_fields_to_all_walls(self) -> None:
        enriched = enrich_graph_wall_geometry(
            self.original_graph,
            REAR_STEP,
            graph_npz=REAR_NPZ,
            opening_axis=(0.0, 1.0, 0.0),
        )
        walls = [n for n in enriched["nodes"] if n.get("class_name") == "wall"]
        self.assertEqual(len(walls), 14)
        for node in walls:
            params = node["params"]
            for key in WALL_PARAM_KEYS:
                self.assertIn(key, params)

    def test_depth_and_clearance_in_sanity_band(self) -> None:
        enriched = enrich_graph_wall_geometry(
            self.original_graph,
            REAR_STEP,
            graph_npz=REAR_NPZ,
            opening_axis=(0.0, 1.0, 0.0),
        )
        for node in enriched["nodes"]:
            if node.get("class_name") != "wall":
                continue
            params = node["params"]
            depth = params["wall_depth_mm"]
            clearance = params["wall_min_clearance_mm"]
            lateral = params["wall_lateral_extent_mm"]
            if depth is not None:
                self.assertGreaterEqual(depth, WALL_DEPTH_MIN_MM)
                self.assertLessEqual(depth, WALL_DEPTH_MAX_MM)
            if clearance is not None:
                self.assertGreaterEqual(clearance, WALL_CLEARANCE_MIN_MM)
                self.assertLessEqual(clearance, WALL_CLEARANCE_MAX_MM)
            self.assertIsNotNone(lateral)
            self.assertGreater(lateral, 0.0)

    def test_original_graph_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy_path = Path(tmp) / "graph_copy.json"
            out_path = Path(tmp) / "enriched.json"
            copy_path.write_bytes(self.original_bytes)

            with open(copy_path, encoding="utf-8") as handle:
                graph = json.load(handle)

            enriched = enrich_graph_wall_geometry(
                graph,
                REAR_STEP,
                graph_npz=REAR_NPZ,
                opening_axis=(0.0, 1.0, 0.0),
            )
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(enriched, handle, indent=2)

            self.assertEqual(copy_path.read_bytes(), self.original_bytes)
            for key in WALL_PARAM_KEYS:
                self.assertNotIn(key, graph["nodes"][0].get("params", {}))

    def test_geometry_separates_shop_wall_split(self) -> None:
        enriched = enrich_graph_wall_geometry(
            self.original_graph,
            REAR_STEP,
            graph_npz=REAR_NPZ,
            opening_axis=(0.0, 1.0, 0.0),
        )
        report = verify_wall_geometry(enriched)
        self.assertEqual(report["separation"]["n_walls"], 14)
        self.assertTrue(report["separation"]["clearance_is_uniform"])
        self.assertGreater(report["separation"]["n_tight_by_lateral"], 0)
        self.assertGreater(report["separation"]["n_open_by_lateral"], 0)
        self.assertTrue(report["separation"]["geometry_explains_shop_split"])

    def test_uncomputable_depth_logs_null_not_crash(self) -> None:
        graph = copy.deepcopy(self.original_graph)
        walls = [n for n in graph["nodes"] if n.get("class_name") == "wall"]
        walls[0]["face_ids"] = [99999]

        with self.assertLogs("cascade.wall_geometry", level="WARNING") as logs:
            enriched = enrich_graph_wall_geometry(
                graph,
                REAR_STEP,
                graph_npz=REAR_NPZ,
                opening_axis=(0.0, 1.0, 0.0),
            )

        bad = next(
            n for n in enriched["nodes"]
            if n.get("class_name") == "wall" and 99999 in n.get("face_ids", [])
        )
        self.assertIsNone(bad["params"]["wall_depth_mm"])
        self.assertTrue(
            any("wall_depth_mm" in line for line in logs.output),
            logs.output,
        )


if __name__ == "__main__":
    unittest.main()
