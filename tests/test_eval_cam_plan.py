"""Unit tests for scripts/eval_cam_plan.py."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cam_plan_schema import load_cam_plan  # noqa: E402
from scripts.eval_cam_plan import (  # noqa: E402
    build_scorecard,
    infer_shop_operations,
    inspect_gt_schema,
    load_emitted_operations,
    load_gt_operations,
)

PLAN_PATH = Path(ROOT) / "examples" / "cam_plan_96260B.json"
GT_REAR_PATH = Path(ROOT) / "eval" / "gt" / "96260B_rear.yaml"
SHOP_GT_PATH = Path(ROOT) / "eval" / "gt" / "96260B_rear_shop_program.yaml"
GRAPH_PATH = Path(ROOT) / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"


class TestGtSchemaInspection(unittest.TestCase):
    def test_rear_gt_is_counts_features_only(self) -> None:
        import yaml

        with GT_REAR_PATH.open("rb") as fh:
            raw = fh.read().decode("cp1252", errors="replace")
        gt = yaml.safe_load(raw)
        report = inspect_gt_schema(gt, GT_REAR_PATH)
        self.assertTrue(report.has_counts)
        self.assertTrue(report.has_features)
        self.assertFalse(report.has_operations)
        self.assertEqual(report.operation_source, "inferred_from_features_graph")


class TestInferenceAndMatching(unittest.TestCase):
    @unittest.skipUnless(GRAPH_PATH.is_file(), "rear cascade graph missing")
    @unittest.skipUnless(GT_REAR_PATH.is_file(), "rear GT missing")
    def test_inferred_shop_ops_cover_coaxial_bore(self) -> None:
        import yaml

        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        with GT_REAR_PATH.open("rb") as fh:
            gt = yaml.safe_load(fh.read().decode("cp1252", errors="replace"))
        ops = infer_shop_operations(gt, graph, part_id="96260B_rear")
        bore_ops = [op for op in ops if op.feature_refs == ("14", "15")]
        self.assertEqual(len(bore_ops), 1)
        self.assertEqual(bore_ops[0].operation_type, "bore")

    @unittest.skipUnless(PLAN_PATH.is_file(), "cam plan example missing")
    def test_emitted_ops_load_from_plan(self) -> None:
        plan = load_cam_plan(PLAN_PATH)
        emitted = load_emitted_operations(plan)
        self.assertEqual(len(emitted), len(plan.operations))
        bore = next(op for op in emitted if op.feature_refs == ("14", "15"))
        self.assertEqual(bore.operation_type, "bore")


@unittest.skipUnless(
    PLAN_PATH.is_file() and GT_REAR_PATH.is_file() and GRAPH_PATH.is_file(),
    "96260B eval fixtures missing",
)
class TestFullScorecard(unittest.TestCase):
    def test_inferred_gt_coaxial_bore_now_matches(self) -> None:
        import yaml

        plan = load_cam_plan(PLAN_PATH)
        with GT_REAR_PATH.open("rb") as fh:
            gt = yaml.safe_load(fh.read().decode("cp1252", errors="replace"))
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        scorecard = build_scorecard(plan, gt, GT_REAR_PATH, graph)

        self.assertEqual(scorecard["counts"]["gt_operations"], 31)
        self.assertLess(scorecard["counts"]["emitted_operations"], 31)
        coaxial = [
            op
            for op in scorecard["emitted_ops"]
            if op.feature_refs == ("14", "15")
        ]
        self.assertEqual(len(coaxial), 1)
        self.assertEqual(coaxial[0].operation_type, "bore")
        mismatches = [
            m for m in scorecard["type_mismatches"] if m.gt.feature_refs == ("14", "15")
        ]
        self.assertEqual(mismatches, [])


class TestYamlOperationsParsing(unittest.TestCase):
    def test_load_operations_from_yaml_block(self) -> None:
        gt = {
            "part_id": "test",
            "operations": [
                {
                    "feature_refs": ["1", "2"],
                    "operation_type": "bore",
                    "tool": {"tool_type": "endmill", "diameter_mm": 12.0},
                    "parameters": {"spindle_rpm": 8000, "feed_mm_per_min": 1200},
                    "sequence": 1,
                },
            ],
        }
        ops, source = load_gt_operations(gt, {"nodes": []}, part_id="test")
        self.assertEqual(source, "gt_yaml")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].operation_type, "bore")
        self.assertEqual(ops[0].tool_type, "endmill")
        self.assertEqual(ops[0].spindle_rpm, 8000.0)

    @unittest.skipUnless(SHOP_GT_PATH.is_file(), "shop program GT missing")
    def test_load_shop_program_operations(self) -> None:
        import yaml

        gt = yaml.safe_load(SHOP_GT_PATH.read_text(encoding="utf-8"))
        ops, source = load_gt_operations(gt, {"nodes": []}, part_id=None)
        self.assertEqual(source, "shop_program")
        self.assertEqual(len(ops), 20)
        self.assertEqual(ops[0].strategy, "roughing")
        self.assertAlmostEqual(ops[0].diameter_mm or 0.0, 0.375 * 25.4, places=2)
        self.assertEqual(ops[0].feature_categories[0], "Face")


@unittest.skipUnless(
    PLAN_PATH.is_file() and SHOP_GT_PATH.is_file() and GRAPH_PATH.is_file(),
    "96260B shop-program eval fixtures missing",
)
class TestShopProgramAggregateScorecard(unittest.TestCase):
    def test_aggregate_scorecard_covers_shop_strategies(self) -> None:
        import yaml

        plan = load_cam_plan(PLAN_PATH)
        gt = yaml.safe_load(SHOP_GT_PATH.read_text(encoding="utf-8"))
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        scorecard = build_scorecard(plan, gt, SHOP_GT_PATH, graph)
        aggregate = scorecard["aggregate"]
        self.assertIsNotNone(aggregate)
        assert aggregate is not None

        self.assertEqual(aggregate["op_count"]["shop"], 20)
        self.assertGreaterEqual(aggregate["op_count"]["emitted"], 8)
        self.assertLessEqual(aggregate["op_count"]["emitted"], 20)
        self.assertGreater(aggregate["op_count"]["emitted"], 10)
        self.assertIn("roughing", aggregate["strategy"]["covered"])
        self.assertIn("finishing_wall", aggregate["strategy"]["covered"])
        self.assertIn("finishing", aggregate["strategy"]["covered"])
        self.assertIn("Wall", aggregate["feature_category"]["covered"])
        self.assertIn("Contour Surface", aggregate["feature_category"]["covered"])
        self.assertEqual(len(aggregate["feature_category"]["missed"]), 0)
        self.assertLess(aggregate["op_count"]["emitted"], 30)


if __name__ == "__main__":
    unittest.main()
