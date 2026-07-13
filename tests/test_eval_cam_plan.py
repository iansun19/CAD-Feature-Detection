"""Unit tests for scripts/eval_cam_plan.py."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from schema.cam_plan_schema import load_cam_plan  # noqa: E402
from scripts.eval_cam_plan import (  # noqa: E402
    build_scorecard,
    infer_shop_operations,
    inspect_gt_schema,
    load_emitted_operations,
    load_gt_operations,
)

# The rear part's own single-setup plan (96260B_rear and 96260B_front are two
# separate parts); the shop program below is the rear part's.
PLAN_PATH = Path(ROOT) / "examples" / "cam_plan_96260B_rear.json"
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
    def test_inferred_shop_ops_no_coaxial_bore_after_reclassification(self) -> None:
        # On the re-baselined 44-node graph, features 14/15 are classified as
        # pockets (filleted_pocket / open_pocket), not a coaxial hole stack, so
        # no ("14", "15") coaxial bore is inferred from the feature graph.
        import yaml

        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        with GT_REAR_PATH.open("rb") as fh:
            gt = yaml.safe_load(fh.read().decode("cp1252", errors="replace"))
        ops = infer_shop_operations(gt, graph, part_id="96260B_rear")
        bore_ops = [op for op in ops if op.feature_refs == ("14", "15")]
        self.assertEqual(bore_ops, [])

    @unittest.skipUnless(PLAN_PATH.is_file(), "cam plan example missing")
    def test_emitted_ops_load_from_plan(self) -> None:
        plan = load_cam_plan(PLAN_PATH)
        emitted = load_emitted_operations(plan)
        self.assertEqual(len(emitted), len(plan.operations))
        # Feature 15 is an open_pocket on the re-baselined plan; its standalone
        # op is the 2D finishing contour, not a bore.
        feat15 = next(op for op in emitted if op.feature_refs == ("15",))
        self.assertEqual(feat15.operation_type, "finish_contour")


@unittest.skipUnless(
    PLAN_PATH.is_file() and GT_REAR_PATH.is_file() and GRAPH_PATH.is_file(),
    "96260B eval fixtures missing",
)
class TestFullScorecard(unittest.TestCase):
    def test_inferred_gt_has_no_coaxial_bore_after_reclassification(self) -> None:
        # Re-baselined 44-node graph: features 14/15 are pockets, so the inferred
        # GT has no ("14", "15") coaxial bore and the emitted plan has none either.
        import yaml

        plan = load_cam_plan(PLAN_PATH)
        with GT_REAR_PATH.open("rb") as fh:
            gt = yaml.safe_load(fh.read().decode("cp1252", errors="replace"))
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        scorecard = build_scorecard(plan, gt, GT_REAR_PATH, graph)

        self.assertEqual(scorecard["counts"]["gt_operations"], 30)
        self.assertLess(scorecard["counts"]["emitted_operations"], 30)
        coaxial = [
            op
            for op in scorecard["emitted_ops"]
            if op.feature_refs == ("14", "15")
        ]
        self.assertEqual(coaxial, [])


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

        # The shop program is the REAR part's own program: 19 ops in fixturing #1
        # plus a 1-op facing flip in fixturing #2 = 20. (Both belong to the rear
        # part; neither is the separate FRONT part.)
        self.assertEqual(aggregate["op_count"]["shop"], 20)
        # 96260B_rear is planned independently as a single-setup part: 7 ops,
        # including its own material removal and the facing pass on the real -Y
        # stock flat (feat 17). It is NOT merged with the FRONT part (a different
        # part). The planner emits per-feature ops; the shop batches by
        # tool+strategy, so 7 emitted vs 20 shop is expected divergence, not a gap.
        self.assertEqual(aggregate["op_count"]["emitted"], 7)
        self.assertIn("roughing", aggregate["strategy"]["covered"])
        self.assertIn("finishing_floor", aggregate["strategy"]["covered"])
        self.assertIn("finishing", aggregate["strategy"]["covered"])
        self.assertIn("Wall", aggregate["feature_category"]["covered"])
        self.assertIn("Contour Surface", aggregate["feature_category"]["covered"])
        # The rear plan does not machine the hole-family categories, so the rear
        # scorecard reports them as missed.
        self.assertEqual(
            sorted(aggregate["feature_category"]["missed"]),
            ["Filleted Blind Hole", "Through Hole"],
        )
        self.assertLess(aggregate["op_count"]["emitted"], 30)


if __name__ == "__main__":
    unittest.main()
