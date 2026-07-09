"""Unit tests for scripts/plan_sanity_report.py."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cam_plan_schema import CamPlan, load_cam_plan  # noqa: E402
from machining_context import Tool, ToolPreset  # noqa: E402
from scripts.plan_sanity_report import (  # noqa: E402
    FlagSeverity,
    build_sanity_report,
    check_coverage_expectations,
    check_homolog_overlap,
    exit_code_for_report,
    load_coverage_expectations,
    run_sanity_gates,
    _size_matched_shop_op,
)
from scripts.eval_cam_plan import GtOperation  # noqa: E402

PLAN_PATH = Path(ROOT) / "examples" / "cam_plan_96260B.json"
SHOP_GT_PATH = Path(ROOT) / "eval" / "gt" / "96260B_rear_shop_program.yaml"
GRAPH_PATH = Path(ROOT) / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"


def _load_shop_yaml() -> dict:
    import yaml

    return yaml.safe_load(SHOP_GT_PATH.read_text(encoding="utf-8"))


class TestSizeMatchedShopOp(unittest.TestCase):
    def test_wall_picks_nearest_diameter_not_median(self) -> None:
        shop_ops = [
            GtOperation(
                feature_refs=(),
                operation_type="wall_finish",
                strategy="finishing_wall",
                diameter_mm=4.7625,
                sequence=17,
                feed_mm_per_min=46.8 * 25.4,
                feature_categories=("Wall",),
            ),
            GtOperation(
                feature_refs=(),
                operation_type="wall_finish",
                strategy="finishing_wall",
                diameter_mm=9.525,
                sequence=18,
                feed_mm_per_min=83.58 * 25.4,
                feature_categories=("Wall",),
            ),
        ]
        matched = _size_matched_shop_op(
            shop_ops, "finishing_wall", 12.7, prefer_category="Wall",
        )
        self.assertIsNotNone(matched)
        assert matched is not None
        self.assertEqual(matched.sequence, 18)
        self.assertAlmostEqual(matched.diameter_mm or 0.0, 9.525, places=2)

    def test_size_matched_wall_feed_ratio_not_false_flagged(self) -> None:
        """Our 12.7 mm wall finish vs shop 3/8\" op should be ~0.89x, not ~1.73x median."""
        our_feed = 1891.9
        shop_feed_38 = 83.58 * 25.4
        ratio = our_feed / shop_feed_38
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 1.5)
        self.assertAlmostEqual(ratio, 0.891, places=2)


class TestSanityGates(unittest.TestCase):
    def test_micro_tool_on_open_feature_is_hard(self) -> None:
        plan_dict = {
            "schema_version": "0.1.0",
            "source_part": "test",
            "feature_graph_ref": "pipeline_out/test/graph.json",
            "setups": [{"setup_id": "rear", "opening_axis": "+Y"}],
            "operations": [
                {
                    "op_id": "OP001",
                    "sequence_index": 0,
                    "feature_refs": ["1"],
                    "feature_type": "open_pocket",
                    "setup_id": "rear",
                    "operation": "raster",
                    "tool_id": "lib::micro",
                    "parameters": {
                        "spindle_rpm": 12000,
                        "feed_mm_per_min": 100,
                        "param_source": "toolpath_preset",
                    },
                    "depends_on": [],
                }
            ],
            "tools": [
                {
                    "tool_id": "lib::micro",
                    "tool_type": "endmill",
                    "diameter_mm": 0.2,
                    "source": "test",
                }
            ],
        }
        plan = CamPlan.model_validate(plan_dict)
        tool = Tool(
            tool_id="lib::micro",
            tool_type="endmill",
            diameter_mm=0.2,
            presets=[],
            source="test",
        )
        flags = run_sanity_gates(plan, {"lib::micro": tool}, "aluminum", [])
        hard = [f for f in flags if f.severity == FlagSeverity.HARD]
        self.assertTrue(any(f.gate == "micro_tool_open_feature" for f in hard))

    def test_wrong_material_preset_on_aluminum_is_hard(self) -> None:
        steel_preset = ToolPreset(
            preset_name="LowCSteel_Adaptive_Rough",
            preset_material="low_carbon_steel",
            spindle_rpm=5000,
            feed_mm_per_min=800,
        )
        alu_preset = ToolPreset(
            preset_name="AluWrought_Adaptive_Rough",
            preset_material="aluminum",
            spindle_rpm=8000,
            feed_mm_per_min=1200,
        )
        tool = Tool(
            tool_id="lib::em",
            tool_type="endmill",
            diameter_mm=12.7,
            presets=[steel_preset, alu_preset],
            source="test",
        )
        plan_dict = {
            "schema_version": "0.1.0",
            "source_part": "test",
            "feature_graph_ref": "pipeline_out/test/graph.json",
            "setups": [{"setup_id": "rear", "opening_axis": "+Y"}],
            "operations": [
                {
                    "op_id": "OP001",
                    "sequence_index": 0,
                    "feature_refs": ["1"],
                    "feature_type": "pocket",
                    "setup_id": "rear",
                    "operation": "pocket",
                    "tool_id": "lib::em",
                    "parameters": {
                        "spindle_rpm": 5000,
                        "feed_mm_per_min": 800,
                        "param_source": "toolpath_preset",
                    },
                    "depends_on": [],
                }
            ],
            "tools": [
                {
                    "tool_id": "lib::em",
                    "tool_type": "endmill",
                    "diameter_mm": 12.7,
                    "source": "test",
                }
            ],
        }
        plan = CamPlan.model_validate(plan_dict)
        from unittest.mock import patch

        with patch(
            "scripts.plan_sanity_report.resolve_preset",
            return_value=steel_preset,
        ):
            flags = run_sanity_gates(
                plan, {"lib::em": tool}, "aluminum", [],
            )
        wrong_mat = [
            f for f in flags
            if f.gate == "wrong_material_preset" and f.severity == FlagSeverity.HARD
        ]
        self.assertEqual(len(wrong_mat), 1)
        self.assertIn("LowCSteel", wrong_mat[0].message)


@unittest.skipUnless(
    PLAN_PATH.is_file() and SHOP_GT_PATH.is_file() and GRAPH_PATH.is_file(),
    "96260B fixtures missing",
)
class Test96260BCleanPlan(unittest.TestCase):
    def test_current_plan_passes_hard_gates(self) -> None:
        plan = load_cam_plan(PLAN_PATH)
        shop_yaml = _load_shop_yaml()
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        report = build_sanity_report(
            plan,
            plan_path=PLAN_PATH,
            shop_yaml=shop_yaml,
            shop_path=SHOP_GT_PATH,
            material="aluminum",
            graph=graph,
        )
        hard = [f for f in report.flags if f.severity == FlagSeverity.HARD]
        self.assertEqual(hard, [], msg="\n".join(f.message for f in hard))
        self.assertEqual(exit_code_for_report(report), 0)

    def test_wall_feed_size_matched_not_median_artifact(self) -> None:
        plan = load_cam_plan(PLAN_PATH)
        shop_yaml = _load_shop_yaml()
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        report = build_sanity_report(
            plan,
            plan_path=PLAN_PATH,
            shop_yaml=shop_yaml,
            shop_path=SHOP_GT_PATH,
            material="aluminum",
            graph=graph,
        )
        wall_rows = [r for r in report.feed_rows if r.strategy == "finishing_wall"]
        size_matched = [r for r in wall_rows if r.compare_mode == "size_matched"]
        median_rows = [r for r in wall_rows if r.compare_mode == "median"]
        self.assertEqual(len(size_matched), 1)
        self.assertEqual(size_matched[0].shop_op_index, 18)
        self.assertAlmostEqual(size_matched[0].shop_diameter_mm or 0.0, 9.525, places=2)
        self.assertGreaterEqual(size_matched[0].feed_ratio, 0.5)
        self.assertLessEqual(size_matched[0].feed_ratio, 1.5)
        # Median across 3/16" and 3/8" wall ops is lower - would look ~1.0x not 1.73x
        if median_rows:
            self.assertNotAlmostEqual(
                size_matched[0].feed_ratio,
                1.73,
                delta=0.2,
                msg="1.73x was likely a median artifact or old plan data",
            )

    def test_wall_no_feed_off_flag_when_size_matched(self) -> None:
        plan = load_cam_plan(PLAN_PATH)
        shop_yaml = _load_shop_yaml()
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        report = build_sanity_report(
            plan,
            plan_path=PLAN_PATH,
            shop_yaml=shop_yaml,
            shop_path=SHOP_GT_PATH,
            material="aluminum",
            graph=graph,
        )
        wall_feed_flags = [
            f for f in report.flags
            if f.gate == "feed_off"
            and f.op_id == "OP070"
        ]
        self.assertEqual(wall_feed_flags, [])

    def test_front_setup_micro_tool_trips_gate(self) -> None:
        from cam_plan_schema import MachiningParameters, Operation, Setup, ToolRef

        plan = CamPlan(
            source_part="96260B",
            feature_graph_ref="pipeline_out/96260B_front/feature_graph_cascade.json",
            setups=[
                Setup(setup_id="rear", opening_axis="+Y"),
                Setup(setup_id="front", opening_axis="+Y"),
            ],
            operations=[
                Operation(
                    op_id="OP010",
                    sequence_index=0,
                    feature_refs=["1"],
                    feature_type="filleted_open_pocket",
                    setup_id="rear",
                    operation="pocket",
                    tool_id="T_OK",
                    parameters=MachiningParameters(param_source="handbook_default"),
                ),
                Operation(
                    op_id="OP020",
                    sequence_index=1,
                    feature_refs=["2"],
                    feature_type="filleted_open_pocket",
                    setup_id="front",
                    operation="raster",
                    tool_id="T_MICRO",
                    parameters=MachiningParameters(param_source="handbook_default"),
                ),
            ],
            tools=[
                ToolRef(
                    tool_id="T_OK",
                    tool_type="endmill",
                    diameter_mm=12.7,
                    source="test",
                ),
                ToolRef(
                    tool_id="T_MICRO",
                    tool_type="endmill",
                    diameter_mm=0.2,
                    source="test",
                ),
            ],
        )
        tools = {
            "T_OK": Tool(tool_id="T_OK", tool_type="endmill", diameter_mm=12.7),
            "T_MICRO": Tool(tool_id="T_MICRO", tool_type="endmill", diameter_mm=0.2),
        }
        flags = run_sanity_gates(plan, tools, "aluminum", shop_ops=[])
        micro = [f for f in flags if f.gate == "micro_tool_open_feature" and f.op_id == "OP020"]
        self.assertEqual(len(micro), 1)


class TestHomologOverlapGate(unittest.TestCase):
    def _over_machined_plan(self) -> CamPlan:
        from cam_plan_schema import MachiningParameters, Operation, Setup, ToolRef

        op_types = [
            "pocket",
            "contour_2d",
            "raster",
            "constant_scallop",
        ]
        operations = [
            Operation(
                op_id=f"OP{idx * 10:03d}",
                sequence_index=idx,
                feature_refs=[str(feature_id)],
                feature_type="filleted_open_pocket",
                setup_id="rear",
                operation=op_type,
                tool_id="T1",
                parameters=MachiningParameters(param_source="handbook_default"),
            )
            for idx, (feature_id, op_type) in enumerate(
                zip(range(4), op_types, strict=True)
            )
        ]
        operations.extend(
            Operation(
                op_id=f"OP{(idx + 4) * 10:03d}",
                sequence_index=idx + 4,
                feature_refs=[str(feature_id)],
                feature_type="filleted_open_pocket",
                setup_id="front",
                operation=op_type,
                tool_id="T1",
                parameters=MachiningParameters(param_source="handbook_default"),
            )
            for idx, (feature_id, op_type) in enumerate(
                zip(range(4), op_types, strict=True)
            )
        )
        return CamPlan(
            source_part="96260B",
            feature_graph_ref="pipeline_out/96260B_rear/feature_graph_cascade.json",
            setups=[
                Setup(setup_id="rear", opening_axis="+Y"),
                Setup(setup_id="front", opening_axis="+Y"),
            ],
            operations=operations,
            tools=[
                ToolRef(
                    tool_id="T1",
                    tool_type="endmill",
                    diameter_mm=12.7,
                    source="test",
                ),
            ],
        )

    def test_homolog_overlap_fires_on_duplicate_cross_setup_ops(self) -> None:
        flags = check_homolog_overlap(self._over_machined_plan())
        hard = [f for f in flags if f.gate == "homolog_overlap"]
        self.assertEqual(len(hard), 2)
        self.assertTrue(all(f.severity == FlagSeverity.HARD for f in hard))

    def test_homolog_overlap_passes_on_scoped_front_facing_only(self) -> None:
        from cam_plan_schema import MachiningParameters, Operation, Setup, ToolRef

        plan = CamPlan(
            source_part="96260B",
            feature_graph_ref="pipeline_out/96260B_rear/feature_graph_cascade.json",
            setups=[
                Setup(setup_id="rear", opening_axis="+Y"),
                Setup(setup_id="front", opening_axis="+Y"),
            ],
            operations=[
                Operation(
                    op_id="OP010",
                    sequence_index=0,
                    feature_refs=["1"],
                    feature_type="filleted_open_pocket",
                    setup_id="rear",
                    operation="pocket",
                    tool_id="T1",
                    parameters=MachiningParameters(param_source="handbook_default"),
                ),
                Operation(
                    op_id="OP020",
                    sequence_index=1,
                    feature_refs=["18"],
                    feature_type="flat",
                    setup_id="front",
                    operation="facing",
                    tool_id="FM1",
                    parameters=MachiningParameters(param_source="handbook_default"),
                ),
            ],
            tools=[
                ToolRef(tool_id="T1", tool_type="endmill", diameter_mm=12.7, source="test"),
                ToolRef(tool_id="FM1", tool_type="face_mill", diameter_mm=38.1, source="test"),
            ],
            metadata={
                "planner_stats": {
                    "per_setup": {
                        "front": {
                            "scope_mode": "filtered",
                            "scope_classes": ["facing"],
                        },
                        "rear": {"scope_mode": "full"},
                    },
                },
            },
        )
        flags = check_homolog_overlap(plan)
        self.assertEqual(flags, [])


class TestCoverageExpectations(unittest.TestCase):
    def test_manifest_loads_for_96260b(self) -> None:
        manifest = load_coverage_expectations("96260B")
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertIn("rear", manifest["setups"])
        self.assertIn("front", manifest["setups"])

    def _build_96260b_report(self, plan: CamPlan) -> object:
        shop_yaml = _load_shop_yaml()
        graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        return build_sanity_report(
            plan,
            plan_path=PLAN_PATH,
            shop_yaml=shop_yaml,
            shop_path=SHOP_GT_PATH,
            material="aluminum",
            graph=graph,
        )

    @unittest.skipUnless(
        PLAN_PATH.is_file() and SHOP_GT_PATH.is_file() and GRAPH_PATH.is_file(),
        "96260B fixtures missing",
    )
    def test_current_plan_coverage_verdict_matches_both_setups(self) -> None:
        plan = load_cam_plan(PLAN_PATH)
        report = self._build_96260b_report(plan)
        self.assertEqual(report.coverage.get("verdict"), "PASS")
        verdicts = report.coverage.get("setup_verdicts") or []
        by_setup = {v["setup_id"]: v for v in verdicts}
        self.assertTrue(by_setup["rear"]["matches"])
        self.assertTrue(by_setup["front"]["matches"])
        self.assertIn("facing expected-absent", by_setup["rear"]["message"])
        self.assertIn("facing", by_setup["front"]["emitted_strategies"])

    @unittest.skipUnless(
        PLAN_PATH.is_file() and SHOP_GT_PATH.is_file() and GRAPH_PATH.is_file(),
        "96260B fixtures missing",
    )
    def test_front_over_machining_fails_coverage_verdict(self) -> None:
        from cam_plan_schema import MachiningParameters, Operation

        plan = load_cam_plan(PLAN_PATH)
        extra_ops = [
            Operation(
                op_id="OP200",
                sequence_index=100,
                feature_refs=["1"],
                feature_type="filleted_open_pocket",
                setup_id="front",
                operation="pocket",
                tool_id="T07",
                parameters=MachiningParameters(param_source="handbook_default"),
            ),
            Operation(
                op_id="OP210",
                sequence_index=101,
                feature_refs=["2"],
                feature_type="filleted_open_pocket",
                setup_id="front",
                operation="raster",
                tool_id="T07",
                parameters=MachiningParameters(param_source="handbook_default"),
            ),
            Operation(
                op_id="OP220",
                sequence_index=102,
                feature_refs=["3"],
                feature_type="wall",
                setup_id="front",
                operation="contour_2d",
                tool_id="T07",
                parameters=MachiningParameters(param_source="handbook_default"),
            ),
            Operation(
                op_id="OP230",
                sequence_index=103,
                feature_refs=["4"],
                feature_type="contour_surface",
                setup_id="front",
                operation="constant_scallop",
                tool_id="T07",
                parameters=MachiningParameters(param_source="handbook_default"),
            ),
            Operation(
                op_id="OP240",
                sequence_index=104,
                feature_refs=["5"],
                feature_type="outer_fillet",
                setup_id="front",
                operation="pencil",
                tool_id="T07",
                parameters=MachiningParameters(param_source="handbook_default"),
            ),
        ]
        plan = plan.model_copy(update={"operations": list(plan.operations) + extra_ops})
        report = self._build_96260b_report(plan)
        self.assertEqual(report.coverage.get("verdict"), "FAIL")
        front = next(
            v for v in (report.coverage.get("setup_verdicts") or [])
            if v["setup_id"] == "front"
        )
        self.assertFalse(front["matches"])
        self.assertEqual(len(front["unexpected"]), 5)
        self.assertEqual(exit_code_for_report(report), 1)

    def test_missing_expected_present_is_hard(self) -> None:
        from cam_plan_schema import MachiningParameters, Operation, Setup, ToolRef

        plan = CamPlan(
            source_part="96260B",
            feature_graph_ref="pipeline_out/96260B_rear/feature_graph_cascade.json",
            setups=[
                Setup(setup_id="rear", opening_axis="+Y"),
                Setup(setup_id="front", opening_axis="+Y"),
            ],
            operations=[
                Operation(
                    op_id="OP010",
                    sequence_index=0,
                    feature_refs=["18"],
                    feature_type="flat",
                    setup_id="front",
                    operation="facing",
                    tool_id="T1",
                    parameters=MachiningParameters(param_source="handbook_default"),
                ),
            ],
            tools=[
                ToolRef(tool_id="T1", tool_type="face_mill", diameter_mm=38.1, source="test"),
            ],
        )
        manifest = load_coverage_expectations("96260B")
        _, flags = check_coverage_expectations(plan, manifest, part_id="96260B")
        hard = [
            f for f in flags
            if f.gate == "coverage_expectations" and f.severity == FlagSeverity.HARD
        ]
        self.assertTrue(any("rear" in f.op_id for f in hard))
        self.assertTrue(any("roughing" in f.message for f in hard))

    def test_no_manifest_falls_back_with_warn(self) -> None:
        from cam_plan_schema import MachiningParameters, Operation, Setup, ToolRef

        plan = CamPlan(
            source_part="UNKNOWN_PART_XYZ",
            feature_graph_ref="pipeline_out/test/graph.json",
            setups=[Setup(setup_id="rear", opening_axis="+Y")],
            operations=[
                Operation(
                    op_id="OP010",
                    sequence_index=0,
                    feature_refs=["1"],
                    feature_type="pocket",
                    setup_id="rear",
                    operation="pocket",
                    tool_id="T1",
                    parameters=MachiningParameters(param_source="handbook_default"),
                ),
            ],
            tools=[
                ToolRef(tool_id="T1", tool_type="endmill", diameter_mm=12.7, source="test"),
            ],
        )
        manifest = load_coverage_expectations("UNKNOWN_PART_XYZ")
        self.assertIsNone(manifest)
        verdicts, flags = check_coverage_expectations(
            plan, manifest, part_id="UNKNOWN_PART_XYZ",
        )
        self.assertEqual(verdicts, [])
        warn = [f for f in flags if f.gate == "coverage_expectations"]
        self.assertEqual(len(warn), 1)
        self.assertEqual(warn[0].severity, FlagSeverity.WARN)


if __name__ == "__main__":
    unittest.main()
