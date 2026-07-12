"""Unit tests for cam_plan_schema.py (no pythonocc required)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cam_plan_schema import (  # noqa: E402
    CamPlan,
    Operation,
    PocketAccess,
    MachiningParameters,
    Setup,
    ToolRef,
    example_cam_plan,
    load_cam_plan,
)


EXAMPLE_PATH = Path(ROOT) / "examples" / "cam_plan_example.json"


class TestCamPlanSchema(unittest.TestCase):
    def test_example_cam_plan_validates(self):
        plan = example_cam_plan()
        self.assertEqual(plan.schema_version, "0.1.0")
        self.assertEqual(len(plan.setups), 1)
        self.assertEqual(len(plan.operations), 2)
        self.assertEqual(plan.operations[1].access, PocketAccess.CLOSED)
        self.assertIsNone(plan.remaining_material)

    def test_example_json_file_validates(self):
        self.assertTrue(EXAMPLE_PATH.is_file(), f"missing {EXAMPLE_PATH}")
        plan = load_cam_plan(EXAMPLE_PATH)
        self.assertEqual(plan.operations[0].operation, "drill")
        self.assertEqual(plan.operations[1].feature_type, "filleted_pocket")

    def test_sequence_index_must_match_list_order(self):
        base = example_cam_plan()
        bad_ops = [
            Operation(
                op_id="OP010",
                sequence_index=1,
                feature_refs=["0"],
                feature_type="through_hole",
                setup_id="primary",
                operation="drill",
                tool_id="T01",
                parameters=MachiningParameters(param_source="handbook_default"),
                depends_on=[],
            ),
        ]
        with self.assertRaises(ValueError):
            CamPlan(
                source_part=base.source_part,
                feature_graph_ref=base.feature_graph_ref,
                setups=base.setups,
                operations=bad_ops,
                tools=base.tools,
            )

    def test_unknown_tool_id_rejected(self):
        base = example_cam_plan()
        ops = base.operations[:]
        ops[0] = ops[0].model_copy(update={"tool_id": "T99"})
        with self.assertRaises(ValueError):
            CamPlan(
                source_part=base.source_part,
                feature_graph_ref=base.feature_graph_ref,
                setups=base.setups,
                operations=ops,
                tools=base.tools,
            )

    def test_json_schema_has_cam_plan_title(self):
        schema = CamPlan.model_json_schema()
        self.assertIn("$defs", schema)
        self.assertEqual(schema.get("title"), "CamPlan")


if __name__ == "__main__":
    unittest.main()
