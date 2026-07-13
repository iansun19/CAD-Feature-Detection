"""Unit tests for cam_plan_schema.py (no pythonocc required)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from schema.cam_plan_schema import (  # noqa: E402
    BRACKET_HIDDEN_FIELDS,
    CamPlan,
    Operation,
    PocketAccess,
    MachiningParameters,
    Setup,
    ToolRef,
    example_cam_plan,
    load_cam_plan,
    to_bracket_dict,
)


EXAMPLE_PATH = Path(ROOT) / "examples" / "cam_plan_example.json"


class TestCamPlanSchema(unittest.TestCase):
    def test_example_cam_plan_validates(self):
        plan = example_cam_plan()
        self.assertEqual(plan.schema_version, "1.0")
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


class TestBracketFormat(unittest.TestCase):
    """Bracket output = native external format; lossless model keeps every field.

    Uses the committed real planner output (examples/cam_plan_96260B_front.json) so the
    assertions hold "after planner.py runs" without needing pythonocc, and falls back to
    the hand-authored example when the planner artifact is absent.
    """

    FRONT_PATH = Path(ROOT) / "examples" / "cam_plan_96260B_front.json"

    def _plan(self) -> CamPlan:
        if self.FRONT_PATH.is_file():
            return load_cam_plan(self.FRONT_PATH)
        return example_cam_plan()

    def _walk_bracket_dicts(self, node, model_name):
        """Yield (model_name, dict) pairs to check against BRACKET_HIDDEN_FIELDS."""
        if model_name == "CamPlan":
            yield ("CamPlan", node)
            for setup in node.get("setups", []):
                yield from self._walk_bracket_dicts(setup, "Setup")
        elif model_name == "Setup":
            yield ("Setup", node)
            for op in node.get("operations", []):
                yield from self._walk_bracket_dicts(op, "Operation")
        elif model_name == "Operation":
            yield ("Operation", node)
            params = node.get("parameters")
            if isinstance(params, dict):
                yield ("MachiningParameters", params)

    def test_a_no_hidden_field_in_bracket_output(self):
        bracket = to_bracket_dict(self._plan())
        for model_name, blob in self._walk_bracket_dicts(bracket, "CamPlan"):
            for hidden in BRACKET_HIDDEN_FIELDS.get(model_name, frozenset()):
                self.assertNotIn(
                    hidden,
                    blob,
                    f"hidden field {model_name}.{hidden} leaked into bracket output",
                )
        # No flat top-level operations[] in the bracket document.
        self.assertNotIn("operations", bracket)

    def test_b_hidden_fields_present_and_populated_on_model(self):
        plan = self._plan()
        lossless = plan.model_dump(mode="json")
        # CamPlan-level hidden fields still serialize in the lossless dump.
        for name in BRACKET_HIDDEN_FIELDS["CamPlan"]:
            self.assertIn(name, lossless)
        self.assertTrue(plan.tools, "tool catalog must stay populated")
        self.assertTrue(plan.feature_graph_ref)
        self.assertIsInstance(plan.metadata, dict)
        # Operation-level hidden fields stay on every in-memory op.
        for op in plan.operations:
            self.assertTrue(op.setup_id)
            self.assertIsInstance(op.sequence_index, int)
            self.assertTrue(op.tool_id)
            self.assertIsInstance(op.depends_on, list)
            self.assertIsInstance(op.attributes, dict)
            self.assertIsInstance(op.parameters.param_source, str)

    def test_c_every_op_nested_under_exactly_one_setup(self):
        plan = self._plan()
        setup_ids = [s.setup_id for s in plan.setups]
        self.assertEqual(len(setup_ids), len(set(setup_ids)), "duplicate setup_id")
        seen: dict[str, int] = {}
        for setup in plan.setups:
            for op in setup.operations:
                seen[op.op_id] = seen.get(op.op_id, 0) + 1
                self.assertEqual(
                    op.setup_id, setup.setup_id, "op nested under the wrong setup"
                )
        # Each op appears under exactly one setup, and the flat view matches the nesting.
        self.assertTrue(all(count == 1 for count in seen.values()))
        self.assertEqual(len(seen), len(plan.operations))

    def test_d_every_op_carries_inline_tool_blob(self):
        plan = self._plan()
        catalog = {t.tool_id: t for t in plan.tools}
        for op in plan.operations:
            self.assertIsNotNone(op.tool, f"op {op.op_id} missing inline tool")
            self.assertEqual(op.tool.id, op.tool_id)
            self.assertEqual(op.tool.type, catalog[op.tool_id].tool_type)
            self.assertEqual(op.tool.dia, catalog[op.tool_id].diameter_mm)
        # And in the serialized bracket output.
        bracket = to_bracket_dict(plan)
        for setup in bracket["setups"]:
            for op in setup["operations"]:
                self.assertIn("tool", op)
                self.assertIn("id", op["tool"])
                self.assertIn("dia", op["tool"])


if __name__ == "__main__":
    unittest.main()
