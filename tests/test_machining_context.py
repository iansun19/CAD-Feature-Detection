"""Unit tests for machining_context.py."""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from machining_context import (  # noqa: E402
    build_context_v0,
    build_setup_context,
    build_stock,
    default_tool_library,
    load_machining_context,
    resolve_pocket_access,
    vector_to_opening_axis_label,
    write_machining_context,
)
from setup_descriptor import (  # noqa: E402
    OpeningAxisSpec,
    ResolvedSetup,
    SetupScope,
    resolve_setup_entry,
)

EXAMPLE_PATH = Path(ROOT) / "examples" / "machining_context_96260B.json"
SETUP_YAML = Path(ROOT) / "eval" / "gt" / "96260B_setup.yaml"
CASCADE_PATH = Path(ROOT) / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
REAR_STEP = Path(ROOT) / "96260B_REAR_XR004_PCD PLATE.stp copy"

MOCK_ENVELOPE = {
    "min": [0.0, -100.0, -50.0],
    "max": [200.0, 0.0, 50.0],
}


class TestStockOffset(unittest.TestCase):
    def test_uniform_oversize(self) -> None:
        stock = build_stock(MOCK_ENVELOPE, oversize_mm=2.0)
        self.assertEqual(stock.stock_type, "bbox")
        self.assertEqual(stock.bbox_min, (-2.0, -102.0, -52.0))
        self.assertEqual(stock.bbox_max, (202.0, 2.0, 52.0))
        self.assertEqual(stock.source, "brep_extents+offset")

    def test_custom_oversize(self) -> None:
        stock = build_stock(MOCK_ENVELOPE, oversize_mm=5.0)
        self.assertEqual(stock.bbox_min[0], -5.0)
        self.assertEqual(stock.bbox_max[0], 205.0)


class TestResolvePocketAccess(unittest.TestCase):
    def test_descriptor_wins_over_cascade(self) -> None:
        with self.assertLogs("machining_context", level="WARNING") as captured:
            result = resolve_pocket_access("0", "closed", "open")
        self.assertEqual(result, "closed")
        self.assertTrue(
            any("feature_id=0" in msg and "descriptor=closed" in msg for msg in captured.output)
        )

    def test_descriptor_value_returned_when_no_cascade(self) -> None:
        self.assertEqual(resolve_pocket_access("1", "open"), "open")

    def test_missing_descriptor_entry_is_unknown(self) -> None:
        self.assertEqual(resolve_pocket_access("16", None, "open"), "unknown")


class TestToolLibrary(unittest.TestCase):
    def test_default_library_size_and_fields(self) -> None:
        tools = default_tool_library()
        self.assertGreaterEqual(len(tools), 6)
        self.assertLessEqual(len(tools), 8)
        ids = {t.tool_id for t in tools}
        self.assertEqual(len(ids), len(tools))
        for tool in tools:
            self.assertEqual(tool.source, "hardcoded_v0")
            self.assertGreater(tool.diameter_mm, 0.0)


class TestSetupAssembly(unittest.TestCase):
    def _load_graph(self) -> dict:
        with open(CASCADE_PATH, encoding="utf-8") as fh:
            return json.load(fh)

    def test_single_setup_from_descriptor_and_cascade(self) -> None:
        from setup_descriptor import load_setup_descriptor

        descriptor = load_setup_descriptor(SETUP_YAML)
        resolved = resolve_setup_entry(descriptor, setup_id="rear")
        setup = build_setup_context(resolved, self._load_graph())

        self.assertEqual(setup.setup_id, "rear")
        self.assertEqual(setup.opening_axis, "+Y")
        self.assertIsNone(setup.fixture)
        self.assertEqual(
            setup.source_step_file,
            "96260B_REAR_XR004_PCD PLATE.stp copy",
        )
        self.assertIn("0", setup.pocket_access)
        self.assertEqual(setup.pocket_access["0"], "closed")

    def test_pocket_with_unspecified_access_resolves_unknown(self) -> None:
        # A pocket whose setup pins no access (machining_side None, pocket_access
        # not open/closed, no seed override) must resolve to "unknown" through
        # build_setup_context. Fixture-independent so it can't drift with feature
        # numbering (the rear golden currently pins every pocket to "closed").
        resolved = ResolvedSetup(
            part_id="synthetic",
            setup_id="unspecified",
            part_step=None,
            machining_side=None,
            opening_axis=OpeningAxisSpec(mode="explicit", vector=(0.0, 1.0, 0.0)),
            pocket_access="unknown",
            pockets_by_seed_face={},
            scope=SetupScope(),
        )
        graph = {
            "approach_frame": {"z": [0.0, 1.0, 0.0]},
            "nodes": [
                {"feature_id": 16, "class_name": "filleted_open_pocket", "params": {}},
            ],
        }
        setup = build_setup_context(resolved, graph)
        self.assertEqual(setup.pocket_access.get("16"), "unknown")

    def test_vector_to_opening_axis_label(self) -> None:
        self.assertEqual(vector_to_opening_axis_label([0.0, 1.0, 0.0]), "+Y")
        self.assertEqual(vector_to_opening_axis_label([0.0, 0.0, -1.0]), "-Z")


class TestFullExample(unittest.TestCase):
    def test_example_json_validates(self) -> None:
        self.assertTrue(EXAMPLE_PATH.is_file(), f"missing {EXAMPLE_PATH}")
        ctx = load_machining_context(EXAMPLE_PATH)
        self.assertEqual(ctx.schema_version, "0.1.0")
        self.assertEqual(len(ctx.setups), 1)
        self.assertEqual(ctx.setups[0].setup_id, "rear")
        self.assertEqual(ctx.completed_operations, [])
        self.assertIsNone(ctx.remaining_material)
        self.assertGreaterEqual(len(ctx.tools), 6)

    @unittest.skipUnless(REAR_STEP.is_file(), "96260B rear STEP not present")
    @unittest.skipUnless(CASCADE_PATH.is_file(), "96260B rear cascade not present")
    def test_build_context_v0_round_trip(self) -> None:
        ctx = build_context_v0(
            REAR_STEP,
            SETUP_YAML,
            CASCADE_PATH,
            oversize_mm=2.0,
            step_path=REAR_STEP,
            setups_source="authored",
        )
        # Rear descriptor pins every pocket to "closed" (machining_side=back);
        # feature 0 is a stable golden pocket. Verifies descriptor access flows in.
        self.assertEqual(ctx.setups[0].pocket_access.get("0"), "closed")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ctx.json"
            write_machining_context(out, ctx)
            reloaded = load_machining_context(out)
            self.assertEqual(reloaded.model_dump(), ctx.model_dump())


if __name__ == "__main__":
    unittest.main()
