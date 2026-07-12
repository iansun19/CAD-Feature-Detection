"""Unit tests for Fusion tool-library import in machining_context.py."""
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
    Tool,
    build_context_v0,
    load_enabled_libraries,
    load_tool_library,
    load_tool_library_payload,
    normalize_tool_type,
)
from tool_store import row_to_tool, tool_to_row  # noqa: E402

SAMPLE_LIBRARY = Path(ROOT) / "tests" / "fixtures" / "Aluminum_Sample_Library__Inch_.json"
EXAMPLE_OUTPUT = Path(ROOT) / "examples" / "tools_aluminum_sample.json"
SETUP_YAML = Path(ROOT) / "eval" / "gt" / "96260B_setup.yaml"
CASCADE_PATH = Path(ROOT) / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
MOCK_ENVELOPE = {"min": [0.0, -100.0, -50.0], "max": [200.0, 0.0, 50.0]}

INCH_TO_MM = 25.4


class TestNormalizeToolType(unittest.TestCase):
    def test_known_fusion_types(self) -> None:
        cases = {
            "flat end mill": "endmill",
            "face mill": "face_mill",
            "ball end mill": "ball_endmill",
            "bull nose end mill": "bullnose_endmill",
            "chamfer mill": "chamfer_mill",
            "drill": "drill",
            "spot drill": "spot_drill",
            "tap right hand": "tap",
            "tapping": "tap",
            "slot mill": "slot_mill",
            "counter sink": "countersink",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_tool_type(raw), expected)

    def test_unknown_type_fallback(self) -> None:
        with self.assertLogs("machining_context", level="WARNING") as captured:
            result = normalize_tool_type("weird custom cutter")
        self.assertEqual(result, "unknown")
        self.assertTrue(any("unknown Fusion tool type" in msg for msg in captured.output))


class TestHolderExclusion(unittest.TestCase):
    def test_holder_entries_skipped_with_single_log(self) -> None:
        payload = {
            "version": 1,
            "data": [
                {
                    "guid": "holder-1",
                    "type": "holder",
                    "description": "ER16 collet chuck",
                },
                {
                    "guid": "holder-2",
                    "type": "holder",
                    "description": "CAT40 holder",
                },
            ],
        }
        with self.assertLogs("machining_context", level="INFO") as captured:
            tools = load_tool_library_payload(
                payload,
                library_name="test_holders",
                source_label="holders.json",
            )
        self.assertEqual(tools, [])
        holder_logs = [
            msg for msg in captured.output if "skipped 2 holder entries" in msg
        ]
        self.assertEqual(len(holder_logs), 1)
        self.assertFalse(any("missing geometry.DC" in msg for msg in captured.output))
        self.assertFalse(any("unknown Fusion tool type" in msg for msg in captured.output))


class TestSlotMillAndCountersinkRoundTrip(unittest.TestCase):
    def _load_one(self, raw_type: str, guid: str) -> Tool:
        payload = {
            "version": 1,
            "data": [
                {
                    "guid": guid,
                    "type": raw_type,
                    "description": f"Test {raw_type}",
                    "unit": "inches",
                    "geometry": {"DC": 0.25, "LCF": 0.5, "LB": 1.0, "SIG": 82.0},
                    "start-values": {
                        "presets": [
                            {
                                "name": "Alu preset",
                                "material": {"category": "aluminum"},
                                "n": 10000,
                                "v_f": 50.0,
                            }
                        ]
                    },
                }
            ],
        }
        tools = load_tool_library_payload(payload, library_name="test_lib")
        self.assertEqual(len(tools), 1)
        return tools[0]

    def test_slot_mill_round_trip(self) -> None:
        tool = self._load_one("slot mill", "slot-mill-guid")
        self.assertEqual(tool.tool_type, "slot_mill")
        row = tool_to_row(
            tool,
            source_library="test_lib",
            raw_type="slot mill",
        )
        restored = row_to_tool(row)
        self.assertEqual(restored.tool_type, "slot_mill")
        self.assertAlmostEqual(restored.diameter_mm, tool.diameter_mm)
        self.assertEqual(len(restored.presets), 1)
        self.assertEqual(restored.presets[0].preset_name, "Alu preset")

    def test_countersink_round_trip(self) -> None:
        tool = self._load_one("counter sink", "countersink-guid")
        self.assertEqual(tool.tool_type, "countersink")
        row = tool_to_row(
            tool,
            source_library="test_lib",
            raw_type="counter sink",
        )
        restored = row_to_tool(row)
        self.assertEqual(restored.tool_type, "countersink")
        self.assertAlmostEqual(restored.point_angle_deg, tool.point_angle_deg)


@unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "aluminum sample library fixture missing")
class TestLoadToolLibrary(unittest.TestCase):
    def test_loads_twelve_tools(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        self.assertEqual(len(tools), 12)

    def test_inch_to_mm_conversion(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        ball_125 = next(
            t for t in tools if t.name == 'Harvey 1/8" 3F Ball'
        )
        self.assertAlmostEqual(ball_125.diameter_mm, 0.125 * INCH_TO_MM, places=3)
        self.assertEqual(ball_125.source_unit, "inches")

        first = tools[0]
        self.assertAlmostEqual(first.diameter_mm, 0.062 * INCH_TO_MM, places=3)
        self.assertAlmostEqual(first.flute_length_mm, 0.093 * INCH_TO_MM, places=3)
        self.assertAlmostEqual(first.max_depth_mm, 0.877 * INCH_TO_MM, places=3)

    def test_type_normalization_covers_sample_types(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        by_type = {t.tool_type for t in tools}
        self.assertEqual(
            by_type,
            {
                "endmill",
                "face_mill",
                "ball_endmill",
                "bullnose_endmill",
                "chamfer_mill",
            },
        )

    def test_presets_parsed_with_rpm_and_feed(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        tool = tools[0]
        self.assertGreater(len(tool.presets), 0)
        preset = next(p for p in tool.presets if p.preset_name == "AluWrought_Adaptive_Rough")
        self.assertEqual(preset.spindle_rpm, 12000.0)
        self.assertAlmostEqual(preset.feed_mm_per_min, 17.856 * INCH_TO_MM, places=2)
        self.assertAlmostEqual(preset.plunge_mm_per_min, 60.0 * INCH_TO_MM, places=2)
        self.assertAlmostEqual(preset.stepdown_mm, 0.0279 * INCH_TO_MM, places=2)
        self.assertEqual(preset.coolant, "flood")

    def test_tool_ids_prefixed_with_library_name(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        for tool in tools:
            self.assertTrue(tool.tool_id.startswith("Aluminum_Sample_Library_Inch::"))
            self.assertEqual(tool.source, "fusion_library:Aluminum_Sample_Library_Inch")

    def test_example_output_matches_loader(self) -> None:
        tools = load_tool_library(SAMPLE_LIBRARY)
        self.assertTrue(EXAMPLE_OUTPUT.is_file(), f"missing {EXAMPLE_OUTPUT}")
        with open(EXAMPLE_OUTPUT, encoding="utf-8") as fh:
            exported = json.load(fh)
        self.assertEqual(len(exported), 12)
        self.assertEqual(exported, [t.model_dump(mode="json") for t in tools])


class TestLoadEnabledLibraries(unittest.TestCase):
    def test_deduplicates_by_guid(self) -> None:
        shared_guid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        tool_a = {
            "guid": shared_guid,
            "description": "First",
            "type": "flat end mill",
            "unit": "millimeters",
            "geometry": {"DC": 6.0, "LCF": 19.0, "LB": 25.0},
            "start-values": {"presets": []},
        }
        tool_b = {
            "guid": shared_guid,
            "description": "Duplicate",
            "type": "flat end mill",
            "unit": "millimeters",
            "geometry": {"DC": 8.0, "LCF": 19.0, "LB": 25.0},
            "start-values": {"presets": []},
        }
        unique = {
            "guid": "11111111-2222-3333-4444-555555555555",
            "description": "Unique",
            "type": "drill",
            "unit": "millimeters",
            "geometry": {"DC": 3.0, "LCF": 20.0, "LB": 30.0},
            "start-values": {"presets": []},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lib_one = tmp_path / "lib_one.json"
            lib_two = tmp_path / "lib_two.json"
            lib_one.write_text(
                json.dumps({"version": 1, "data": [tool_a, unique]}),
                encoding="utf-8",
            )
            lib_two.write_text(
                json.dumps({"version": 1, "data": [tool_b]}),
                encoding="utf-8",
            )

            tools = load_enabled_libraries([lib_one, lib_two])
            self.assertEqual(len(tools), 2)
            by_guid = {t.guid: t for t in tools}
            self.assertEqual(by_guid[shared_guid].name, "First")
            self.assertEqual(by_guid[unique["guid"]].tool_type, "drill")


@unittest.skipUnless(SAMPLE_LIBRARY.is_file(), "aluminum sample library fixture missing")
@unittest.skipUnless(CASCADE_PATH.is_file(), "96260B rear cascade not present")
class TestBuildContextWithLibrary(unittest.TestCase):
    def test_build_context_v0_uses_library_tools(self) -> None:
        ctx = build_context_v0(
            MOCK_ENVELOPE,
            SETUP_YAML,
            CASCADE_PATH,
            setup_id="rear",
            tool_library_paths=[SAMPLE_LIBRARY],
            setups_source="authored",
        )
        self.assertGreater(len(ctx.tools), 8)
        self.assertEqual(len(ctx.tools), 12)
        self.assertTrue(all(t.source.startswith("fusion_library:") for t in ctx.tools))


if __name__ == "__main__":
    unittest.main()
