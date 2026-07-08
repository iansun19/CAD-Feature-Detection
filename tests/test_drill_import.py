"""Unit tests for Fusion drill-library import in machining_context.py."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from machining_context import (  # noqa: E402
    load_tool_library,
    normalize_tool_type,
)

DRILL_LIBRARY = Path(ROOT) / "tool_libraries" / "Kennametal_Standard_Drills__Inch_.json"
EXAMPLE_OUTPUT = Path(ROOT) / "examples" / "tools_kennametal_drills.json"
INCH_TO_MM = 25.4


@unittest.skipUnless(DRILL_LIBRARY.is_file(), "Kennametal drill library missing")
class TestDrillImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tools = load_tool_library(DRILL_LIBRARY)

    def test_loads_two_hundred_twenty_four_drills(self) -> None:
        self.assertEqual(len(self.tools), 224)

    def test_drill_type_normalizes(self) -> None:
        self.assertEqual(normalize_tool_type("drill"), "drill")
        self.assertTrue(all(t.tool_type == "drill" for t in self.tools))

    def test_sig_maps_to_point_angle_deg(self) -> None:
        angles = {t.point_angle_deg for t in self.tools}
        self.assertEqual(angles, {140.0, 180.0})

        drill_0748 = next(
            t for t in self.tools if abs(t.diameter_mm - 0.0748 * INCH_TO_MM) < 1e-3
        )
        self.assertEqual(drill_0748.point_angle_deg, 140.0)

    def test_loads_without_shoulder_diameter(self) -> None:
        for tool in self.tools:
            self.assertGreater(tool.diameter_mm, 0.0)
            self.assertIsNotNone(tool.max_depth_mm)

    def test_drill_preset_parses_feed_fields_without_stepover(self) -> None:
        tool = next(
            t for t in self.tools if abs(t.diameter_mm - 0.0748 * INCH_TO_MM) < 1e-3
        )
        preset = next(p for p in tool.presets if p.preset_name == "LowCSteel_Drill")
        self.assertAlmostEqual(preset.feed_per_rev_mm, 0.0015362679997015592 * INCH_TO_MM, places=6)
        self.assertAlmostEqual(preset.plunge_mm_per_min, 20.5906 * INCH_TO_MM, places=3)
        self.assertAlmostEqual(preset.retract_mm_per_min, 20.5906 * INCH_TO_MM, places=3)
        self.assertEqual(preset.use_feed_per_revolution, False)
        self.assertIsNone(preset.stepover_mm)
        self.assertIsNone(preset.stepdown_mm)
        self.assertIsNone(preset.feed_mm_per_min)

    def test_unspecified_bmc_passthrough(self) -> None:
        unspecified = [t for t in self.tools if t.tool_material == "unspecified"]
        self.assertEqual(len(unspecified), 12)
        self.assertTrue(all(t.tool_type == "drill" for t in unspecified))

    def test_inch_to_mm_on_dc(self) -> None:
        drill = next(t for t in self.tools if abs(t.diameter_mm - 0.0748 * INCH_TO_MM) < 1e-3)
        self.assertAlmostEqual(drill.diameter_mm, 1.9, places=1)

    def test_example_output_matches_loader(self) -> None:
        self.assertTrue(EXAMPLE_OUTPUT.is_file(), f"missing {EXAMPLE_OUTPUT}")
        with open(EXAMPLE_OUTPUT, encoding="utf-8") as fh:
            exported = json.load(fh)
        self.assertEqual(len(exported), 224)
        self.assertEqual(exported, [t.model_dump(mode="json") for t in self.tools])


if __name__ == "__main__":
    unittest.main()
