"""Regression tests for STOCK/CUT face classification (old vs convexity-primary new)."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from OCC.Core.STEPControl import STEPControl_Reader  # noqa: F401

    HAS_OCC = True
except ImportError:
    HAS_OCC = False

from stock_cut_classification import (  # noqa: E402
    CLASSIFICATION_FIXTURES,
    format_flip_report,
    run_fixture_diff,
)


def _fixture_step(name: str) -> Path:
    return Path(ROOT) / CLASSIFICATION_FIXTURES[name]["step"]


@unittest.skipUnless(HAS_OCC, "pythonocc-core not installed")
class StockCutClassificationFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [
            name
            for name in CLASSIFICATION_FIXTURES
            if not _fixture_step(name).is_file()
        ]
        if missing:
            raise unittest.SkipTest(
                "missing fixture STEP files: " + ", ".join(missing)
            )

    def test_fish_mold_expects_chamfer_flips(self):
        """fish_mold: convexity-primary should flip corner chamfers CUT->STOCK."""
        diff = run_fixture_diff("fish_mold", repo_root=ROOT)
        corner_ids = set(CLASSIFICATION_FIXTURES["fish_mold"]["corner_chamfer_face_ids"])

        self.assertGreater(
            diff.n_flipped,
            0,
            "expected non-empty flip set for fish_mold convexity-primary change",
        )
        self.assertEqual(diff.old_bounding_stock_count, 6, diff.summary_line())
        self.assertEqual(diff.new_bounding_stock_count, 10, diff.summary_line())

        corner_flips = [
            f for f in diff.flips
            if f["face_id"] in corner_ids
            and f["old_label"] == "CUT"
            and f["new_label"] == "STOCK"
        ]
        self.assertEqual(
            len(corner_flips),
            len(corner_ids),
            "expected all corner chamfer faces to flip CUT->STOCK:\n"
            + format_flip_report(diff),
        )
        for flip in corner_flips:
            self.assertIn("chamfer", flip["reason"].lower())

    def test_96260B_front_stable(self):
        """
        96260B_front tripwire: any flip is a decision point, not an automatic bug.

        Inspect flip direction, driving signals, convex_summary, and normal_consistent
        to distinguish a correction from a regression (normal flip / tangent bug).
        """
        diff = run_fixture_diff("96260B_front", repo_root=ROOT)
        if diff.n_flipped:
            self.fail(
                "96260B_front must be stable (empty flip set); got:\n"
                + format_flip_report(diff)
            )

    def test_96260B_rear_stable(self):
        """
        96260B_rear tripwire: same contract as front - emergent stability only.
        """
        diff = run_fixture_diff("96260B_rear", repo_root=ROOT)
        if diff.n_flipped:
            self.fail(
                "96260B_rear must be stable (empty flip set); got:\n"
                + format_flip_report(diff)
            )


if __name__ == "__main__":
    unittest.main()
