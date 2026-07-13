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

from cascade.stock_cut_classification import (  # noqa: E402
    CLASSIFICATION_FIXTURES,
    classify_report,
    format_flip_report,
    run_fixture_diff,
    stock_face_ids,
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

    def test_fish_mold_stock_gate_disabled(self):
        """fish_mold is a fully-machined mold: the STOCK gate is disabled, so no
        face is gated and the envelope-extreme planes (114, 99) plus the corner
        chamfers all stay CUT candidates for the cascade."""
        step = _fixture_step("fish_mold")
        recs = classify_report(step, classifier="new")
        stock = {r.face_id for r in recs if r.label == "STOCK"}
        self.assertEqual(
            stock,
            set(),
            f"expected no STOCK faces for fish_mold; got {sorted(stock)}",
        )
        self.assertEqual(stock_face_ids(step, classifier="new"), set())

        # Faces previously gated as envelope stock are now CUT candidates.
        corner_ids = set(CLASSIFICATION_FIXTURES["fish_mold"]["corner_chamfer_face_ids"])
        for fid in {114, 99} | corner_ids:
            self.assertNotIn(fid, stock)

        # With the gate off, old vs new collapse to the same all-CUT partition.
        diff = run_fixture_diff("fish_mold", repo_root=ROOT)
        self.assertEqual(diff.new_stock_count, 0, diff.summary_line())
        self.assertEqual(
            diff.n_flipped,
            0,
            "gate-disabled fish_mold should have no old/new flips:\n"
            + format_flip_report(diff),
        )

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
