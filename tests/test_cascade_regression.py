"""OCC integration tests for cascade golden regression fixtures."""
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

from cascade_regression import (  # noqa: E402
    format_partition_diff_report,
    list_regression_fixture_ids,
    load_regression_fixture_by_id,
    run_fixture_regression,
)


@unittest.skipUnless(HAS_OCC, "pythonocc-core not installed")
class CascadeRegressionFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_ids = list_regression_fixture_ids(repo_root=ROOT)
        if not cls.fixture_ids:
            raise unittest.SkipTest("no fixtures under eval/regression/fixtures/")

        missing_golden: list[str] = []
        missing_steps: list[str] = []
        for fixture_id in cls.fixture_ids:
            fixture = load_regression_fixture_by_id(fixture_id, repo_root=ROOT)
            if not fixture.golden_path.is_file():
                missing_golden.append(str(fixture.golden_path))
            if not fixture.step_exists:
                missing_steps.append(f"{fixture_id} ({fixture.step})")

        if missing_golden:
            raise AssertionError(
                "missing golden partition file(s) (misconfig, not skip):\n"
                + "\n".join(missing_golden)
            )
        if missing_steps:
            raise unittest.SkipTest(
                "missing fixture STEP files: " + ", ".join(missing_steps)
            )

    def test_fixtures_match_golden_partitions(self):
        for fixture_id in self.fixture_ids:
            with self.subTest(fixture_id=fixture_id):
                fixture = load_regression_fixture_by_id(fixture_id, repo_root=ROOT)
                diff = run_fixture_regression(fixture)
                self.assertTrue(
                    diff.gate_passed,
                    f"{fixture_id} regression:\n"
                    + format_partition_diff_report(diff),
                )


if __name__ == "__main__":
    unittest.main()
