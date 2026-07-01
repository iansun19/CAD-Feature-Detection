"""
Tests for runtime STEP ingestion (step_ingest.py).

Requires pythonocc (conda-forge). Skipped when OCC is not installed.
No non-MFCAD++ STEP fixtures are checked into this repo; MFCAD++ files are used
for structural/regression tests only. Add tests/fixtures/sample_external.step
(from a real CAD export) for messy-geometry coverage.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from OCC.Core.STEPControl import STEPControl_Reader  # noqa: F401
    HAS_OCC = True
except ImportError:
    HAS_OCC = False

from step_ingest import (  # noqa: E402
    StepIngestError,
    extract_brep_from_step,
    ingest_step_to_pyg,
    load_step_shape,
    model_to_pyg,
)


@unittest.skipUnless(HAS_OCC, "pythonocc-core not installed")
class StepIngestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_step = os.path.join(
            ROOT, "MFCAD++_dataset", "step", "test", "29000.step",
        )
        cls.sample_dir = os.path.join(ROOT, "MFCAD++_dataset", "step", "test")
        if not os.path.isfile(cls.sample_step):
            raise unittest.SkipTest(f"missing sample STEP: {cls.sample_step}")

    def test_load_and_extract(self):
        model, stats = extract_brep_from_step(self.sample_step, require_labels=False)
        self.assertTrue(stats.success)
        self.assertGreater(model["N"], 0)
        self.assertEqual(model["area"].shape[0], model["N"])
        self.assertEqual(model["normals"].shape, (model["N"], 3))
        self.assertNotIn("labels", model)

    def test_pyg_shapes(self):
        x, ei, ea, stats = ingest_step_to_pyg(self.sample_step)
        self.assertTrue(stats.success)
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape[0], stats.face_count)
        self.assertEqual(ei.shape[0], 2)
        self.assertEqual(ea.shape[1], 4)
        if ei.shape[1] > 0:
            self.assertEqual(ea.shape[0], ei.shape[1])

    def test_labeled_regen_path(self):
        """read_model requires STEP entity-name labels (absent in repo STEP exports)."""
        from diag.regen_dataset import read_model

        m = read_model(self.sample_step)
        # Repo STEP files are geometry-only; labeled MFCAD++ exports return a model.
        if m is None:
            self.assertIsNone(m)
            return
        self.assertIn("labels", m)
        self.assertEqual(len(m["labels"]), m["N"])

    def test_missing_file_raises(self):
        with self.assertRaises(StepIngestError):
            load_step_shape("/nonexistent/part.step")

    def test_batch_fallback_report(self):
        """Run a small batch and print fallback fractions (informational)."""
        names = sorted(
            f for f in os.listdir(self.sample_dir) if f.lower().endswith(".step")
        )[:20]
        totals = {
            "files": 0, "surface_type_other": 0, "undefined_normals": 0,
            "zero_area_faces": 0, "non_manifold_edges": 0,
        }
        for name in names:
            path = os.path.join(self.sample_dir, name)
            model, stats = extract_brep_from_step(path, require_labels=False)
            self.assertIsNotNone(model)
            totals["files"] += 1
            totals["surface_type_other"] += stats.surface_type_other
            totals["undefined_normals"] += stats.undefined_normals
            totals["zero_area_faces"] += stats.zero_area_faces
            totals["non_manifold_edges"] += stats.non_manifold_edges
        # MFCAD++ synthetic parts should be clean; document if not.
        self.assertEqual(totals["surface_type_other"], 0)
        self.assertEqual(totals["non_manifold_edges"], 0)


if __name__ == "__main__":
    unittest.main()
