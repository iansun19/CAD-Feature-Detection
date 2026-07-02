"""Tests for pipeline/core.py and pipeline/ingest.py."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.core import face_to_feature_map, write_face_predictions  # noqa: E402
from pipeline.ingest import require_occ  # noqa: E402
from feature_params import HAS_OCC  # noqa: E402


class TestPipelineCore(unittest.TestCase):
    def test_face_to_feature_map(self):
        graph = {
            "nodes": [
                {"feature_id": 0, "face_ids": [1, 2]},
                {"feature_id": 1, "face_ids": [5]},
            ]
        }
        m = face_to_feature_map(graph)
        self.assertEqual(m[1], 0)
        self.assertEqual(m[5], 1)
        self.assertNotIn(0, m)

    def test_write_face_predictions(self):
        pred = np.array([11, 0, 0], dtype=np.int64)
        conf = np.array([1.0, 0.9, 0.8])
        ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        graph = {"nodes": [{"feature_id": 0, "face_ids": [1, 2]}]}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "faces.jsonl")
            write_face_predictions(
                __import__("pathlib").Path(path),
                pred, conf, ei, graph,
            )
            with open(path) as f:
                rows = [json.loads(l) for l in f]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["feature_id"], 0)


class TestPipelineIngest(unittest.TestCase):
    @unittest.skipUnless(not HAS_OCC, "only when pythonocc is absent")
    def test_require_occ_without_pythonocc(self):
        with self.assertRaises(RuntimeError) as ctx:
            require_occ("test action")
        self.assertIn("environment.yml", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
