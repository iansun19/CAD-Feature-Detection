"""Unit tests for feature_graph.py."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from feature_graph import build_feature_graph, feature_adjacency  # noqa: E402
from feature_instances import FeatureInstance, instances_from_labels  # noqa: E402


class TestFeatureGraph(unittest.TestCase):
    def test_two_features_one_edge(self):
        # faces 0-1 class A, faces 2-3 class B; edge between face 1 and 2
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        conf = np.ones(4, dtype=np.float64)
        ei = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)
        g = build_feature_graph(labels, conf, ei, part_id="test", ignore_class=None)
        self.assertEqual(g["n_features"], 2)
        self.assertEqual(g["n_edges"], 1)
        self.assertEqual(g["edges"][0]["type"], "adjacent")

    def test_stock_excluded_from_nodes(self):
        labels = np.array([11, 11, 2, 2], dtype=np.int64)
        conf = np.ones(4, dtype=np.float64)
        ei = np.array([[0, 1, 2, 1], [1, 0, 3, 2]], dtype=np.int64)
        g = build_feature_graph(labels, conf, ei, part_id="test")
        self.assertEqual(g["n_features"], 1)
        self.assertEqual(g["nodes"][0]["class_id"], 2)
        self.assertEqual(g["n_edges"], 0)

    def test_adjacency_dedupes_undirected(self):
        inst = instances_from_labels(
            np.array([0, 0, 1], dtype=np.int64),
            np.array([0, 0, 1], dtype=np.int64),
        )
        ei = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
        edges = feature_adjacency(inst, ei, 3)
        self.assertEqual(len(edges), 1)


if __name__ == "__main__":
    unittest.main()
