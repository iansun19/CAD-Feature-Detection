"""Unit tests for feature_instances.py."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from feature_instances import (  # noqa: E402
    STOCK_CLASS,
    count_split_merge_events,
    face_iou,
    instances_from_labels,
    match_instances,
    union_find_instances,
)


class TestFeatureInstances(unittest.TestCase):
    def test_chain_one_instance(self):
        # 0 — 1 — 2  same class
        labels = np.array([0, 0, 0], dtype=np.int64)
        ei = np.array([[0, 1, 1], [1, 2, 2]], dtype=np.int64)
        inst = union_find_instances(3, labels, ei, ignore_class=None)
        self.assertEqual(len(set(inst.tolist())), 1)

    def test_two_disconnected_same_class(self):
        labels = np.array([0, 0, 0, 0], dtype=np.int64)
        ei = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)
        inst = union_find_instances(4, labels, ei, ignore_class=None)
        self.assertEqual(len(set(inst.tolist())), 2)

    def test_stock_ignored(self):
        labels = np.array([STOCK_CLASS, STOCK_CLASS, 0, 0], dtype=np.int64)
        ei = np.array([[0, 1, 2, 0], [1, 0, 3, 2]], dtype=np.int64)
        inst = union_find_instances(4, labels, ei, ignore_class=STOCK_CLASS)
        self.assertEqual(inst[0], -1)
        self.assertEqual(inst[1], -1)
        self.assertEqual(inst[2], inst[3])
        self.assertGreaterEqual(inst[2], 0)

    def test_face_iou(self):
        self.assertAlmostEqual(face_iou({1, 2, 3}, {2, 3, 4}), 2 / 4)

    def test_match_and_split_merge(self):
        gt = instances_from_labels(
            np.array([0, 0, 1, 1], dtype=np.int64),
            np.array([2, 2, 2, 2], dtype=np.int64),
        )
        # pred splits first GT instance into two singleton preds
        pred = instances_from_labels(
            np.array([0, 1, 2, 2], dtype=np.int64),
            np.array([2, 2, 2, 2], dtype=np.int64),
        )
        matches, _, _ = match_instances(gt, pred, iou_threshold=0.5)
        self.assertEqual(len(matches), 2)  # one partial match per split half + full second
        splits, merges = count_split_merge_events(gt, pred, iou_threshold=0.25)
        self.assertGreaterEqual(splits, 1)
        self.assertEqual(merges, 0)


if __name__ == "__main__":
    unittest.main()
