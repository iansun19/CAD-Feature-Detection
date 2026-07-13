"""Unit tests for taxonomy.py — canonical 25→12 mapping."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brep.taxonomy import (  # noqa: E402
    NUM_CLASSES,
    OLD_TO_NEW,
    old_to_new,
    validate,
)


class TestTaxonomy(unittest.TestCase):
    def test_validate_passes(self):
        validate()

    def test_num_classes(self):
        self.assertEqual(NUM_CLASSES, 12)

    def test_every_old_id_maps(self):
        self.assertEqual(set(OLD_TO_NEW.keys()), set(range(25)))
        for old_id in range(25):
            self.assertIn(old_to_new(old_id), range(NUM_CLASSES))

    def test_image_is_exactly_zero_to_eleven(self):
        self.assertEqual(set(OLD_TO_NEW.values()), set(range(12)))

    def test_collapse_groups(self):
        groups = {
            0: {1},
            1: {2, 3, 4},
            2: {5, 6, 7},
            3: {8, 9, 10},
            4: {11},
            5: {12},
            6: {13, 14, 15, 16},
            7: {17, 18, 19},
            8: {20, 21, 22},
            9: {0},
            10: {23},
            11: {24},
        }
        for new_id, old_ids in groups.items():
            for old_id in old_ids:
                self.assertEqual(old_to_new(old_id), new_id, f"old {old_id} → new {new_id}")

    def test_old_to_new_raises_outside_range(self):
        for bad in (-1, 25, 100):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    old_to_new(bad)


if __name__ == "__main__":
    unittest.main()
