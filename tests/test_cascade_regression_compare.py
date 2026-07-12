"""Pure unit tests for cascade_regression.py (no OCC)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cascade_regression import (  # noqa: E402
    FaceFingerprint,
    FaceFingerprintMismatchError,
    PartitionValidationError,
    T_FEATURE_FEATURE,
    T_FEATURE_STOCK,
    T_STOCK_UNCLAIMED,
    build_face_to_bucket,
    build_face_to_coarse_bucket,
    compare_partitions,
    dump_partition,
    face_iou,
    format_partition_diff_report,
    load_partition,
    partition_from_dict,
    validate_partition,
    verify_face_fingerprints,
)


def _fingerprints(n_faces: int, *, surface_type: str = "plane", area: float = 1.0):
    return [
        FaceFingerprint(surface_type=surface_type, area_mm2=area + i * 0.001)
        for i in range(n_faces)
    ]


def _partition(raw: dict):
    base = {
        "fixture_id": "synthetic",
        "n_faces": 6,
        "instances": [
            {"class_name": "filleted_pocket", "face_ids": [0, 1, 2]},
            {"class_name": "filleted_open_pocket", "face_ids": [3, 4]},
        ],
        "stock_face_ids": [5],
        "unclaimed_face_ids": [],
        "face_fingerprints": [fp.to_dict() for fp in _fingerprints(6)],
    }
    base.update(raw)
    return partition_from_dict(base)


def _expected_bucket_changes(golden, fresh):
    golden_buckets = build_face_to_bucket(golden)
    fresh_buckets = build_face_to_bucket(fresh)
    return {
        face_id
        for face_id in range(golden.n_faces)
        if golden_buckets[face_id] != fresh_buckets[face_id]
    }


def _expected_boundary_crossings(golden, fresh):
    golden_coarse = build_face_to_coarse_bucket(golden)
    fresh_coarse = build_face_to_coarse_bucket(fresh)
    return {
        face_id
        for face_id in range(golden.n_faces)
        if golden_coarse[face_id] != fresh_coarse[face_id]
    }


class TestCascadeRegressionCompare(unittest.TestCase):
    def test_exact_match_passes(self):
        golden = _partition({"fixture_id": "exact_match"})
        fresh = _partition({"fixture_id": "exact_match"})
        diff = compare_partitions(golden, fresh)
        self.assertTrue(diff.gate_passed)
        self.assertEqual(diff.n_bucket_changes, 0)
        self.assertEqual(diff.n_changed, 0)
        self.assertEqual(diff.n_boundary_crossings, 0)
        self.assertEqual(diff.regrouping_events, [])
        self.assertEqual(diff.summary_line(), "PASS - partition unchanged")
        report = format_partition_diff_report(diff)
        self.assertIn("gate: PASS (partition unchanged)", report)

    def test_n_changed_exact_on_sub_threshold_reassignment(self):
        """Cross-class reassignment: strict gate vs honest reporting signals."""
        golden = _partition({
            "fixture_id": "majority_reassign",
            "n_faces": 15,
            "instances": [
                {"class_name": "filleted_pocket", "face_ids": list(range(10))},
                {"class_name": "filleted_open_pocket", "face_ids": [10, 11, 12, 13, 14]},
            ],
            "stock_face_ids": [],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(15)],
        })
        fresh = _partition({
            "fixture_id": "majority_reassign",
            "n_faces": 15,
            "instances": [
                {"class_name": "filleted_pocket", "face_ids": [0, 1, 2, 3]},
                {
                    "class_name": "filleted_open_pocket",
                    "face_ids": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                },
            ],
            "stock_face_ids": [],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(15)],
        })

        golden_pocket = frozenset(range(10))
        fresh_pocket = frozenset([0, 1, 2, 3])
        fresh_open = frozenset([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        self.assertLess(face_iou(golden_pocket, fresh_pocket), 0.5)
        self.assertLess(face_iou(golden_pocket, fresh_open), 0.5)

        expected_bucket_changes = _expected_bucket_changes(golden, fresh)
        expected_boundary_crossings = _expected_boundary_crossings(golden, fresh)
        # Strict gate: every face whose instance membership string changed (15).
        self.assertEqual(expected_bucket_changes, set(range(15)))
        # Honest signal: only the 6 faces that crossed pocket -> open class.
        self.assertEqual(expected_boundary_crossings, set(range(4, 10)))

        diff = compare_partitions(golden, fresh)
        self.assertFalse(diff.gate_passed)
        self.assertEqual(diff.n_bucket_changes, 15)
        self.assertEqual(diff.n_changed, 15)
        self.assertEqual(diff.n_boundary_crossings, 6)
        self.assertEqual(diff.regrouping_events, [])
        self.assertNotEqual(diff.n_bucket_changes, len(golden_pocket))

    def test_pure_same_class_split_fails_gate_with_zero_boundary_crossings(self):
        golden = _partition({
            "fixture_id": "same_class_split",
            "n_faces": 10,
            "instances": [{"class_name": "filleted_pocket", "face_ids": list(range(10))}],
            "stock_face_ids": [],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(10)],
        })
        fresh = _partition({
            "fixture_id": "same_class_split",
            "n_faces": 10,
            "instances": [
                {"class_name": "filleted_pocket", "face_ids": [0, 1, 2, 3, 4]},
                {"class_name": "filleted_pocket", "face_ids": [5, 6, 7, 8, 9]},
            ],
            "stock_face_ids": [],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(10)],
        })

        diff = compare_partitions(golden, fresh)
        self.assertFalse(diff.gate_passed)
        self.assertEqual(diff.n_bucket_changes, 10)
        self.assertEqual(diff.n_boundary_crossings, 0)
        self.assertEqual(len(diff.regrouping_events), 1)
        self.assertEqual(diff.regrouping_events[0].kind, "split")
        self.assertIn("filleted_pocket{0..9}", diff.regrouping_events[0].format_line())

        report = format_partition_diff_report(diff)
        self.assertIn("REGRESSION \u2014 gate FAIL", report)
        self.assertIn("gate: FAIL (partition changed)", report)
        self.assertIn(
            "boundary crossings (faces changing class/stock/unclaimed): 0",
            report,
        )
        self.assertIn("regrouping: 1 split", report)
        self.assertIn("filleted_pocket{0..4}", report)
        self.assertIn("filleted_pocket{5..9}", report)

    def test_uneven_same_class_split_6_4(self):
        """6/4 split: small child fails IoU-0.5 but passes share-of-origin."""
        golden = _partition({
            "fixture_id": "uneven_split",
            "n_faces": 10,
            "instances": [{"class_name": "filleted_pocket", "face_ids": list(range(10))}],
            "stock_face_ids": [],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(10)],
        })
        fresh = _partition({
            "fixture_id": "uneven_split",
            "n_faces": 10,
            "instances": [
                {"class_name": "filleted_pocket", "face_ids": [0, 1, 2, 3, 4, 5]},
                {"class_name": "filleted_pocket", "face_ids": [6, 7, 8, 9]},
            ],
            "stock_face_ids": [],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(10)],
        })

        g_faces = frozenset(range(10))
        child_large = frozenset(range(6))
        child_small = frozenset(range(6, 10))
        self.assertGreaterEqual(len(g_faces & child_large) / len(child_large), 0.5)
        self.assertGreaterEqual(len(g_faces & child_small) / len(child_small), 0.5)
        self.assertEqual(len(g_faces & child_large) / len(g_faces), 0.6)
        self.assertEqual(len(g_faces & child_small) / len(g_faces), 0.4)
        self.assertLess(face_iou(g_faces, child_small), 0.5)

        diff = compare_partitions(golden, fresh)
        self.assertFalse(diff.gate_passed)
        self.assertEqual(diff.n_boundary_crossings, 0)
        self.assertEqual(len(diff.regrouping_events), 1)
        event = diff.regrouping_events[0]
        self.assertEqual(event.kind, "split")
        self.assertEqual(event.leftover_face_count, 0)
        self.assertIn("filleted_pocket{0..5}", event.format_line())
        self.assertIn("filleted_pocket{6..9}", event.format_line())
        self.assertNotIn("left the feature", event.format_line())

    def test_same_class_split_with_face_leak_to_stock(self):
        """Split plus one face to stock: partial coverage, still one split event."""
        golden = _partition({
            "fixture_id": "split_with_leak",
            "n_faces": 10,
            "instances": [{"class_name": "filleted_pocket", "face_ids": list(range(10))}],
            "stock_face_ids": [],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(10)],
        })
        fresh = _partition({
            "fixture_id": "split_with_leak",
            "n_faces": 10,
            "instances": [
                {"class_name": "filleted_pocket", "face_ids": [0, 1, 2, 3, 4]},
                {"class_name": "filleted_pocket", "face_ids": [5, 6, 7, 8]},
            ],
            "stock_face_ids": [9],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(10)],
        })

        g_faces = frozenset(range(10))
        child_a = frozenset(range(5))
        child_b = frozenset(range(5, 9))
        child_union = child_a | child_b
        self.assertEqual(len(child_union & g_faces) / len(g_faces), 0.9)

        diff = compare_partitions(golden, fresh)
        self.assertFalse(diff.gate_passed)
        self.assertEqual(diff.n_boundary_crossings, 1)
        self.assertEqual(len(diff.regrouping_events), 1)
        event = diff.regrouping_events[0]
        self.assertEqual(event.kind, "split")
        self.assertEqual(event.leftover_face_count, 1)
        formatted = event.format_line()
        self.assertIn("filleted_pocket{0..4}", formatted)
        self.assertIn("filleted_pocket{5..8}", formatted)
        self.assertIn("(1 face left the feature: see boundary crossings)", formatted)

    def test_same_class_merge_two_pockets(self):
        """Two same-class golden pockets merge into one fresh instance."""
        golden = _partition({
            "fixture_id": "same_class_merge",
            "n_faces": 10,
            "instances": [
                {"class_name": "filleted_pocket", "face_ids": [0, 1, 2, 3, 4]},
                {"class_name": "filleted_pocket", "face_ids": [5, 6, 7, 8, 9]},
            ],
            "stock_face_ids": [],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(10)],
        })
        fresh = _partition({
            "fixture_id": "same_class_merge",
            "n_faces": 10,
            "instances": [{"class_name": "filleted_pocket", "face_ids": list(range(10))}],
            "stock_face_ids": [],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(10)],
        })

        f_faces = frozenset(range(10))
        parent_a = frozenset(range(5))
        parent_b = frozenset(range(5, 10))
        self.assertGreaterEqual(len(parent_a & f_faces) / len(parent_a), 0.5)
        self.assertGreaterEqual(len(parent_b & f_faces) / len(parent_b), 0.5)
        self.assertEqual(
            len((parent_a | parent_b) & f_faces) / len(f_faces),
            1.0,
        )

        diff = compare_partitions(golden, fresh)
        self.assertFalse(diff.gate_passed)
        self.assertEqual(diff.n_boundary_crossings, 0)
        self.assertEqual(len(diff.regrouping_events), 1)
        event = diff.regrouping_events[0]
        self.assertEqual(event.kind, "merge")
        self.assertEqual(event.leftover_face_count, 0)
        formatted = event.format_line()
        self.assertIn("filleted_pocket{0..4}", formatted)
        self.assertIn("filleted_pocket{5..9}", formatted)
        self.assertIn("-> filleted_pocket{0..9}", formatted)
        self.assertNotIn("joined from outside", formatted)

    def test_feature_to_feature_move_fails_as_single_iou_grouped_move(self):
        golden = _partition({"fixture_id": "move_one_face"})
        fresh = _partition({
            "fixture_id": "move_one_face",
            "instances": [
                {"class_name": "filleted_pocket", "face_ids": [0, 1]},
                {"class_name": "filleted_open_pocket", "face_ids": [2, 3, 4]},
            ],
            "stock_face_ids": [5],
        })
        diff = compare_partitions(golden, fresh)
        self.assertFalse(diff.gate_passed)
        expected_bucket_changes = _expected_bucket_changes(golden, fresh)
        expected_boundary_crossings = _expected_boundary_crossings(golden, fresh)
        # Strict gate: face 2 moved plus instance-key relabel on survivors {0,1,3,4}.
        self.assertEqual(expected_bucket_changes, {0, 1, 2, 3, 4})
        # Honest signal: only face 2 crossed filleted_pocket -> filleted_open_pocket.
        self.assertEqual(expected_boundary_crossings, {2})
        self.assertEqual(diff.n_bucket_changes, 5)
        self.assertEqual(diff.n_changed, 5)
        self.assertEqual(diff.n_boundary_crossings, 1)
        self.assertEqual(diff.regrouping_events, [])
        moved = [move for move in diff.feature_moves if move.face_id == 2]
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0].transition, T_FEATURE_FEATURE)
        self.assertEqual(diff.vanished_instances, [])
        self.assertEqual(diff.appeared_instances, [])

        report = format_partition_diff_report(diff)
        self.assertIn("gate: FAIL (partition changed)", report)
        self.assertIn(
            "boundary crossings (faces changing class/stock/unclaimed): 1",
            report,
        )
        self.assertIn("regrouping: none", report)
        self.assertIn("--- boundary-crossing feature faces (1) ---", report)
        self.assertIn("face 2:", report)
        self.assertNotIn("vanished:", report)
        self.assertNotIn("appeared:", report)

    def test_n_changed_one_face_bucket_change(self):
        golden = _partition({
            "fixture_id": "one_face_bucket",
            "n_faces": 4,
            "instances": [{"class_name": "filleted_pocket", "face_ids": [0, 1, 2]}],
            "stock_face_ids": [3],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(4)],
        })
        fresh = _partition({
            "fixture_id": "one_face_bucket",
            "n_faces": 4,
            "instances": [{"class_name": "filleted_pocket", "face_ids": [0, 1, 2]}],
            "stock_face_ids": [],
            "unclaimed_face_ids": [3],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(4)],
        })
        diff = compare_partitions(golden, fresh)
        self.assertFalse(diff.gate_passed)
        self.assertEqual(diff.n_bucket_changes, 1)
        self.assertEqual(diff.n_changed, 1)
        self.assertEqual(diff.n_boundary_crossings, 1)
        self.assertEqual(diff.regrouping_events, [])

    def test_feature_to_stock_move_fails_and_is_attributed(self):
        golden = _partition({
            "fixture_id": "feature_to_stock",
            "n_faces": 4,
            "instances": [{"class_name": "filleted_pocket", "face_ids": [0, 1]}],
            "stock_face_ids": [2],
            "unclaimed_face_ids": [3],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(4)],
        })
        fresh = _partition({
            "fixture_id": "feature_to_stock",
            "n_faces": 4,
            "instances": [{"class_name": "filleted_pocket", "face_ids": [0]}],
            "stock_face_ids": [1, 2],
            "unclaimed_face_ids": [3],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(4)],
        })
        diff = compare_partitions(golden, fresh)
        self.assertFalse(diff.gate_passed)
        expected_bucket_changes = _expected_bucket_changes(golden, fresh)
        expected_boundary_crossings = _expected_boundary_crossings(golden, fresh)
        # Strict gate: face 1 left the pocket and face 0's instance key shrank.
        self.assertEqual(expected_bucket_changes, {0, 1})
        # Honest signal: only face 1 crossed feature -> stock.
        self.assertEqual(expected_boundary_crossings, {1})
        self.assertEqual(diff.n_bucket_changes, 2)
        self.assertEqual(diff.n_changed, 2)
        self.assertEqual(diff.n_boundary_crossings, 1)
        self.assertEqual(diff.regrouping_events, [])
        face_one_change = next(c for c in diff.face_changes if c.face_id == 1)
        self.assertEqual(face_one_change.transition, T_FEATURE_STOCK)
        self.assertEqual(len(diff.stock_changes), 1)

        report = format_partition_diff_report(diff)
        self.assertIn(
            "boundary crossings (faces changing class/stock/unclaimed): 1",
            report,
        )
        self.assertIn(T_FEATURE_STOCK, report)
        self.assertIn("face 1:", report)
        self.assertIn("filleted_pocket", report)
        self.assertIn("stock", report)

    def test_stock_to_unclaimed_only_still_fails_gate(self):
        golden = _partition({
            "fixture_id": "stock_only",
            "n_faces": 4,
            "instances": [{"class_name": "filleted_pocket", "face_ids": [2, 3]}],
            "stock_face_ids": [0],
            "unclaimed_face_ids": [1],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(4)],
        })
        fresh = _partition({
            "fixture_id": "stock_only",
            "n_faces": 4,
            "instances": [{"class_name": "filleted_pocket", "face_ids": [2, 3]}],
            "stock_face_ids": [1],
            "unclaimed_face_ids": [0],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(4)],
        })
        diff = compare_partitions(golden, fresh)
        self.assertFalse(diff.gate_passed)
        expected_boundary_crossings = _expected_boundary_crossings(golden, fresh)
        # Two faces swap stock <-> unclaimed coarse buckets.
        self.assertEqual(expected_boundary_crossings, {0, 1})
        self.assertEqual(diff.n_bucket_changes, 2)
        self.assertEqual(diff.n_changed, 2)
        self.assertEqual(diff.n_boundary_crossings, 2)
        self.assertEqual(diff.regrouping_events, [])
        self.assertEqual(diff.feature_moves, [])
        for change in diff.stock_changes:
            self.assertEqual(change.transition, T_STOCK_UNCLAIMED)

        report = format_partition_diff_report(diff)
        self.assertIn("gate: FAIL (partition changed)", report)
        self.assertIn(
            "boundary crossings (faces changing class/stock/unclaimed): 2",
            report,
        )
        self.assertIn("regrouping: none", report)
        self.assertIn(f"  (no {T_FEATURE_FEATURE} or feature\u2194unclaimed changes)", report)
        self.assertIn("--- stock partition (attribution summary) ---", report)

    def test_fingerprint_mismatch_raises_before_partition_diff(self):
        golden = _partition({"fixture_id": "fp_mismatch"})
        fresh = _partition({"fixture_id": "fp_mismatch"})
        fresh.face_fingerprints[2] = FaceFingerprint(surface_type="cylinder", area_mm2=9.999)
        with self.assertRaises(FaceFingerprintMismatchError) as ctx:
            compare_partitions(golden, fresh)
        self.assertIn("face_id=2", str(ctx.exception))

        with self.assertRaises(FaceFingerprintMismatchError):
            verify_face_fingerprints(golden.face_fingerprints, fresh.face_fingerprints)

    def test_partition_invariant_overlap(self):
        bad = _partition({
            "instances": [{"class_name": "filleted_pocket", "face_ids": [0, 1, 2]}],
            "stock_face_ids": [2, 5],
        })
        with self.assertRaises(PartitionValidationError) as ctx:
            validate_partition(bad)
        self.assertIn("overlap", str(ctx.exception).lower())

    def test_partition_invariant_gap(self):
        bad = _partition({
            "n_faces": 7,
            "instances": [{"class_name": "filleted_pocket", "face_ids": [0, 1, 2]}],
            "stock_face_ids": [5],
            "unclaimed_face_ids": [],
            "face_fingerprints": [fp.to_dict() for fp in _fingerprints(7)],
        })
        with self.assertRaises(PartitionValidationError) as ctx:
            validate_partition(bad)
        self.assertIn("gap", str(ctx.exception).lower())

    def test_mute_flags_affect_report_only_not_gate(self):
        golden = _partition({"fixture_id": "mute_flags"})
        fresh = _partition({
            "fixture_id": "mute_flags",
            "instances": [
                {"class_name": "filleted_pocket", "face_ids": [0, 1]},
                {"class_name": "filleted_open_pocket", "face_ids": [2, 3, 4]},
            ],
            "stock_face_ids": [5],
        })
        diff = compare_partitions(golden, fresh)
        self.assertFalse(diff.gate_passed)

        full_report = format_partition_diff_report(diff)
        muted_stock = format_partition_diff_report(diff, mute_stock=True)
        muted_features = format_partition_diff_report(diff, mute_features=True)
        muted_both = format_partition_diff_report(
            diff, mute_stock=True, mute_features=True,
        )

        self.assertIn("feature-attributed changes", full_report)
        self.assertIn("stock partition", full_report)
        self.assertNotIn("stock partition", muted_stock)
        self.assertNotIn("feature-attributed changes", muted_features)
        self.assertNotIn("stock partition", muted_both)
        self.assertNotIn("feature-attributed changes", muted_both)

        for report in (full_report, muted_stock, muted_features, muted_both):
            self.assertIn("gate: FAIL (partition changed)", report)

    def test_dump_and_load_round_trip_sorts_instances(self):
        partition = _partition({
            "instances": [
                {"class_name": "z_pocket", "face_ids": [3, 4]},
                {"class_name": "filleted_pocket", "face_ids": [2, 0, 1]},
            ],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.partition.json"
            dump_partition(partition, path)
            loaded = load_partition(path)
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(
                [inst["class_name"] for inst in payload["instances"]],
                ["filleted_pocket", "z_pocket"],
            )
            self.assertEqual(payload["instances"][0]["face_ids"], [0, 1, 2])
            self.assertEqual(len(loaded.face_fingerprints), loaded.n_faces)


if __name__ == "__main__":
    unittest.main()
