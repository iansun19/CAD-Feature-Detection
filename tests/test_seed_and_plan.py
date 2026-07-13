"""Unit tests for the seed_and_plan orchestration wrapper (offline; no live DB).

Mirrors tests/test_tool_store.py: an in-memory Supabase double injected via the
`client` param, exercised against the real fish_mold GT graph. The planner/OCC
steps (compute_extents, run_planner) are covered by the live end-to-end run, not
here; these tests pin the persist / verify / coverage / guard logic.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import seed_and_plan as sp  # noqa: E402
import planning.ground_truth_store as gts  # noqa: E402

GT_PATH = Path(ROOT) / "pipeline_out" / "fish_mold_cascade" / "feature_graph_cascade.json"


class _Resp:
    def __init__(self, data=None, count=None):
        self.data = data
        self.count = count


class _Exec:
    def __init__(self, data=None, count=None):
        self._data, self._count = data, count

    def execute(self):
        return _Resp(self._data, self._count)


class _Query:
    def __init__(self, rows, on_delete=None):
        self._rows, self._f, self._n, self._count = rows, [], None, False
        self._delete = False
        self._on_delete = on_delete  # emulate FK cascade

    def select(self, *a, **k):
        self._count = k.get("count") == "exact"
        return self

    def eq(self, col, val):
        self._f.append((col, val))
        return self

    def limit(self, n):
        self._n = n
        return self

    def delete(self):
        self._delete = True
        return self

    def _matched(self):
        rows = self._rows
        for c, v in self._f:
            rows = [r for r in rows if str(r.get(c)) == str(v)]
        return rows

    def execute(self):
        matched = self._matched()
        if self._delete:
            for r in matched:
                self._rows.remove(r)
            if self._on_delete is not None:
                self._on_delete(matched)
            return _Resp(matched)
        count = len(matched) if self._count else None
        return _Resp(matched[: self._n], count)


class InMemorySupabaseClient:
    """Minimal double implementing the rpc + table surface seed_and_plan uses."""

    def __init__(self, tools_count: int = 5):
        self.molds: list[dict] = []
        self.features: list[dict] = []
        self._tools_count = tools_count
        self._seq = 0

    def rpc(self, name, params):
        assert name == gts.INSERT_RPC, name
        self._seq += 1
        mid = f"mold-{self._seq}"
        mold = dict(params["p_mold"])
        mold["id"] = mid
        self.molds.append(mold)
        for f in params["p_features"]:
            row = dict(f)
            row["mold_id"] = mid
            row["id"] = f"feat-{len(self.features)}"
            self.features.append(row)
        return _Exec(mid)

    def _cascade_delete_features(self, deleted_molds):
        ids = {m["id"] for m in deleted_molds}
        self.features = [f for f in self.features if f.get("mold_id") not in ids]

    def table(self, name):
        if name == "molds":
            return _Query(self.molds, on_delete=self._cascade_delete_features)
        if name == "features":
            return _Query(self.features)
        if name == "tools":
            return _Exec(data=[], count=self._tools_count)
        raise AssertionError(f"unexpected table {name}")


def _load_graph():
    return json.loads(GT_PATH.read_text(encoding="utf-8"))


class PersistTests(unittest.TestCase):
    def setUp(self):
        self.graph = _load_graph()
        self.client = InMemorySupabaseClient()

    def _persist(self, force=False):
        return sp.persist(
            self.client,
            self.graph,
            name="fish_mold",
            detection_version="cascade-v6",
            step_file_ref="fixtures/step/fish_mold.stp",
            descriptor=None,
            extents={"min": [0, 0, 0], "max": [1, 1, 1]},
            force=force,
        )

    def test_insert_then_reuse_no_duplicates(self):
        mid1, reused1 = self._persist()
        self.assertFalse(reused1)
        mid2, reused2 = self._persist()
        self.assertTrue(reused2)
        self.assertEqual(mid1, mid2)
        self.assertEqual(len(self.client.molds), 1)  # no pile-up

    def test_force_replaces(self):
        mid1, _ = self._persist()
        n_features = len(self.client.features)
        mid2, reused = self._persist(force=True)
        self.assertFalse(reused)
        self.assertNotEqual(mid1, mid2)
        self.assertEqual(len(self.client.molds), 1)
        # old mold gone; feature count for the single live mold is unchanged
        self.assertEqual(len(self.client.features), n_features)


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.graph = _load_graph()
        self.client = InMemorySupabaseClient()
        self.mold_id, _ = sp.persist(
            self.client, self.graph, name="fish_mold", detection_version="v",
            step_file_ref=None, descriptor=None,
            extents={"min": [0, 0, 0], "max": [1, 1, 1]}, force=False,
        )

    def test_round_trip_ok(self):
        sp.verify_round_trip(self.client, self.mold_id, self.graph)  # no raise

    def test_round_trip_mismatch_raises_with_summary(self):
        for row in self.client.features:  # corrupt a persisted node
            row["metadata"]["class_name"] = "MANGLED"
            break
        with self.assertRaises(sp.PreflightError) as ctx:
            sp.verify_round_trip(self.client, self.mold_id, self.graph)
        self.assertIn("round-trip mismatch", str(ctx.exception))


class CoverageAndGuardTests(unittest.TestCase):
    def setUp(self):
        self.graph = _load_graph()

    def test_fish_mold_has_no_class_unmapped_features(self):
        # Current planner scopes every fish_mold class (incl. inner_fillet via
        # FILLET_CLASSES), so the "no planner rule" gap is empty.
        empty_plan = {"operations": []}
        unmapped_by_class, _ = sp.coverage_gaps(self.graph, empty_plan)
        self.assertEqual(unmapped_by_class, [])

    def test_uncovered_surfaces_inner_fillet_with_reason(self):
        # A plan that covers nothing -> every GT feature is uncovered; inner_fillet
        # must appear, labelled as a scope drop (not a class-unmapped gap).
        empty_plan = {"operations": []}
        _, uncovered = sp.coverage_gaps(self.graph, empty_plan)
        by_class = {u["class_name"] for u in uncovered}
        self.assertIn("inner_fillet", by_class)
        inner = [u for u in uncovered if u["class_name"] == "inner_fillet"]
        self.assertEqual(len(inner), 2)
        for u in inner:
            self.assertEqual(u["reason"], "no-op-in-plan (setup/reachability scope)")

    def test_covered_features_excluded_from_uncovered(self):
        # If an op references a feature_id, it is not reported as uncovered.
        fid = str(self.graph["nodes"][0]["feature_id"])
        plan = {"operations": [{"feature_refs": [fid]}]}
        _, uncovered = sp.coverage_gaps(self.graph, plan)
        self.assertNotIn(fid, {str(u["feature_id"]) for u in uncovered})

    def test_guard_requires_machining_side(self):
        descriptor = {"setups": {"rear": {"opening_axis": {"mode": "explicit"}}}}
        with self.assertRaises(sp.PreflightError) as ctx:
            sp.guard_planning_inputs(self.graph, descriptor, "rear", None)
        self.assertIn("machining-side", str(ctx.exception))

    def test_guard_names_rear_when_setup_missing(self):
        descriptor = {"setups": {"default": {}}}
        with self.assertRaises(sp.PreflightError) as ctx:
            sp.guard_planning_inputs(self.graph, descriptor, "rear", "back")
        self.assertIn("rear", str(ctx.exception))

    def test_guard_passes_with_valid_inputs(self):
        descriptor = {"setups": {"rear": {"opening_axis": {"mode": "explicit"}}}}
        sp.guard_planning_inputs(self.graph, descriptor, "rear", "back")  # no raise


class EnvTests(unittest.TestCase):
    def test_check_env_missing(self):
        with mock.patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_SERVICE_ROLE_KEY": ""}, clear=False):
            with self.assertRaises(sp.PreflightError) as ctx:
                sp.check_env()
        self.assertIn("SUPABASE_URL", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
