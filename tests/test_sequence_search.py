"""Tests for score_sequence.py and sequence_search.py."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cam_plan_schema import MachiningParameters  # noqa: E402
from machining_context import load_feature_graph  # noqa: E402
from planner import (  # noqa: E402
    OpSpec,
    build_precedence,
    sequence,
)
from score_sequence import score_sequence  # noqa: E402
from sequence_search import (  # noqa: E402
    search_sequence,
    validate_sequence_precedence,
)


def _op(
    op_id: str,
    *,
    setup_id: str = "rear",
    tool_id: str = "T01",
    operation: str = "pocket",
    feature_refs: list[str] | None = None,
    tool_type_needed: str = "endmill",
) -> OpSpec:
    return OpSpec(
        op_id=op_id,
        feature_refs=feature_refs or [op_id],
        setup_id=setup_id,
        operation=operation,
        tool_id=tool_id,
        tool_type_needed=tool_type_needed,
        parameters=MachiningParameters(param_source="handbook_default"),
    )


def _machined_faces(plan, graph_refs: dict[str, str]) -> set[int]:
    faces: set[int] = set()
    for sid, ref in graph_refs.items():
        graph = load_feature_graph(Path(ROOT) / ref)
        by_id = {str(node["feature_id"]): node for node in graph["nodes"]}
        for op in plan.operations:
            if op.setup_id != sid:
                continue
            for fid in op.feature_refs:
                node = by_id.get(fid)
                if node:
                    faces.update(node.get("face_ids", []))
    return faces


class TestScoreSequence(unittest.TestCase):
    def test_setup_changes_dominate_ping_pong(self) -> None:
        ops = [
            _op("A", setup_id="rear", tool_id="T01"),
            _op("B", setup_id="front", tool_id="T01"),
            _op("C", setup_id="rear", tool_id="T01"),
        ]
        grouped = [
            _op("A", setup_id="rear", tool_id="T01"),
            _op("D", setup_id="rear", tool_id="T01"),
            _op("E", setup_id="front", tool_id="T01"),
            _op("F", setup_id="front", tool_id="T01"),
        ]
        ping_pong = score_sequence(ops).total
        block = score_sequence(grouped).total
        self.assertLess(block, ping_pong)
        self.assertEqual(score_sequence(ops).setup_changes, 2)
        self.assertEqual(score_sequence(grouped).setup_changes, 1)

    def test_tool_changes_counted_within_setup(self) -> None:
        ops = [
            _op("A", tool_id="T01"),
            _op("B", tool_id="T02"),
            _op("C", tool_id="T02"),
        ]
        scored = score_sequence(ops)
        self.assertEqual(scored.tool_changes, 1)


class TestSequenceSearch(unittest.TestCase):
    def _chain_ops(self) -> list[OpSpec]:
        return [
            _op("OP010", setup_id="rear", tool_id="T05", feature_refs=["1"]),
            _op("OP020", setup_id="rear", tool_id="T07", feature_refs=["2"]),
            _op("OP030", setup_id="front", tool_id="T05", feature_refs=["3"]),
            _op("OP040", setup_id="front", tool_id="T07", feature_refs=["4"]),
        ]

    def test_validity_by_construction(self) -> None:
        ops = self._chain_ops()
        precedence = {
            "OP010": [],
            "OP020": ["OP010"],
            "OP030": ["OP020"],
            "OP040": ["OP030"],
        }
        for strategy in ("none", "greedy", "beam"):
            with self.subTest(strategy=strategy):
                result = search_sequence(ops, precedence, strategy=strategy)
                validate_sequence_precedence(result.ordered, precedence)
                self.assertEqual(
                    {op.op_id for op in result.ordered},
                    {op.op_id for op in ops},
                )

    def test_beam_le_greedy_le_none_on_tool_swaps(self) -> None:
        ops = [
            _op("OP010", setup_id="rear", tool_id="T05"),
            _op("OP020", setup_id="rear", tool_id="T05", feature_refs=["2"]),
            _op("OP030", setup_id="rear", tool_id="T07", feature_refs=["3"]),
            _op("OP040", setup_id="rear", tool_id="T07", feature_refs=["4"]),
            _op("OP050", setup_id="front", tool_id="T05", feature_refs=["5"]),
            _op("OP060", setup_id="front", tool_id="T05", feature_refs=["6"]),
        ]
        precedence = {op.op_id: [] for op in ops}
        precedence["OP050"] = ["OP040"]
        precedence["OP060"] = ["OP050"]

        none = search_sequence(ops, precedence, strategy="none").score.total
        greedy = search_sequence(ops, precedence, strategy="greedy").score.total
        beam = search_sequence(ops, precedence, strategy="beam", beam_width=8).score.total
        self.assertLessEqual(greedy, none + 1e-9)
        self.assertLessEqual(beam, greedy + 1e-9)

    def test_precedence_never_violated_for_planner_rules(self) -> None:
        ops = [
            _op("OP010", operation="pocket", feature_refs=["1"]),
            _op(
                "OP020",
                operation="contour_2d",
                feature_refs=["1"],
            ),
            _op("OP030", operation="drill", feature_refs=["2"], tool_type_needed="drill"),
            _op("OP040", operation="drill", feature_refs=["2"], tool_type_needed="tap"),
        ]
        precedence = build_precedence(ops)
        result = search_sequence(ops, precedence, strategy="beam")
        validate_sequence_precedence(result.ordered, precedence)
        rough_idx = next(
            i for i, op in enumerate(result.ordered) if op.operation == "pocket"
        )
        finish_idx = next(
            i for i, op in enumerate(result.ordered) if op.operation == "contour_2d"
        )
        drill_idx = next(
            i
            for i, op in enumerate(result.ordered)
            if op.operation == "drill" and op.tool_type_needed == "drill"
        )
        tap_idx = next(
            i for i, op in enumerate(result.ordered) if op.tool_type_needed == "tap"
        )
        self.assertLess(rough_idx, finish_idx)
        self.assertLess(drill_idx, tap_idx)

    def test_geometry_signature_unchanged_across_strategies(self) -> None:
        ops = self._chain_ops()
        precedence = {op.op_id: [] for op in ops}
        precedence["OP030"] = ["OP010"]
        precedence["OP040"] = ["OP020"]

        def signature(ordered: list[OpSpec]) -> set[tuple]:
            return {
                (
                    tuple(op.feature_refs),
                    op.setup_id,
                    op.operation,
                    op.tool_id,
                )
                for op in ordered
            }

        signatures = [
            search_sequence(ops, precedence, strategy=strategy).ordered
            for strategy in ("none", "greedy", "beam")
        ]
        baseline = signature(sequence(ops, precedence))
        for ordered in signatures:
            self.assertEqual(signature(ordered), baseline)


class TestSequenceSearch96260B(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rear_graph = Path(ROOT) / "pipeline_out" / "96260B_rear" / "feature_graph_cascade.json"
        cls.front_graph = Path(ROOT) / "pipeline_out" / "96260B_front" / "feature_graph_cascade.json"
        cls.setup_yaml = Path(ROOT) / "eval" / "gt" / "96260B_setup.yaml"
        if not cls.rear_graph.is_file() or not cls.front_graph.is_file():
            raise unittest.SkipTest("96260B cascade artifacts missing")

    def test_multi_setup_scores_improve_with_search(self) -> None:
        from machining_context import build_context_v0  # noqa: PLC0415
        from planner import SetupPlanInput, plan_multi_setups  # noqa: PLC0415

        rear_step = Path(ROOT) / "96260B_REAR_XR004_PCD PLATE.stp copy"
        front_step = Path(ROOT) / "96260B_FRONT_XR004_PCD PLATE.stp copy"
        ctx_kwargs = {
            "material": "aluminum",
            "tool_source": "hardcoded",
            "setups_source": "authored",
        }
        try:
            rear_ctx = build_context_v0(
                rear_step,
                self.setup_yaml,
                self.rear_graph,
                setup_id="rear",
                **ctx_kwargs,
            )
            front_ctx = build_context_v0(
                front_step,
                self.setup_yaml,
                self.front_graph,
                setup_id="front",
                **ctx_kwargs,
            )
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        inputs = [
            SetupPlanInput(self.rear_graph, rear_ctx),
            SetupPlanInput(self.front_graph, front_ctx),
        ]
        plans = {
            strategy: plan_multi_setups(
                inputs,
                setup_order=("rear", "front"),
                source_part="96260B",
                seq_search=strategy,
            )
            for strategy in ("none", "greedy", "beam")
        }

        def work_signature(plan) -> set[tuple]:
            return {
                (
                    tuple(sorted(op.feature_refs)),
                    op.setup_id,
                    op.operation,
                    op.tool_id,
                )
                for op in plan.operations
            }

        baseline_sig = work_signature(plans["none"])
        for strategy in ("greedy", "beam"):
            self.assertEqual(work_signature(plans[strategy]), baseline_sig)

        none_score = plans["none"].metadata["sequence_score"]["total"]
        greedy_score = plans["greedy"].metadata["sequence_score"]["total"]
        beam_score = plans["beam"].metadata["sequence_score"]["total"]
        self.assertLessEqual(greedy_score, none_score + 1e-9)
        self.assertLessEqual(beam_score, greedy_score + 1e-9)

        graph_refs = plans["none"].metadata.get("feature_graph_refs", {})
        baseline_faces = _machined_faces(plans["none"], graph_refs)
        for plan in plans.values():
            self.assertEqual(_machined_faces(plan, graph_refs), baseline_faces)
        self.assertEqual(len(baseline_faces), 257)


if __name__ == "__main__":
    unittest.main()
