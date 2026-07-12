"""Score a valid operation sequence; lower total cost is better.

Cost terms are weighted and individually toggleable via :data:`DEFAULT_SEQUENCE_WEIGHTS`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from operation_bank import FINISH_PHASE_OPS, ROUGH_PHASE_OPS


class _OpLike(Protocol):
    op_id: str
    setup_id: str
    tool_id: str
    operation: str
    feature_refs: Sequence[str]


ROUGH_OPERATION_TYPES = ROUGH_PHASE_OPS
FINISH_OPERATION_TYPES = FINISH_PHASE_OPS


@dataclass(frozen=True)
class SequenceScoreWeights:
    """Per-term weights; set a weight to 0.0 to disable that term."""

    setup_change: float = 100.0
    tool_change: float = 10.0
    approach_change: float = 1.0
    rough_finish_grouping_bonus: float = 2.0

    def as_dict(self) -> dict[str, float]:
        return {
            "setup_change": self.setup_change,
            "tool_change": self.tool_change,
            "approach_change": self.approach_change,
            "rough_finish_grouping_bonus": self.rough_finish_grouping_bonus,
        }


DEFAULT_SEQUENCE_WEIGHTS = SequenceScoreWeights()


@dataclass(frozen=True)
class SequenceScore:
    """Total sequence cost plus per-term breakdown."""

    total: float
    setup_changes: int
    tool_changes: int
    approach_changes: int
    rough_finish_groupings: int
    weighted: dict[str, float] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "setup_changes": self.setup_changes,
            "tool_changes": self.tool_changes,
            "approach_changes": self.approach_changes,
            "rough_finish_groupings": self.rough_finish_groupings,
            "weighted": dict(self.weighted),
        }


def _refs_overlap(left: Sequence[str], right: Sequence[str]) -> bool:
    return bool(set(left) & set(right))


def _is_rough(op: _OpLike) -> bool:
    return op.operation in ROUGH_OPERATION_TYPES


def _is_finish(op: _OpLike) -> bool:
    return op.operation in FINISH_OPERATION_TYPES


def _approach_for_setup(
    setup_id: str,
    setup_approach: Mapping[str, str] | None,
) -> str:
    if setup_approach is None:
        return setup_id
    return setup_approach.get(setup_id, setup_id)


def score_sequence(
    ops: Sequence[_OpLike],
    *,
    setup_approach: Mapping[str, str] | None = None,
    weights: SequenceScoreWeights | None = None,
) -> SequenceScore:
    """Return total weighted cost and raw term counts for *ops* in execution order."""
    w = weights or DEFAULT_SEQUENCE_WEIGHTS
    setup_changes = 0
    tool_changes = 0
    approach_changes = 0
    rough_finish_groupings = 0

    for idx in range(1, len(ops)):
        prev = ops[idx - 1]
        curr = ops[idx]
        if prev.setup_id != curr.setup_id:
            setup_changes += 1
        if prev.tool_id != curr.tool_id:
            tool_changes += 1
        prev_approach = _approach_for_setup(prev.setup_id, setup_approach)
        curr_approach = _approach_for_setup(curr.setup_id, setup_approach)
        if prev_approach != curr_approach:
            approach_changes += 1
        if _is_rough(prev) and _is_finish(curr) and _refs_overlap(prev.feature_refs, curr.feature_refs):
            rough_finish_groupings += 1

    weighted = {
        "setup_change": setup_changes * w.setup_change,
        "tool_change": tool_changes * w.tool_change,
        "approach_change": approach_changes * w.approach_change,
        "rough_finish_grouping_bonus": (
            -rough_finish_groupings * w.rough_finish_grouping_bonus
        ),
    }
    total = sum(weighted.values())
    return SequenceScore(
        total=total,
        setup_changes=setup_changes,
        tool_changes=tool_changes,
        approach_changes=approach_changes,
        rough_finish_groupings=rough_finish_groupings,
        weighted=weighted,
    )


def marginal_transition_cost(
    partial: Sequence[_OpLike],
    next_op: _OpLike,
    *,
    setup_approach: Mapping[str, str] | None = None,
    weights: SequenceScoreWeights | None = None,
) -> float:
    """Incremental cost of appending *next_op* after *partial* (greedy tie-break)."""
    if not partial:
        return 0.0
    return score_sequence(
        [*partial, next_op],
        setup_approach=setup_approach,
        weights=weights,
    ).total - score_sequence(
        partial,
        setup_approach=setup_approach,
        weights=weights,
    ).total
