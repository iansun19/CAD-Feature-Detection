"""Search over valid topological operation orders (greedy + beam)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

logger = logging.getLogger(__name__)

from planning.score_sequence import (
    DEFAULT_SEQUENCE_WEIGHTS,
    SequenceScore,
    SequenceScoreWeights,
    marginal_transition_cost,
    score_sequence,
)

SeqSearchStrategy = Literal["none", "greedy", "beam"]


@dataclass(frozen=True)
class SequenceSearchResult:
    ordered: list[Any]
    score: SequenceScore
    strategy: SeqSearchStrategy


def _by_id(ops: Sequence[Any]) -> dict[str, Any]:
    return {op.op_id: op for op in ops}


def _build_graph(
    op_ids: Sequence[str],
    precedence: Mapping[str, Sequence[str]],
) -> tuple[dict[str, int], dict[str, list[str]]]:
    in_degree = {op_id: 0 for op_id in op_ids}
    dependents: dict[str, list[str]] = {op_id: [] for op_id in op_ids}
    known = set(op_ids)
    for op_id in op_ids:
        for dep in precedence.get(op_id, ()):
            if dep not in known or dep == op_id:
                continue
            in_degree[op_id] += 1
            dependents[dep].append(op_id)
    return in_degree, dependents


def ready_op_ids(
    remaining: set[str],
    completed: set[str],
    precedence: Mapping[str, Sequence[str]],
) -> list[str]:
    ready: list[str] = []
    for op_id in remaining:
        deps = precedence.get(op_id, ())
        if all(dep in completed for dep in deps):
            ready.append(op_id)
    return ready


def validate_sequence_precedence(
    ordered: Sequence[Any],
    precedence: Mapping[str, Sequence[str]],
) -> None:
    """Raise ValueError when any dependency appears after its dependent."""
    position = {op.op_id: idx for idx, op in enumerate(ordered)}
    violations: list[str] = []
    for op in ordered:
        for dep in precedence.get(op.op_id, ()):
            if dep not in position:
                continue
            if position[dep] >= position[op.op_id]:
                violations.append(f"{op.op_id} depends on {dep}")
    if violations:
        raise ValueError(
            "sequence violates precedence: " + "; ".join(sorted(violations))
        )


def _search_greedy(
    ops: Sequence[Any],
    precedence: Mapping[str, Sequence[str]],
    *,
    setup_approach: Mapping[str, str] | None,
    weights: SequenceScoreWeights,
    tie_break_key,
) -> list[Any]:
    by_id = _by_id(ops)
    remaining = set(by_id)
    completed: set[str] = set()
    ordered: list[Any] = []

    while remaining:
        ready = ready_op_ids(remaining, completed, precedence)
        if not ready:
            raise ValueError(
                "precedence graph blocked search; remaining ops: "
                + ", ".join(sorted(remaining))
            )
        ready.sort(
            key=lambda op_id: (
                marginal_transition_cost(
                    ordered,
                    by_id[op_id],
                    setup_approach=setup_approach,
                    weights=weights,
                ),
                tie_break_key(by_id[op_id]),
            )
        )
        chosen = ready[0]
        ordered.append(by_id[chosen])
        remaining.remove(chosen)
        completed.add(chosen)

    return ordered


def _search_beam(
    ops: Sequence[Any],
    precedence: Mapping[str, Sequence[str]],
    *,
    beam_width: int,
    setup_approach: Mapping[str, str] | None,
    weights: SequenceScoreWeights,
    tie_break_key,
) -> list[Any]:
    by_id = _by_id(ops)
    all_ids = set(by_id)
    if not ops:
        return []

    # Beam state: (ordered_ids, completed_set, score_total)
    beam: list[tuple[tuple[str, ...], frozenset[str], float]] = [
        ((), frozenset(), 0.0)
    ]
    best_complete: tuple[tuple[str, ...], float] | None = None

    while beam:
        next_beam: list[tuple[tuple[str, ...], frozenset[str], float]] = []
        for prefix, completed, _ in beam:
            remaining = all_ids - set(completed)
            if not remaining:
                score = score_sequence(
                    [by_id[op_id] for op_id in prefix],
                    setup_approach=setup_approach,
                    weights=weights,
                ).total
                if best_complete is None or score < best_complete[1]:
                    best_complete = (prefix, score)
                continue

            ready = ready_op_ids(remaining, completed, precedence)
            if not ready:
                continue

            for op_id in ready:
                new_prefix = (*prefix, op_id)
                ordered_partial = [by_id[i] for i in new_prefix]
                score = score_sequence(
                    ordered_partial,
                    setup_approach=setup_approach,
                    weights=weights,
                ).total
                next_beam.append(
                    (new_prefix, frozenset((*completed, op_id)), score)
                )

        if not next_beam:
            break

        next_beam.sort(
            key=lambda item: (
                item[2],
                tie_break_key(by_id[item[0][-1]]),
            )
        )
        beam = next_beam[: max(1, beam_width)]

    if best_complete is None:
        raise ValueError(
            "beam search found no complete sequence; precedence graph may be blocked"
        )
    return [by_id[op_id] for op_id in best_complete[0]]


def search_sequence(
    ops: Sequence[Any],
    precedence: Mapping[str, Sequence[str]],
    *,
    strategy: SeqSearchStrategy = "beam",
    beam_width: int = 5,
    setup_approach: Mapping[str, str] | None = None,
    weights: SequenceScoreWeights | None = None,
    tie_break_key=None,
    topo_sort_fn=None,
    tool_lookup: Mapping[str, Any] | None = None,
) -> SequenceSearchResult:
    """Choose a precedence-respecting order using *strategy*.

    *topo_sort_fn* defaults to :func:`planner.sequence` when ``strategy='none'``.
    """
    if not ops:
        empty_score = score_sequence([], setup_approach=setup_approach, weights=weights)
        return SequenceSearchResult(ordered=[], score=empty_score, strategy=strategy)

    w = weights or DEFAULT_SEQUENCE_WEIGHTS
    if tie_break_key is None:
        tie_break_key = lambda op: (op.op_id,)  # noqa: E731

    def _topo() -> list[Any]:
        fn = topo_sort_fn
        if fn is None:
            from planner import sequence as fn  # noqa: PLC0415
        return list(fn(ops, precedence, tool_lookup=tool_lookup or {}))

    if strategy == "none":
        ordered = _topo()
    elif strategy in ("greedy", "beam"):
        search = _search_greedy if strategy == "greedy" else _search_beam
        kwargs = dict(
            setup_approach=setup_approach, weights=w, tie_break_key=tie_break_key
        )
        if strategy == "beam":
            kwargs["beam_width"] = beam_width
        try:
            ordered = search(ops, precedence, **kwargs)
        except ValueError:
            # Heuristic search can strand a valid order under tight precedence
            # (e.g. a whole-setup deburr that depends on every other op). The
            # topological sort always yields a precedence-respecting order.
            logger.warning(
                "%s sequence search found no complete order; falling back to topo sort",
                strategy,
            )
            ordered = _topo()
    else:
        raise ValueError(f"unknown seq-search strategy: {strategy!r}")

    validate_sequence_precedence(ordered, precedence)
    score = score_sequence(ordered, setup_approach=setup_approach, weights=w)
    return SequenceSearchResult(ordered=ordered, score=score, strategy=strategy)
