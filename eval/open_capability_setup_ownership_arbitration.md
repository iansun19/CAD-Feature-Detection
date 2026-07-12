# Open capability: setup-ownership-arbitration

Date opened: 2026-07-12. Status: OPEN. No code in this task attempts to fix it.

## The gap

Setup scope is now geometry-derived: `setup_generation.derive_setup_scope`
(see `setup_generation.py:156-160`) scopes a setup by *what it can reach*, never
by the part's filename. That was the right fix — `default_scope_for_side` deciding
scope from the STEP filename was wrong and is gone.

But "what it can reach" is not "what it should cut". On 96260B, walls that are the
same physical diameter are reachable from **both** the front and rear approaches.
Reachability grants each setup all of that shared geometry, so the front emits its
full 10-op milling program even though the shop runs the front (Setup 2) as a
single facing flip and cuts those walls once, from the rear.

Reachability is **necessary but not sufficient**. Nothing arbitrates which setup
*owns* geometry that both setups can reach. Until something does, geometry-derived
scope over-grants, and the plan over-machines.

## How it currently surfaces (known-red true positives on 96260B)

- `per_setup_op_count` HARD: `setup front: emitted 10 ops vs shop setup 1 ops`
  (`scripts/plan_sanity_report.py`, front emitted > shop total). This is the HARD
  over-machining signal.
- `coverage_expectations` verdict FAIL: the front emits 5 milling strategies
  (roughing + finishing_*) that the manifest marks `expected_absent` (front is
  facing-only per shop / rear owns the geometry). These are WARN-level `unexpected`
  strategies that flip the coverage verdict to FAIL. Facing itself is correct:
  rear owns facing (envelope stock flat feat 17 / face 322), front faces nothing.
- Failing tests (correct, do not "fix" by relaxing the gate):
  `test_plan_sanity_report.py::Test96260BCleanPlan::test_current_plan_passes_hard_gates`
  and `TestCoverageExpectations::test_current_plan_coverage_verdict_matches_both_setups`.

These reds are the missing arbitration talking. Do NOT re-downgrade the gate or
re-flip the coverage GT to silence them.

## Relationship to homolog-map-split-panel

Same missing capability seen from the other end. Arbitration needs to know which
faces across setups are the same physical geometry — that is exactly the
world-space homolog map. See `open_capability_homolog_map_split_panel.md`.
