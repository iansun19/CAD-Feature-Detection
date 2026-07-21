# mlcad — CAD → CAM machining-plan pipeline

This repo turns a solid-model part (a STEP file) into a structured, machinable
process plan. It is a **STEP → JSON** pipeline: ingest a B-rep, recognize
machining features geometrically, and emit a validated CAM plan.

Three stages run in sequence, sharing one B-rep front end:

1. **Perception** — a geometric **cascade** recognizes machining features on each
   face, then layers approach vectors and verified 3-axis reachability on top.
2. **Machining context** — assemble the planner input contract from the STEP
   extents, a setup descriptor, and the cascade graph.
3. **Planning** — group features into operations, enforce precedence, assign
   tools/parameters, order the operations, and emit a CAM plan.

> **Status (2026-07-12).** For the defined scope — one orientation per STEP file,
> 3-axis milling, heuristic engine — the pipeline runs end to end and emits a
> validated CAM plan. Known open work is scope expansion and generalization, most
> of which is gated on new labeled parts. See [Roadmap & status](#roadmap--status).

---

## Quick start

One command, arbitrary part, STEP in → CamPlan JSON out:

```bash
conda activate mlcad
python run_step_to_plan.py <part.step> [--opening-axis +Z] [--machining-side ...]
```

[run_step_to_plan.py] chains the three stages for **any** part (no
96260B / fish_mold hardcoding): the cascade ingests the face graph straight from
the STEP (no pre-built `graph.npz` needed) and writes a `feature_graph_cascade.json`
plus a single generated `setup_descriptor.yaml`; context derives the stock
envelope from the B-rep extents; the planner emits the CamPlan.

A lone STEP is its own single-setup part. Separate STEP files are **separate
parts** — e.g. `96260B_front.stp` and `96260B_rear.stp` are two independent parts,
each planned on its own. For a genuine flip-job (one part, one stock, refixtured
across setups) declared by an explicit multi-setup descriptor, use
`planner.py --multi-setup --setup-yaml <descriptor>`.

The equivalent staged invocation (useful when inspecting intermediates):

```bash
python run_cascade.py "fixtures/step/96260B_rear.stp"  --export-dir pipeline_out/96260B_rear
python run_cascade.py "fixtures/step/96260B_front.stp" --export-dir pipeline_out/96260B_front
python planner.py --multi-setup --setups generated --scope-diff
```

```
STEP file
  └─ run_cascade.py       B-rep → feature graph (cascade) → approach vectors
                          → verified 3-axis reachability → setup descriptor
  └─ machining_context.py feature graph + setup YAML + stock → planner input contract
  └─ planner.py           features → operations → precedence → sequencing → CamPlan
```

---

## Pipeline stages

### Stage 1 — Perception (the cascade)

**Cascade** ([run_cascade.py]). Specific-first feature recognition: the more
specific recognizer runs first and claims its faces; the generic one takes the
residual. Order:

```
pockets → holes → coaxial_stack → flats → outer_fillets → wall → profile → residual
```

Running pockets before holes stops a hole recognizer from grabbing concave pocket
walls as false "holes"; the prismatic-profile pass runs before holes so an outer
boundary reads as one profile. Each pass is independent — the cascade only threads
one pass's `remaining_faces` into the next as its candidate pool. A terminal-pass
invariant raises if any face is left unclaimed. Detection modules:
[pocket_detection.py], [hole_detection.py], [coaxial_stack_detection.py],
[flats_detection.py], [outer_fillet_detection.py], [inner_fillet_detection.py],
[wall_detection.py], [profile_detection.py], [residual_detection.py],
[lobe_tier_detection.py], [exterior_boundary.py].

The export path writes `feature_graph_cascade.json` and, layered on top:

- **Approach vectors** ([approach_vectors.py]) — per-feature 3-axis approach
  direction. Purely geometric candidacy; no collision test. Bumps the graph
  `schema_version` to 3.
- **Verified reachability** ([reachability.py]) — a swept-tool-cylinder collision
  test against the real solid, upgrading candidacy to verified reachability. Bumps
  `schema_version` to 4 and adds `reachability_summary`. Walls are exempt (lateral
  tool access, not axial plunge).

Feature classes recognized include pockets, through/blind holes (drill-tip blind
holes discriminated from countersinks), coaxial bore stacks, counterbores,
countersinks, flats, inner/outer fillets, walls, and prismatic profiles.

### Stage 2 — Machining context

[machining_context.py] assembles the planner input contract (Pydantic v2) from the
STEP extents, the setup descriptor YAML, and the cascade graph: stock envelope,
per-setup facts, tool catalog. v0 assumptions are documented at the top of the
module — single orientation per slice, 3-axis discrete opening axis,
`remaining_material` always `null` (no in-process stock tracking — see
[Roadmap & status](#roadmap--status)), `completed_operations` always empty.

The **setup descriptor** ([setup_descriptor.py]) is the sole source of truth for
pocket open/closed access — geometry cannot recover it. Descriptors can be
hand-authored (`eval/gt/96260B_setup.yaml`) or synthesized from a cascade run by
[setup_generation.py] (geometric facts derived; `machining_side` and `scope` are
carried-through process inputs).

Real Fusion tool libraries are ingested and queried via [tool_store.py] /
[tool_library_store.py] (`scripts/ingest_tool_libraries.py`).

### Stage 3 — Planning

[planner.py] maps features to operations from the canonical [operation_bank.py],
enforces precedence, selects tools and parameters, orders the operations, and
emits a **CamPlan** ([cam_plan_schema.py]). Parameters come from a shop
`toolpath_preset` when available, else a diameter-keyed `handbook_default`.

Operation ordering runs through the sequencing search ([sequence_search.py] +
[score_sequence.py]): strategies `none` / `greedy` / `beam` (default, width 5).
The scorer weights `setup_change (100) > tool_change (10) > approach_change (1)`
plus a rough→finish grouping bonus. Every path validates precedence, and search
only reorders — it never changes the work (identical machined faces across all
strategies).

---

## Output: what a finished plan contains

`planner.py` writes a `CamPlan` (see [cam_plan_schema.py]) with:

- **Setups** — id, opening axis, fixture (name only), source STEP file.
- **Operations** (ordered) — feature id(s), operation type, tool ref, machining
  parameters (rpm / feed / plunge / stepdown / stepover), source
  (`toolpath_preset` or `handbook_default`).
- **Tools** — resolved tool refs; unresolved tools are flagged `UNRESOLVED`.
- **Metadata** — `planner_stats`, `sequence_score`, schema/version info.
- `remaining_material` — currently always `null` by decision (no validated
  in-process stock model yet).

Persistence: molds and their features round-trip through Supabase; a saved mold
can be replanned without re-running the cascade via [generate_operation_plan.py]
(reconstructs the cascade graph from stored features → context → planner).

---

## Roadmap & status

6-step plan (dependency-ordered). "Done" = in-scope MVP built and green.

1. **Generalize cascade** — de-scaling ([part_scale.py]) and lobe-tier
   concentric-bore relationship done, validated byte-identical on 96260B. *Open:
   labeled new test parts (data-gated).*
2. **Per-feature approach vectors** — Z-only MVP done; six-cardinal ±X/±Y/±Z
   approach/reachability landed but unvalidated. *Open: a validated side-access
   part.*
3. **Candidate setup generation** — generation done. *Blocked: setup-count
   minimization — needs lateral axes + a unified single-part model.*
4. **Machining-state completion** — (a) verified reachability done; (b)
   `remaining_material` **descoped to `null`**: no validated toolpath layer or
   ground-truth reference to certify accuracy against.
5. **Sequencing search** — beam / greedy / none done.
6. **NN upgrades** — last; op-mapper and sequence scorer built pluggable for a
   future learned model. Not started.

**How close:** for the defined scope (single orientation per file, 3-axis,
heuristic), essentially feature-complete. For the full ambition (cross-part
generalization, minimal multi-orientation setups, learned components), the
critical-path blocker is **new labeled parts** — especially one with side
features — which unblocks items 1, 2, and 3 at once.

The safety net for all cascade changes is the byte-identical golden regression on
96260B (`tests/test_cascade_regression.py`).

---

## Environments

The pipeline needs pythonocc-core (OpenCASCADE):

```bash
conda env create -f environment.yml && conda activate mlcad
```

The repo `.venv` lacks OCC; run cascade / reachability / context tests under the
conda `mlcad` env. If `conda run` gets hijacked by the repo `.venv`, invoke the
mlcad python by absolute path.

## Tests

```bash
# Cascade golden regression (byte-identical safety net) — conda mlcad env
/Users/iansun19/miniconda3/envs/mlcad/bin/python -m unittest tests.test_cascade_regression

# Focused suites
python -m unittest tests.test_planner tests.test_reachability tests.test_sequence_search
```
