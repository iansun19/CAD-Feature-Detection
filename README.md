# mlcad — CAD → CAM machining-plan pipeline

This repo turns a solid-model part (a STEP file or an MFCAD++ sample) into a
structured, machinable process plan. It has two engines that share one B-rep
front end:

1. **Perception** — recognize machining features on each face.
2. **Planning** — group features into setups, assign tools/parameters, order the
   operations, and emit a CAM plan.

Perception ships in two forms: a trained per-face **GNN classifier** (the
original MFCAD++ "Path 1") and a geometric **cascade** that is the real engine on
industrial parts. Everything downstream of perception (machining context,
reachability, planner, sequencing) is built on the cascade output.

> **Status (2026-07-08).** For the defined scope — one orientation per STEP file,
> 3-axis milling, heuristic engine — the pipeline runs end to end and emits a
> validated CAM plan. Known open work is scope expansion and generalization, most
> of which is gated on new labeled parts. See [Roadmap & status](#roadmap--status).

---

## Two ways in

### A. CAM pipeline (cascade → context → planner) — the current focus

```
STEP file
  └─ run_cascade.py       B-rep → feature graph (cascade) → approach vectors
                          → verified 3-axis reachability → setup descriptor
  └─ machining_context.py feature graph + setup YAML + stock → planner input contract
  └─ planner.py           features → operations → precedence → sequencing → CamPlan
```

Typical run on a split-panel part (front/rear are separate STEP files, one
orientation each):

```bash
conda activate mlcad
python run_cascade.py "fixtures/step/96260B_rear.stp" --export-dir pipeline_out/96260B_rear
python run_cascade.py "fixtures/step/96260B_front.stp" --export-dir pipeline_out/96260B_front
python planner.py --multi-setup --setups generated --scope-diff
```

### B. MFCAD++ GNN face classifier ("Path 1") — the original model

A minimal PyTorch Geometric pipeline for per-face feature classification on the
MFCAD++ dataset. Node = B-rep face, edge = shared edge; output = one class label
per face. See [GNN classifier](#gnn-classifier-path-1) below.

---

## Pipeline stages

### Stage 1 — Perception

**Cascade** ([run_cascade.py]). Specific-first feature recognition: the more
specific recognizer runs first and claims its faces; the generic one takes the
residual. Order:

```
pockets → holes → coaxial_stack → flats → outer_fillets → wall → profile → residual
```

Running pockets before holes stops a hole recognizer from grabbing concave pocket
walls as false "holes". Each pass is independent — the cascade only threads one
pass's `remaining_faces` into the next as its candidate pool. Detection modules:
[pocket_detection.py], [hole_detection.py], [coaxial_stack_detection.py],
[flats_detection.py], [outer_fillet_detection.py], [inner_fillet_detection.py],
[wall_detection.py], [profile_detection.py], [residual_detection.py],
[lobe_tier_detection.py], [lobe_contour_merge.py], [exterior_boundary.py].

The export path writes `feature_graph_cascade.json` and, layered on top:

- **Approach vectors** ([approach_vectors.py]) — per-feature 3-axis approach
  direction (Z-only MVP). Purely geometric candidacy; no collision test. Bumps
  the graph `schema_version` to 3.
- **Verified reachability** ([reachability.py]) — a swept-tool-cylinder collision
  test against the real solid, upgrading candidacy to verified reachability. Bumps
  `schema_version` to 4 and adds `reachability_summary`. Walls are exempt (lateral
  tool access, not axial plunge).

**GNN classifier** ([legacy/model.py], [legacy/train.py]) — the trained alternative; see below.

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

---

## Roadmap & status

6-step plan (dependency-ordered). "Done" = in-scope MVP built and green.

1. **Generalize cascade** — de-scaling ([part_scale.py]) and lobe-tier
   concentric-bore relationship done, validated byte-identical on 96260B. *Open:
   labeled new test parts (data-gated).*
2. **Per-feature approach vectors** — Z-only MVP done. *Open: lateral ±X/±Y axes
   (needs a side-access part).*
3. **Candidate setup generation** — generation (interpretation A) done. *Blocked:
   setup-count minimization — needs lateral axes + a unified single-part model.*
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

## GNN classifier (Path 1)

A single face-adjacency graph GNN (UV-Net-style, simplified; GINEConv stack with
edge features + residuals).

- [legacy/dataset.py] — loads MFCAD++ into PyG `Data` graphs. **Two loaders:** one for
  prebuilt H5 graphs, one that parses STEP via pythonocc. Read the comments and
  confirm which matches your files — field-name mismatch is the #1 cause of silent
  label misalignment.
- [legacy/model.py] — the GNN.
- [legacy/train.py] — training loop with early stopping, checkpointing, logging.
- [legacy/overfit_check.py] — fast sanity run on ~20 samples. **Run this first.**
- [config.yaml] — all hyperparameters.

**Order of operations (do not skip):**

1. Install env (`requirements.txt`, or the unified `environment.yml` for OCC).
2. Download MFCAD++ (~1.5 GB) from
   [Queen's University Belfast](https://pure.qub.ac.uk/en/datasets/mfcad-dataset-dataset-for-paper-hierarchical-cadnet-learning-from/)
   and unzip into `MFCAD++_dataset/` at the repo root.
3. `python -m legacy.setup_data` — confirms `train.txt` / `val.txt` / `test.txt` and the H5 are present.
4. `python -m legacy.overfit_check` — must reach ~100% train acc on 20 parts in a couple
   minutes. If it can't memorize 20 parts, the data loader is wrong. Fix before step 5.
5. `python -m legacy.train` overnight.
6. Read `runs/<timestamp>/log.txt` and `best_model.pt`.

**Mac / Apple Silicon:** use the minimal Mac install in `requirements.txt`
(`torch` + `torch_geometric` only — no `torch_scatter` / `torch_sparse`). Device
auto-selects MPS (`config.yaml: device: auto`); the overfit check prints
`device=mps` at startup. Set `device: cpu` if you hit a rare MPS op gap.

> **Data-hygiene warning:** in MFCAD++ STEP files the `ADVANCED_FACE` name field
> equals the ground-truth label. Strip it before any raw-text or LLM use, or you
> leak labels.

---

## Environments

- **CAM pipeline (perception → planning):** needs pythonocc-core (OpenCASCADE).
  `conda env create -f environment.yml && conda activate mlcad`.
- **GNN training only:** the minimal `requirements.txt` (PyTorch + PyG) is enough.
- The repo `.venv` lacks OCC; run cascade / reachability / context tests under the
  conda `mlcad` env (`python -m unittest ...`).

## Tests

```bash
# Cascade golden regression (byte-identical safety net) — conda mlcad env
/Users/iansun19/miniconda3/envs/mlcad/bin/python -m unittest tests.test_cascade_regression

# Focused suites
python -m unittest tests.test_planner tests.test_reachability tests.test_sequence_search
```
