# Phase 0 audit — planner setup source & feature scoping

Date: 2026-07-08. Read-only review before wiring generated setups and verified
reachability into the planner.

## a. Hand-authored setup YAML — load & consumption

| Location | Role |
|----------|------|
| `machining_context.build_context_v0()` | **Primary planner entry.** `load_setup_descriptor(setup_yaml_path)` ? `resolve_setup_entry()` ? `build_setup_context()`. Default path: `eval/gt/96260B_setup.yaml`. |
| `planner.py` `__main__` | Hard-codes `eval/gt/96260B_setup.yaml`; passes it to `build_context_v0` for rear-only and `--multi-setup` (rear + front). |
| `machining_context.build_setup_context()` | Uses resolved descriptor for `opening_axis`, per-pocket `pocket_access` (descriptor wins over cascade `params.access` with warning), and `SetupContext.scope`. |
| `run_cascade._resolve_pocket_setup_for_cascade()` | Cascade pocket pass reads the same YAML (default `96260B_setup.yaml`) for `PocketSetupConfig` — **not consumed by the planner**, but keeps cascade pocket access aligned with the descriptor. |

**Not planner wiring:** `eval/gt/96260B_{front,rear}.yaml` are toolpath GT for eval, not setup descriptors.

**`run_pipeline.py`:** Phase 4 GNN ingest only (STEP ? `feature_graph.json`). It does **not** load setup YAML or invoke the planner. CAM planning is orchestrated via `run_cascade.py --export-dir` (cascade + reachability) and `planner.py` (context + plan).

## b. Class-based scope filter

Implemented in `planner.filter_features_for_setup()` ? `_feature_in_setup_scope()`:

- Reads `context.setups[0].scope` (`SetupScopeSpec` from descriptor YAML).
- `mode: full` ? keep all planner-eligible features (96260B **rear** setup).
- `mode: filtered` + `classes: [facing]` ? keep only envelope-stock **flat** faces (96260B **front** setup); uses `STOCK_BOUNDARY_SCOPE_TOKENS` and `envelope_stock_face_ids`.
- Other filtered scopes map tokens (`hole`, `pocket`, `wall`, …) to recognizer class groups via `_SCOPE_CLASS_GROUPS`.

Called from `_plan_one_setup()` after `filter_planner_features()` (recognizer-class gate).

## c. Step-3 generated setup descriptors

**Producer:** `setup_generation.py` — `generate_setup_entry_from_graph()` / `generate_part_setup_descriptor()`.

**On disk today:** No generated descriptor files under `pipeline_out/` yet. Generation is tested in `tests/test_setup_generation.py` (in-memory parity against `eval/gt/96260B_setup.yaml`).

**Shape:** Same schema as hand-authored YAML — `PartSetupDescriptor` with `part_id`, `defaults`, `setups.{front,rear}` entries (`setup_id`, `part_step`, `machining_side`, `opening_axis`, `pocket_access`, `scope`). Matches `setup_descriptor.parse_setup_descriptor()` / `load_setup_descriptor()`.

**Process inputs still explicit:** `machining_side` and `scope` are not inferred from geometry (fixturing decisions). Geometry (`opening_axis` from `approach_frame.z`) is taken from the cascade graph.

**Convention added by this change:** `run_cascade --export-dir` writes `setup_descriptor.yaml` per export; multi-setup planner merges per-panel exports or builds the family descriptor at load time.

## d. Reachability on `feature_graph_cascade.json` v4

**Producer:** `run_cascade.py` export path — `annotate_reachability()` after `build_cascade_feature_graph()` (step 2 approach vectors required). Sets `schema_version: 4` and top-level `reachability_summary`.

**Per-feature fields** (under `node.approach`, not top-level):

| Field | Level | Notes |
|-------|-------|-------|
| `approach.setup_dir` | per-feature | Step-2 candidate; reconciled by step 4a |
| `approach.reachable_3axis` | per-feature | bool after verification |
| `approach.reachability.verified` | per-feature | bool (false for walls / no geometry) |
| `approach.reachability.exempt` | per-feature | walls — lateral access |
| `approach.reachability.reachable_dirs` | per-feature | `["+Z"]`, `["+Z","-Z"]`, or `[]` |
| `approach.reachability.occluded` | per-feature | no reachable direction |
| `approach.reachability.per_direction` | per-feature | `+Z` / `-Z` occlusion detail |
| `reachability_summary` | graph | aggregate counts from `annotate_reachability()` |

**Existing `pipeline_out/*/feature_graph_cascade.json` files are schema v2** (no `approach` / reachability). Planner must run against freshly exported v4 graphs.

**Planner usage before this change:** **Unused.** Scope is class-based only. `group_operations_by_tool_strategy` ? `_split_members_by_reachability` is **tool-depth batching**, not step-4a feature?setup assignment.

## Material difference from assumptions

Assumptions hold with one clarification: **`run_pipeline.py` is not the planner pipeline**; swaps land in `machining_context.py`, `planner.py`, and cascade export (`run_cascade.py`). `run_pipeline.py` docstring updated to point at the CAM path.
