# Operation-bank audit — current planner emissions → canonical bank

> **STATUS 2026-07-08: Track A EXECUTED.** The planner now emits ONLY canonical
> bank ops (single flat `operation` field; `operation_type`/`strategy` retired,
> enum-validated in cam_plan_schema). Mapping decisions applied: **tapping**→`drill`
> (tool_type=tap carries the tap cycle) but **any other threading**→`thread_mill`
> (tool_type=thread_mill; tap-specific flag wins when both present); roughing
> geometry-driven via `3D_surface`
> (3D→`optirough`) + pocket `access` (closed→`pocket`, open→`dynamic_mill_2d`);
> wall/contour finish geometry-driven (`waterline` if 3D else `contour_2d`);
> freeform surface→`constant_scallop`; flat floor→`raster`; fillet→`pencil`;
> deep holes→`chip_break_drill`. Verified: test_planner 54 OK (mlcad env),
> test_cam_plan_schema/test_sequence_search OK, cascade regression byte-identical,
> golden regenerated with valid bank ops. Reverse-map to shop strategies covers all
> 6 shop strategies (0 missed). Two behavior notes: (1) open pockets now emit a
> distinct `dynamic_mill_2d` (intended per the geometry rule); (2) to avoid a
> wall+pocket-floor `contour_2d` over-merge with mismatched feeds, `_batch_key`
> keeps a wall/floor sub-role so batch granularity matches the pre-collapse plan.
> Remaining eval-suite failures (test_eval_cam_plan gt-count, aggregate "Profile"
> coverage; test_plan_sanity_report feed/coverage) are PRE-EXISTING / orthogonal
> (stale GT count, unrecognized profile features, handbook-feed tuning) — not caused
> by the rename.

---


Date: 2026-07-08. Canonical bank: [operation_bank.py](../operation_bank.py) (21 ops,
single flat vocabulary, from Operations.pdf). This audit maps every operation the
planner emits today onto that bank. **No rewire has been done** — this is the
mapping to approve first.

## Decision chosen
- **Bank shape:** single flat vocabulary (retire the `operation_type`/`strategy` split).
- **Scope:** define bank + audit, report mapping, no rewire yet.

## Current emissions → proposed bank op

The planner emits a `(operation_type, strategy)` pair per op. Collapsed to the flat bank:

| # | Feature branch | current `operation_type` | current `strategy` | → proposed bank op | Confidence |
|---|---|---|---|---|---|
| 1 | hole → drill | `drill` | `peck_drill` | **`drill`** | high |
| 2 | hole → bore | `bore` | `helical_bore` | **`helix_bore`** | high |
| 3 | hole → tapped | `tap` | `rigid_tap` | **`drill`** (tap cycle) | ⚠ see Q1 |
| 4 | pocket → rough | `pocket_mill` | `roughing` | **`pocket`** | ⚠ see Q2 |
| 5 | pocket → finish | `finish_contour` | `finishing_floor` | **`contour_2d`** | ⚠ see Q3 |
| 6 | wall | `wall_finish` | `finishing_wall` | **`waterline`** | ⚠ see Q3 |
| 7 | contour_surface | `surface_finish` | `finishing` (ball) | **`raster`** | ⚠ see Q4 |
| 8 | fillet | `fillet_finish` | `finishing_fillet` | **`pencil`** | high |
| 9 | profile → rough | `adaptive_rough` | `roughing` | **`optirough`** | ⚠ see Q2 |
| 10 | profile → finish | `wall_finish` | `finishing_wall` | **`contour_2d`** | ⚠ see Q3 |
| 11 | flat → facing | `facing` | `facing` | **`facing`** | high |
| 12 | flat → floor | `floor_finish` | `finishing_floor` | **`raster`** | ⚠ see Q4 |

High-confidence, unambiguous: rows 1, 2, 8, 11 (4 of 12).

## Open mapping questions (need your call before rewire)

**Q1 — tapped holes.** Current `tap`/`rigid_tap` uses a physical tap tool (a drilling
cycle), which fits the bank **`drill`** ("peck, tap, bore, ream cycles"). But the bank
also has **`thread_mill`** (mills threads with an endmill). These are different tools.
Should recognized tapped holes map to `drill` (tap cycle, matches current tool) or
`thread_mill` (single-point interpolated thread)? *Recommend `drill` — preserves the
current tool_type and the shop's rigid-tap intent.*

**Q2 — roughing: 2.5D vs 3D.** Current planner has one roughing shape (`roughing`
strategy) for both pockets and profiles. The bank distinguishes:
- `pocket` (closed 2D region), `dynamic_mill_2d` (fast adaptive 2.5D), `optirough`
  (3D adaptive/trochoidal), `area_roughing` (3D Z-level).
Pocket-rough is currently proposed → `pocket`, profile-rough → `optirough`, but the
"right" choice depends on whether these features are truly prismatic (2.5D) or 3D. Do
you want a **fixed** mapping (pocket→`pocket`, profile→`optirough`) or should it key
off geometry (e.g. depth/floor-flatness → 2.5D vs 3D)? *Recommend fixed for now;
geometry-driven selection is a step-5 concern.*

**Q3 — wall/contour finishing.** Three current finish branches (`finish_contour`,
`wall_finish`) collapse onto `contour_2d` vs `waterline`. `waterline` = constant-Z
(steep walls, 3D); `contour_2d` = 2.5D outline. Current names don't record whether the
wall is a straight prismatic wall (→`contour_2d`) or a stepped/steep 3D wall
(→`waterline`). Proposed: wall→`waterline`, pocket/profile finish→`contour_2d`. Is that
the split you want, or should all prismatic walls be `contour_2d`?

**Q4 — flat/surface finishing.** `surface_finish` (ball, freeform contour_surface) →
proposed `raster`, but `constant_scallop` and `radial_spiral` are also freeform-finish
candidates; `raster` suits shallow/flat-ish. `floor_finish` (flat faces) → also `raster`.
Confirm both freeform ball-finish and flat-floor finish should be `raster`, or split
(freeform→`constant_scallop`, flat→`raster`).

## Bank ops the planner NEVER emits (coverage gaps)

12 of 21 canonical ops have no producer today. These are legitimately unreachable
until new recognizers / logic exist — flagged so the gap is explicit, not silent:

- Roughing: `area_roughing`, `rest_roughing`
- Finishing: `radial_spiral`, `constant_scallop`, `steep_shallow`, `rest_finish`
- 2.5D: `dynamic_mill_2d`
- Holes: `chip_break_drill`, `thread_mill`
- Auxiliary: `engraving`, `chamfer`, `deburr`

`rest_roughing`/`rest_finish` depend on multi-tool sequencing (step 5). `chamfer`/
`deburr`/`engraving` need edge/text recognizers not in the cascade. `chip_break_drill`
is a depth-triggered variant of `drill`. None block the rename; they define the
producer work that follows.

## Downstream touchpoints a rewire must update (for scoping only)

- [planner.py](../planner.py) `map_feature_to_operations` (branch literals),
  `ROUGH_OPERATION_TYPES`/`FINISH_OPERATION_TYPES`, `FINISH_STRATEGY_ORDER`,
  `OPEN_MILLING_OPERATION_TYPES`, `_operation_phase`, tool-preset ranking
  (`_tool_has_strategy_preset`, `_strategy_match_rank`, `_iso_category_rank`).
- [cam_plan_schema.py](../cam_plan_schema.py) `operation_type`/`strategy` fields
  (currently free-form str; would become one enum-validated field).
- Tests: `tests/test_planner.py` and any golden CamPlan fixtures asserting op names.
- Golden/eval fixtures under `eval/` and `examples/*cam_plan*.json`.
