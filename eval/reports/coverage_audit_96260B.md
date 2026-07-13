# Coverage audit -- 96260B (post opening-axis fix)

Generated: 2026-07-08 21:03 UTC

## VERDICT: **PASS**

## Notes

- Rear old +Z scope is 49 on current graph (triage cited 98); drop set size 19.
- In-memory wrong-axis replay produced 19 ops (brief cited 20).
- Disappeared-op detector found 3 ops (brief cited 4): ['OP020', 'OP030', 'OP060']

## Scope counts

| Setup | Old wrong-axis rear filter (+Z) | Planner kept (current) | Delta |
|-------|--------------------------------|------------------------|-------|
| rear  | 49 | 36 | -13 |
| front | n/a | 39 | n/a |

Logical resolution tests each panel's node against rear `-Z` and front `+Z` reachability tokens on that panel's graph (per-panel, not cross-panel id join).

## Feature ledger summary

- **KEPT_REAR**: 6
- **KEPT_FRONT**: 53
- **KEPT_BOTH**: 35
- **FLAGGED_ORPHAN**: 1
- **FLAGGED_LATERAL**: 4
- **MERGED_CAP_DEBRIS**: 7
- **UNACCOUNTED**: 0

## Full feature ledger

| panel | feature_id | feature_type | reachable_dirs | resolved_setups | planner_scope | bucket |
|-------|------------|--------------|----------------|-----------------|---------------|--------|
| front | 0 | filleted_open_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 1 | filleted_open_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 2 | filleted_open_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 3 | filleted_open_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 4 | filleted_open_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 5 | filleted_open_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 6 | filleted_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 7 | filleted_open_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 8 | filleted_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 9 | filleted_open_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 10 | filleted_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 11 | filleted_open_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 12 | filleted_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 13 | filleted_open_pocket | ["+Z", "-Z"] | rear,front | front | KEPT_BOTH |
| front | 14 | filleted_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 15 | filleted_open_pocket | ["+Z", "-Z"] | rear,front | front | KEPT_BOTH |
| front | 16 | filleted_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 17 | filleted_open_pocket | ["+Z", "-Z"] | rear,front | front | KEPT_BOTH |
| front | 18 | filleted_pocket | ["+Z"] | front | front | KEPT_FRONT |
| front | 19 | filleted_blind_hole | ["+Z"] | front | front | KEPT_FRONT |
| front | 20 | through_hole | ["+Z", "-Z"] | rear,front | front | KEPT_BOTH |
| front | 21 | flat | ["+Z"] | front | front | KEPT_FRONT |
| front | 22 | outer_fillet | ["+Z"] | front | front | KEPT_FRONT |
| front | 23 | wall | ["+Z"] | rear,front | front | KEPT_BOTH |
| front | 24 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 25 | contour_surface | ["-Z"] | rear | - | KEPT_REAR |
| front | 26 | contour_surface | ["-Z"] | rear | - | KEPT_REAR |
| front | 27 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 28 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 29 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 30 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 31 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 32 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 33 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 34 | contour_surface | ["-Z"] | rear | - | FLAGGED_ORPHAN |
| front | 35 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 36 | contour_surface | ["-Z"] | rear | - | FLAGGED_LATERAL |
| front | 37 | contour_surface | ["-Z"] | rear | - | FLAGGED_LATERAL |
| front | 38 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 39 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 40 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 41 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 42 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| front | 43 | contour_surface | ["+Z"] | front | front | KEPT_FRONT |
| rear | 0 | filleted_open_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 1 | filleted_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 2 | filleted_open_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 3 | filleted_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 4 | filleted_open_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 5 | filleted_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 6 | filleted_open_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 7 | filleted_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 8 | filleted_open_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 9 | filleted_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 10 | filleted_open_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 11 | filleted_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 12 | filleted_open_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 13 | filleted_pocket | ["+Z"] | front | - | KEPT_FRONT |
| rear | 14 | filleted_blind_hole | ["+Z"] | front | - | KEPT_FRONT |
| rear | 15 | through_hole | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 16 | flat | ["+Z"] | front | - | KEPT_FRONT |
| rear | 17 | flat | ["-Z"] | rear | rear | KEPT_REAR |
| rear | 18 | outer_fillet | ["+Z"] | front | - | KEPT_FRONT |
| rear | 19 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 20 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 21 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 22 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 23 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 24 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 25 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 26 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 27 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 28 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 29 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 30 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 31 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 32 | wall | ["+Z"] | rear,front | rear | KEPT_BOTH |
| rear | 33 | contour_surface | ["+Z"] | front | - | KEPT_FRONT |
| rear | 34 | contour_surface | [] | - | - | MERGED_CAP_DEBRIS |
| rear | 35 | contour_surface | [] | - | - | MERGED_CAP_DEBRIS |
| rear | 36 | contour_surface | ["-Z"] | rear | rear | KEPT_REAR |
| rear | 37 | contour_surface | ["-Z"] | rear | rear | FLAGGED_LATERAL |
| rear | 38 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 39 | contour_surface | [] | - | - | MERGED_CAP_DEBRIS |
| rear | 41 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 42 | contour_surface | ["+Z"] | front | - | KEPT_FRONT |
| rear | 43 | contour_surface | [] | - | - | MERGED_CAP_DEBRIS |
| rear | 46 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 47 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 48 | contour_surface | [] | - | - | MERGED_CAP_DEBRIS |
| rear | 50 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 51 | contour_surface | [] | - | - | MERGED_CAP_DEBRIS |
| rear | 52 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 54 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 56 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 57 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 58 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 59 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 60 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 61 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 62 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 63 | contour_surface | [] | - | - | MERGED_CAP_DEBRIS |
| rear | 64 | contour_surface | ["+Z", "-Z"] | rear,front | rear | KEPT_BOTH |
| rear | 102 | contour_surface | ["-Z"] | rear | rear | KEPT_REAR |
| rear | 103 | contour_surface | ["-Z"] | rear | rear | FLAGGED_LATERAL |
| rear | 104 | contour_surface | ["-Z"] | rear | rear | KEPT_REAR |

## Rear scope drop reconciliation (19 features)

Rear graph features in OLD wrong-axis `+Z` scope (49) minus NEW rear planner scope (36):

| feature_id | feature_type | reachable_dirs | bucket |
|------------|--------------|----------------|--------|
| 0 | filleted_open_pocket | ["+Z"] | KEPT_FRONT |
| 1 | filleted_pocket | ["+Z"] | KEPT_FRONT |
| 2 | filleted_open_pocket | ["+Z"] | KEPT_FRONT |
| 3 | filleted_pocket | ["+Z"] | KEPT_FRONT |
| 4 | filleted_open_pocket | ["+Z"] | KEPT_FRONT |
| 5 | filleted_pocket | ["+Z"] | KEPT_FRONT |
| 6 | filleted_open_pocket | ["+Z"] | KEPT_FRONT |
| 7 | filleted_pocket | ["+Z"] | KEPT_FRONT |
| 8 | filleted_open_pocket | ["+Z"] | KEPT_FRONT |
| 9 | filleted_pocket | ["+Z"] | KEPT_FRONT |
| 10 | filleted_open_pocket | ["+Z"] | KEPT_FRONT |
| 11 | filleted_pocket | ["+Z"] | KEPT_FRONT |
| 12 | filleted_open_pocket | ["+Z"] | KEPT_FRONT |
| 13 | filleted_pocket | ["+Z"] | KEPT_FRONT |
| 14 | filleted_blind_hole | ["+Z"] | KEPT_FRONT |
| 16 | flat | ["+Z"] | KEPT_FRONT |
| 18 | outer_fillet | ["+Z"] | KEPT_FRONT |
| 33 | contour_surface | ["+Z"] | KEPT_FRONT |
| 42 | contour_surface | ["+Z"] | KEPT_FRONT |

### Tally

- **KEPT_FRONT**: 19
- **KEPT_BOTH**: 0
- **FLAGGED_ORPHAN**: 0
- **FLAGGED_LATERAL**: 0
- **UNACCOUNTED**: 0
- **KEPT_REAR** (still rear-only): 0

### UNACCOUNTED (must be empty)

*(empty)*

## Op-count reconciliation

- Old wrong-axis plan (in-memory replay): **19** ops
- New fixed-axis plan on disk: **18** ops
- Net delta: **1** ops
- Disappeared rear/front op slots: **3**

### The 4 disappeared operations

| old_op | setup | op_type | features | classification | evidence |
|--------|-------|---------|----------|----------------|----------|
| OP020 | rear | pocket_mill | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 | **SPURIOUS** | wrong-axis rear (+Z) scope only; features ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'] |
| OP030 | rear | finish_contour | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 | **SPURIOUS** | wrong-axis rear (+Z) scope only; features ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'] |
| OP060 | rear | surface_finish | 33 | **SPURIOUS** | wrong-axis rear (+Z) scope only; features ['33'] |

### Regrouped (not counted in the 4)

- Old OP070 regrouped into new rear OP040; 9 +Z-only rear-only contour ids not in new plan: 41, 47, 52, 56, 57, 59, 60, 61 ... (+1 more). These are logically KEPT_FRONT on the rear graph but absent from the front-panel export (no planner path).

## Prior-findings confirmation

| Check | Expected | Observed |
|-------|----------|----------|
| front 25 (-Z-only) | lands on rear, not orphaned | bucket=KEPT_REAR, resolved rear=True |
| front 26 (-Z-only) | lands on rear, not orphaned | bucket=KEPT_REAR, resolved rear=True |
| front 36 (-Z-only) | lands on rear, not orphaned | bucket=FLAGGED_LATERAL, resolved rear=True |
| rear 37 | FLAGGED_LATERAL | bucket=FLAGGED_LATERAL, planner_rear=True |
| cross-panel id 34 | per-panel, not id-joined | rear: bucket=MERGED_CAP_DEBRIS dirs=[]; front: bucket=FLAGGED_ORPHAN dirs=['-Z'] |
