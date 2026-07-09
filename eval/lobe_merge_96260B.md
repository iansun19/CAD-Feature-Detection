# Lobe contour merge — 96260B validation

Generated: 2026-07-08

## Top-line VERDICT: **PASS**

Plan-neutral (identical machined geometry), zero KEPT_FRONT orphans via node elimination, and cross-part regression clean (only 96260B rear lobe-cap `contour_surface` counts change).

---

## Merge pass summary

- Module: `lobe_contour_merge.py`
- Flag: `--merge-lobe-contours` / `--no-merge-lobe-contours` (default **on**)
- Pipeline order: reachability annotation ? **merge** ? export (anchors preserved; +Z-only cap debris merged per lobe)
- Rear graph: **105 ? 62** nodes (**43** removed); only `contour_surface` class affected (**72 ? 29**)

### Merge rule (no area threshold)

Per lobe (angular mouth-step sector + exterior axial band `Y > mouth_axial`):

1. **Candidates**: single-lobe `contour_surface` nodes in the cap band.
2. **Partition**: machining anchors (`-Z` reachable on rear) stay **unchanged**; `+Z`-only fragments are debris.
3. **Debris merge**: per lobe, union-find not needed — all debris candidates collapse to **one sink node** when ?2 debris fragments exist.
4. **Provenance**: `params.merged_from_fragment_ids`, `params.lobe_contour_merge_kind` (`debris`), `params.lobe_id`.
5. **Reachability**: debris sinks get `reachable_dirs: []` (not planner-selected, not KEPT_FRONT orphans).

---

## Cluster ? merged-feature map (rear, 7 lobes)

| Lobe | Debris node | Fragment ids absorbed | Anchors preserved (unchanged) | Debris faces |
|------|-------------|----------------------|-------------------------------|--------------|
| L0 NW | **39** | 39,40,66,67,68,69,71,72 | 38,41 | 8 |
| L1 W | **43** | 43,44,45,75,76,77,78 | 42 | 7 |
| L2 NE | **48** | 48,49,73,74,80,81,82,83 | 46,47 | 8 |
| L3 E | **51** | 51,53,79,85,86,87,88 | 50,52 | 7 |
| L4 SE | **34** | 34,55,84,91,92,93 | 54,56,57 | 7 |
| L5 N | **35** | 35,89,90,94,95,96,97 | 58,59,60,61 | 8 |
| L6 SW | **63** | 63,65,70,98,99,100,101 | 62,64 | 7 |

All **38** prior KEPT_FRONT orphan ids (44,45,48,…,101) are **eliminated** — absorbed into the seven debris nodes above, not re-labeled.

---

## Plan-neutrality diff

Baseline: pre-merge graph ? `cam_plan_baseline_rerun.json` (16 ops).  
Post-merge: `pipeline_out/96260B_{rear,front}/feature_graph_cascade.json` ? `examples/cam_plan_96260B.json`.

| Check | Result |
|-------|--------|
| Op count | **16 = 16** |
| Op type + setup sequence | **identical** |
| Per-op `feature_refs` | **identical** |
| Machined face set (geometry) | **257 = 257** faces |

No op-type, sequence, or machined-material change. Feature ids in the plan are unchanged because machining anchors were not merged.

---

## Coverage audit (honest)

### `scripts/kept_front_resolution_96260B.py`

| Metric | Pre-merge | Post-merge |
|--------|-----------|------------|
| Enumerated KEPT_FRONT targets | 38 | **0** |
| ORPHAN | 38 | **0** |
| **VERDICT** | FAIL | **PASS** |

Zero orphans from **node elimination** (43 fragment nodes removed; 7 debris sinks unreachable), not bucket relabeling.

### `scripts/coverage_audit_96260B.md`

- **VERDICT: PASS**
- Six debris sink nodes bucketed `MERGED_CAP_DEBRIS` (accounted, not UNACCOUNTED)
- UNACCOUNTED count: **0**

---

## Cross-part node-count regression

| Part / export | Nodes before | Nodes after | Removed | Class deltas |
|---------------|-------------|-------------|---------|--------------|
| **96260B rear** | 105 | 62 | 43 | `contour_surface` 72?29 only |
| **96260B front** | 44 | 44 | 0 | none |
| fish mold | — | — | — | skipped (non-lobed pocket setup) |

**Regression guard: PASS** — only 96260B rear lobe-cap `contour_surface` fragmentation changes; front panel and non-lobed parts untouched.

---

## Files touched

- `lobe_contour_merge.py` — merge pass
- `run_cascade.py` — `--merge-lobe-contours` flag, pipeline wiring
- `scripts/coverage_audit_96260B.py` — `MERGED_CAP_DEBRIS` bucket
- `tests/test_lobe_contour_merge.py` — unit tests
- `examples/cam_plan_96260B.json` — regenerated (byte-identical machining plan)
- `pipeline_out/96260B_rear/feature_graph_cascade.json` — merged export
