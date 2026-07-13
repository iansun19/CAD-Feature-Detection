# KEPT_FRONT resolution -- 96260B

Generated: 2026-07-08 21:03 UTC

## VERDICT: **PASS**

## Enumeration

Target set: rear-graph features that are **rear-only ids**, **+Z-only** reachable (`+Z` yes, `-Z` no), and absent from the front graph node set.

- **Count: 0**
- **Ids:** 

## Matching method

Twin detection uses STEP face centroids/normals (world frame), per feature:
- same `class_name`
- same `surface_type_histogram`
- `total_area` within 8%
- centroid X/Z within 3.0 mm (split-panel Y offset ignored)
- normal dot >= 0.85 (Y-flip allowed)
- greedy 1:1 assignment on match score (no shared front twin)
- **COVERED** only if the matched front id appears in a `setup_id: front` operation in `examples/cam_plan_96260B.json`

## Tally

- **COVERED_VIA_FRONT_TWIN:** 0
- **ORPHAN:** 0

## Per-feature resolution

| rear_id | bucket | front_twin | op_id | evidence |
|---------|--------|------------|-------|----------|

## Sanity cross-checks

- 1:1 twin assignment: 0 pairs; duplicate front twins: none
- CamPlan OP130 (front surface_finish) refs: ['28', '32']
- All COVERED twins are present in OP130 feature_refs (verified in JSON).
- ORPHAN ids (0): 
