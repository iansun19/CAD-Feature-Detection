# Orphan characterization — 96260B KEPT_FRONT orphans

Generated: 2026-07-08 (read-only geometry audit)

## Executive summary

| Classification | Count |
|---|---|
| **DEBRIS_OF_COVERED** | **38** |
| **DEBRIS_UNCOVERED** | **0** |
| **GENUINELY_DISTINCT** | **0** |

### VERDICT

**Root cause: CASCADE OVER-SEGMENTATION.** All 38 orphans are geometric fragments of lobe-tier contour surfaces that are **already machined** by existing rear ops (OP040/OP050) and/or front OP130 surface_finish via the seven successful front twins. None are standalone uncovered features. Recommended fix is **upstream cascade merge** in perception; planner and twin gates should remain untouched.

**Merged-surface count:** 38 fragments collapse into **7 lobe contour surfaces** (one per main-torus lobe cap), **all already covered** — 0 newly uncovered surfaces after merge.

---

## Method

Data sources (read-only):

- Rear graph: `pipeline_out/96260B_rear/feature_graph_cascade.json`
- Front graph: `pipeline_out/96260B_front/feature_graph_cascade.json`
- Face centroids/normals: embedded in `pipeline_out/96260B_rear/viewer.html` (triangulated STEP)
- Plan: `examples/cam_plan_96260B.json`
- Prior enumeration: `eval/kept_front_resolution_96260B.md`

For each of the 38 orphan ids:

1. **Spatial containment** — centroid/normal from viewer face geometry; check proximity (<20 mm XZ, <8 mm Y) to planned rear contour features (OP040/OP050) or covered main-torus twins (66/70/73/79/84/89/94 ? OP130).
2. **Fragment clustering** — graph `adjacent` edges + lobe-tier topology (wall 19–32 ? bspline strip ? analytic patches). Each lobe cap splits into ~6–8 features: 2 bspline strips (73.6 + 20.8 mm²) + main torus (241 mm², sometimes covered via front twin) + sliver torus (4.7 mm²) + sphere (6.0 mm²) + cone (0.005 mm²).
3. **Survivors** — anything not merging into a planned-anchored lobe group and not forming a coherent uncovered cluster ? GENUINELY_DISTINCT.

Gate probe (no retuning): re-ran area (8%), XZ (3 mm), normal-dot (0.85) checks from `scripts/kept_front_resolution_96260B.py` against front contour_surface candidates.

---

## Segmentation pattern (why these exist)

The cascade over-segments each lobe-tier **contour ring** by analytic surface type:

```
wall (19–32) ?? bspline strip (large 73.6 mm²) ?? bspline strip (small 20.8 mm²)
                      ?                                    ?
                      ??? main torus (241 mm²)  ? 7 covered via OP130 front twin
                      ??? sliver torus (4.7 mm²)
                      ??? sphere patch (6.0 mm²)
                      ??? cone sliver (0.005 mm²)
```

The planner picks **one** bspline per lobe (OP040 rear) and/or the **main torus** (OP130 front twin). The unselected bspline twin and all analytic slivers become graph nodes that pass +Z reachability but have no direct op ref — the "orphans."

---

## Cluster ? merged-surface map

Seven lobe groups absorb all 38 fragments. Every group has at least one planned anchor.

| Group | Planned anchors | Orphan fragments | Merged area (mm²) | Types |
|---|---|---|---|---|
| **L1 NW** | rear **38** (OP040), twin **66?front 40** (OP130) | 44, 45, 67, 68, 69, 77, 78 | 97.4 | 2 bspline, 1 torus, 2 sphere, 2 cone |
| **L2 W** | rear **41** (OP040), twin **70?front 42** (OP130) | 63, 71, 72, 98, 99 | 37.5 | 1 bspline, 1 sphere, 2 cone |
| **L3 NE** | twin **73?front 43** (OP130) | 48, 49, 74, 75, 76, 82, 83 | 111.1 | 2 bspline, 1 torus, 2 sphere, 2 cone |
| **L4 E** | rear **52** (OP040), twin **79?front 35** (OP130) | 53, 80, 81, 87, 88 | 37.5 | 1 bspline, 1 sphere, 2 cone |
| **L5 SE** | rear **50** (OP040), twin **84?front 38** (OP130) | 51, 85, 86, 93 | 32.8 | 1 bspline, 1 sphere, 2 cone |
| **L6 N** | rear **54** (OP040), twin **89?front 39** (OP130) | 55, 90, 91, 92, 96, 97 | 37.5 | 1 bspline, 1 torus, 2 sphere, 2 cone |
| **L7 SW** | rear **64** (OP040), twin **94?front 41** (OP130) | 65, 95, 100, 101 | 32.8 | 1 bspline, 1 sphere, 2 cone |

**Totals:** 38 fragments ? **7 merged lobe contours**, **7/7 already covered** (mix of OP040 rear bspline + OP130 front main-torus), **0 newly uncovered**.

---

## Gate near-miss check

### Seven successful twins — all pass with clear margin

| rear | front twin | area_rel | XZ (mm) | norm dot | Pass |
|---|---|---|---|---|---|
| 66 | 40 | 0.062 (limit 0.08) | 0.39 (limit 3.0) | 0.998 | ? |
| 70 | 42 | 0.062 | 0.39 | 0.998 | ? |
| 73 | 43 | 0.062 | 0.39 | 0.998 | ? |
| 79 | 35 | 0.062 | 0.39 | 0.998 | ? |
| 84 | 38 | 0.062 | 0.28 | 0.998 | ? |
| 89 | 39 | 0.062 | 0.39 | 0.998 | ? |
| 94 | 41 | 0.062 | 0.39 | 0.998 | ? |

Headroom: ~18% on area, ~2.6 mm on XZ, dot ? 0.998 (limit 0.85). Successful matches are whole main-torus features (241 mm²), not slivers.

### Orphan near-misses — none legitimate

Every orphan's nearest same-surface-type front candidate fails **?2 gates simultaneously**:

- **Area:** rel ? 0.98–1.00 (fragment 4.7/6.0/0.005 mm² vs whole 241 mm² torus, or bspline 20.8/73.6 vs different lobe) — **correct rejection**.
- **XZ:** 13–140 mm (different lobe slot or fragment offset) — **correct rejection**.
- **Surface type:** bspline orphans nearest front candidates are mixed cone/torus composites — **correct rejection**.

**No orphan is a near-miss for a legitimate whole-feature twin.** The area and XZ gates are doing the right job: blocking fragment?whole matches. Retuning would create false COVERED_VIA_FRONT_TWIN assignments without adding real coverage.

Representative orphan gate failures (from `kept_front_resolution_96260B.md`):

| orphan | type | area | nearest front | failure |
|---|---|---|---|---|
| 67 | torus 4.7 mm² | rel=0.988 | 32 | surface_type + area |
| 68 | sphere 6.0 mm² | rel=0.985 | 32 | surface_type + area |
| 44 | bspline 73.6 mm² | rel=0.816 | 24 | surface_type + area + xz=65 mm |
| 99 | cone 0.005 mm² | rel=1.000 | 24 | surface_type + area + xz=13 mm |

---

## Per-orphan classification

Evidence columns: **faces** (STEP face index), **surf** (surface_type_histogram), **centroid** (world mm), **anchor** (planned feature covering merged surface), **link** (how assigned).

| id | class | surf | area | centroid (X,Y,Z) | anchor | link |
|---|---|---|---|---|---|---|
| 44 | DEBRIS_OF_COVERED | bspline | 73.6 | (-30.0,-62.8,-69.8) | 38 + 66 | duplicate bspline split; 6.3 mm from planned 38; wall-22 slot of L1 |
| 45 | DEBRIS_OF_COVERED | bspline | 20.8 | (-20.5,-61.9,-71.6) | 38 + 66 | small-bspline twin of 44; same L1 contour |
| 48 | DEBRIS_OF_COVERED | bspline | 73.6 | (35.8,-62.8,-67.0) | 73 | large strip; 16.5 mm from covered torus 73 (L3) |
| 49 | DEBRIS_OF_COVERED | bspline | 20.8 | (43.2,-61.9,-60.7) | 73 | small strip paired with 48; L3 |
| 51 | DEBRIS_OF_COVERED | bspline | 20.8 | (65.3,-61.9,35.9) | 50 + 84 | 10.0 mm from planned 50; L5 |
| 53 | DEBRIS_OF_COVERED | bspline | 20.8 | (74.4,-61.9,-4.0) | 52 + 79 | 9.7 mm from planned 52; L4 |
| 55 | DEBRIS_OF_COVERED | bspline | 20.8 | (12.6,-61.9,73.4) | 54 + 89 | 10.0 mm from planned 54; L6 |
| 63 | DEBRIS_OF_COVERED | bspline | 20.8 | (-74.4,-61.9,-4.0) | 62 + 70 | 10.0 mm from planned 62; L2 |
| 65 | DEBRIS_OF_COVERED | bspline | 20.8 | (-65.3,-61.9,35.9) | 64 + 94 | 9.7 mm from planned 64; L7 |
| 67 | DEBRIS_OF_COVERED | torus | 4.7 | (-29.5,-68.5,-61.3) | 66 | sliver torus; adjacent 68/77; L1 cap |
| 68 | DEBRIS_OF_COVERED | sphere | 6.0 | (-31.7,-68.1,-60.2) | 66 | graph-adjacent planned 38 + spatial cluster torus 66; L1 |
| 69 | DEBRIS_OF_COVERED | cone | 0.005 | (-41.4,-66.6,-55.6) | 66 | cone sliver on L1 cap ring |
| 71 | DEBRIS_OF_COVERED | sphere | 6.0 | (-65.8,-68.1,-17.5) | 70 | sliver on covered torus 70; L2 |
| 72 | DEBRIS_OF_COVERED | cone | 0.005 | (-63.4,-66.6,-28.0) | 70 | cone sliver; graph-adjacent 41; L2 |
| 74 | DEBRIS_OF_COVERED | torus | 4.7 | (29.5,-68.5,-61.3) | 73 | sliver torus; adjacent 75/82; L3 |
| 75 | DEBRIS_OF_COVERED | sphere | 6.0 | (27.4,-68.1,-62.3) | 73 | sliver; adjacent 74; L3 |
| 76 | DEBRIS_OF_COVERED | cone | 0.005 | (17.6,-66.6,-67.1) | 73 | cone sliver; adjacent 43 (unplanned bspline sibling); L3 |
| 77 | DEBRIS_OF_COVERED | sphere | 6.0 | (-27.4,-68.1,-62.3) | 66 | adjacent orphan bspline 44; L1 |
| 78 | DEBRIS_OF_COVERED | cone | 0.005 | (-17.6,-66.6,-67.1) | 66 | adjacent orphan bspline 45; L1 |
| 80 | DEBRIS_OF_COVERED | sphere | 6.0 | (65.8,-68.1,-17.5) | 79 | sliver on covered torus 79; L4 |
| 81 | DEBRIS_OF_COVERED | cone | 0.005 | (63.4,-66.6,-28.0) | 79 | graph-adjacent planned 47; L4 |
| 82 | DEBRIS_OF_COVERED | sphere | 6.0 | (31.7,-68.1,-60.2) | 73 | adjacent orphan bspline 48; L3 |
| 83 | DEBRIS_OF_COVERED | cone | 0.005 | (41.4,-66.6,-55.6) | 73 | adjacent orphan bspline 49; L3 |
| 85 | DEBRIS_OF_COVERED | sphere | 6.0 | (54.7,-68.1,40.5) | 84 | graph-adjacent planned 50; L5 |
| 86 | DEBRIS_OF_COVERED | cone | 0.005 | (61.5,-66.6,32.1) | 84 | adjacent orphan bspline 51; L5 |
| 87 | DEBRIS_OF_COVERED | sphere | 6.0 | (66.8,-68.1,-12.8) | 79 | graph-adjacent planned 52; L4 |
| 88 | DEBRIS_OF_COVERED | cone | 0.005 | (69.3,-66.6,-2.3) | 79 | adjacent orphan bspline 53; L4 |
| 90 | DEBRIS_OF_COVERED | torus | 4.7 | (0.0,-68.5,68.0) | 89 | sliver; adjacent 91/96; L6 |
| 91 | DEBRIS_OF_COVERED | sphere | 6.0 | (2.4,-68.1,68.0) | 89 | graph-adjacent planned 54; L6 |
| 92 | DEBRIS_OF_COVERED | cone | 0.005 | (13.2,-66.6,68.1) | 89 | adjacent orphan bspline 55; L6 |
| 93 | DEBRIS_OF_COVERED | cone | 0.005 | (45.0,-66.6,52.8) | 84 | graph-adjacent planned 57; L5 |
| 95 | DEBRIS_OF_COVERED | cone | 0.005 | (-45.0,-66.6,52.8) | 94 | graph-adjacent planned 59; L7 |
| 96 | DEBRIS_OF_COVERED | sphere | 6.0 | (-2.4,-68.1,68.0) | 89 | graph-adjacent planned 60; L6 |
| 97 | DEBRIS_OF_COVERED | cone | 0.005 | (-13.2,-66.6,68.1) | 89 | graph-adjacent planned 61; L6 |
| 98 | DEBRIS_OF_COVERED | sphere | 6.0 | (-66.8,-68.1,-12.8) | 70 | graph-adjacent planned 62; L2 |
| 99 | DEBRIS_OF_COVERED | cone | 0.005 | (-69.3,-66.6,-2.3) | 70 | adjacent orphan bspline 63; L2 |
| 100 | DEBRIS_OF_COVERED | sphere | 6.0 | (-54.7,-68.1,40.5) | 94 | graph-adjacent planned 64; L7 |
| 101 | DEBRIS_OF_COVERED | cone | 0.005 | (-61.5,-66.6,32.1) | 94 | adjacent orphan bspline 65; L7 |

---

## Why not GENUINELY_DISTINCT?

Strict checks applied:

- **76** (cone 0.005 mm²): only neighbor is unplanned bspline 43; still 16.5 mm from covered torus 73 centroid, same L3 cap ring as planned twin 43. Absorbing into L3 contour — not standalone.
- **69** (cone 0.005 mm²): spatially on L1 cap despite parent internal-id collision with perimeter feature 36; cluster with torus 66 slivers at Y??66..?68 band, not the Y??87 perimeter chamfer.
- No orphan occupies a region isolated from all planned anchors by >25 mm XZ with normal discontinuity.

---

## Fork decision

| Fork | Result |
|---|---|
| Overwhelmingly DEBRIS_* ? cascade over-segmentation, upstream merge | **YES — 38/38 DEBRIS_OF_COVERED** |
| Meaningful GENUINELY_DISTINCT ? group with 37 for lateral setup work | **NO — 0/38** |

**One-line verdict:** 38 fragments ? **7 lobe contour surfaces**, **7/7 already covered** by OP040 rear bspline and/or OP130 front main-torus twins; **0 newly uncovered** after merge. Root cause is **cascade over-segmentation by analytic surface type**; fix upstream, planner untouched.

---

## Sanity cross-checks

- Orphan count matches `kept_front_resolution_96260B.md`: 38.
- Seven covered twins unchanged: 66?40, 70?42, 73?43, 79?35, 84?38, 89?39, 94?41.
- All orphan bspline strips (44–65) pair spatially (?10 mm XZ) with either a planned bspline twin or a covered main torus on the same lobe — consistent with double bspline split, not independent features.
- No code, graph, plan, or gate changes made.
