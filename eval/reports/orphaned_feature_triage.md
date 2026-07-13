# Orphaned feature triage � 96260B reachability scope diff

Date: 2026-07-08. Read-only diagnosis of features dropped by verified-reachability
scoping on the 96260B rear + front panel exports. No code or graph changes.

## Setup axes (correction)

The task brief mentions �rear +Z / front ?Z�. On disk today both panels use the
**same world opening axis** and the planner applies the **same approach label** to
both setups:

| Setup | STEP panel | `machining_side` | `opening_axis` (world) | `approach_frame.z` | Planner `setup_approach_dir` |
|-------|------------|------------------|------------------------|--------------------|------------------------------|
| rear  | `fixtures/step/96260B_rear.stp`  | `back`  | `[0, 1, 0]` (+Y) | `[0, 1, 0]` | **`+Z` always** (`planner._primary_setup_approach_dir`) |
| front | `fixtures/step/96260B_front.stp` | `front` | `[0, 1, 0]` (+Y) | `[0, 1, 0]` | **`+Z` always** |

Reachability labels `+Z` / `?Z` are **along �opening_axis** (here �Y world), not
world �Z. Rear `part_axis_top` = ?57.84 mm, `part_axis_bottom` = ?91.66 mm;
front `part_axis_top` = ?25.76 mm, `part_axis_bottom` = ?54.83 mm.

**Root wiring gap:** `_primary_setup_approach_dir()` returns `+Z` for every setup.
Back-side features verified reachable only from `?Z` (?Y) are therefore dropped from
rear *and* from front when both filters require `+Z`. That explains shared orphans.

**Source graphs:** `pipeline_out/96260B_rear/feature_graph_cascade.json` and
`pipeline_out/96260B_front/feature_graph_cascade.json` (both `schema_version: 4`).

**Replicated scope diff** (mirrors `planner.print_scope_assignment_diff`):

| Setup | Class filter kept | Reachability kept (`+Z` only) | Class-only drops |
|-------|-------------------|-------------------------------|------------------|
| rear  | 104 | **98** | `[17, 36, 37, 102, 103, 104]` |
| front | 43 (full scope; reachability bypasses facing class scope on v4) | **38** | `[25, 26, 34, 36, 37]` |

Cross-panel IDs present in **both** graphs and unreachable from `+Z` on **both**
panels: **`[36, 37]`** � the true orphans under current wiring.

---

## A. Orphans 36 and 37

Feature IDs are **per-panel** (rear and front graphs partition different face
index spaces). Below, �rear 36� means `feature_id: 36` on the rear export; front
has different geometry at the same numeric id on some features.

### Rear feature 36

| Field | Value |
|-------|-------|
| `class_name` | `contour_surface` |
| Faces | `324` (cone), `325` (torus) |
| Area / bbox | 3121 mm�; Y span ? 2.76 mm, full-plate X/Z extent |
| Face normals (viewer) | 324: `[~0, ?0.50, ?0.87]`; 325: `[0, ?0.87, ?0.50]` |
| Centroids (viewer) | 324: Y ? ?88.34; 325: Y ? ?85.54 (bottom band of rear panel) |

**Step-2 approach**

```json
{
  "axis": [0.0, 0.692857, 0.721075],
  "source": "aggregate_normal",
  "setup_dir": "-Z",
  "reachable_3axis": true
}
```

**`node.approach.reachability` (verbatim)**

```json
{
  "verified": true,
  "reachable_dirs": ["-Z"],
  "occluded": false,
  "required_depth_mm": 5.0419,
  "effective_tool_radius_mm": 4.7625,
  "per_direction": {
    "+Z": {
      "occluded": true,
      "exposed_fraction": 0.0,
      "n_targets": 12,
      "required_depth_mm": 5.0419,
      "effective_tool_radius_mm": 4.7625
    },
    "-Z": {
      "occluded": false,
      "exposed_fraction": 0.167,
      "n_targets": 12,
      "required_depth_mm": 5.0419,
      "effective_tool_radius_mm": 4.7625
    }
  },
  "corrected_from": null
}
```

**Classification: planner wiring / ?Z back-side feature � not a third axis, not a cascade artifact**

**Evidence**

- Geometry is the large perimeter cone+torus chamfer at the **back (low-Y) lip** of
  the plate. Face normals tilt toward ?Y/?Z; `+Z` (+Y) rays are fully occluded
  (0% exposed), `?Z` (?Y) clears at 16.7% � consistent with machining from the
  back setup.
- Step-2 aggregate axis is oblique to opening_axis (dot ? 0.69 &lt; cos 15�), so
  candidacy was weak; step-4a verification legitimately found **only `?Z`**.
- The same physical band appears on the front export as **front 25** (faces 275�276,
  same areas/normals, Y shifted ~+37 mm), also `reachable_dirs: ["-Z"]` only.
- Orphanhood is because **both setups filter `+Z`**, not because reachability is
  empty or wrong for this feature type.

**Recommendation:** **Fix planner approach-dir mapping** (rear/back ? `?Z`, front ?
`+Z`); do not add a third setup or drop the feature.

---

### Rear feature 37

| Field | Value |
|-------|-------|
| `class_name` | `contour_surface` |
| Faces | `330` (plane), `331` (cylinder) |
| Area / bbox | 123 mm�; bbox 3.43 � 0.43 � 6.10 mm |
| Face normals (viewer) | 330: `[-1, 0, 0]`; 331: `[0, 0, 1]` |
| Centroids (viewer) | 330: `(?3.43, ?88.67, 70.07)`; 331: `(0, ?88.12, 77.56)` |

**Step-2 approach**

```json
{
  "axis": [-0.563881, 0.0, 0.825856],
  "source": "aggregate_normal",
  "setup_dir": "-Z",
  "reachable_3axis": true
}
```

(`dot(axis, opening_axis) = 0` � **oblique** to �Y; step-2 candidacy should be null,
but step-4a overwrote `setup_dir` from verification.)

**`node.approach.reachability` (verbatim)**

```json
{
  "verified": true,
  "reachable_dirs": ["-Z"],
  "occluded": false,
  "required_depth_mm": 7.069149,
  "effective_tool_radius_mm": 4.7625,
  "per_direction": {
    "+Z": {
      "occluded": true,
      "exposed_fraction": 0.0,
      "n_targets": 12,
      "required_depth_mm": 7.069149,
      "effective_tool_radius_mm": 4.7625
    },
    "-Z": {
      "occluded": false,
      "exposed_fraction": 0.333,
      "n_targets": 12,
      "required_depth_mm": 7.069149,
      "effective_tool_radius_mm": 4.7625
    }
  },
  "corrected_from": null
}
```

**Classification: (ii) genuine lateral / third-approach edge detail (with reachability false-positive on `?Z`)**

**Evidence**

- Dominant face 330 is a **?X shelf** (normal `[?1,0,0]`); companion cylinder 331
  is **+Z-facing**. True finishing wants **�X** (or wrapped wall/contour), not �Y.
- Aggregate normal is **perpendicular** to opening_axis; step-2 should mark
  `setup_dir: null`. Step-4a still reports `?Z` reachable at 33% exposed � likely
  edge grazing from ?Y rays, not a valid 3-axis facing strategy for a �X plane.
- On the front export the same edge splits into **front 36** (face 281, normal
  `?X`) and **front 37** (face 283, normal `+X`), each ~50 mm�, also
  `reachable_dirs: ["-Z"]` only with step-2 axes `�X` � confirming lateral intent.
- Too small and geometrically distinct from the perimeter chamfer (rear 36) to
  dismiss as the same ?Y back-side part.

**Recommendation:** **Add lateral setup** (or fold into wall/edge finishing and
**drop** standalone contour ops). Do not rely on `?Z` reachability for this face;
treat reported `?Z` as a ray-test artifact.

---

## B. Migration check � rear drops 17, 102, 103, 104

Question: after rear reachability drop, are these ids kept in front�s
reachability-scoped assignment (38 features)?

| Feature | In front graph? | In front reachability keep (`+Z`)? | Verdict |
|---------|-----------------|-------------------------------------|---------|
| **17** | Yes � but **different feature** (`filleted_open_pocket`, 11 faces, `reachable_dirs: ["+Z","-Z"]`) | **Yes** | **Yes** for id 17 on front, **No** for migrating rear flat 17 |
| **102** | **No** (id absent) | � | **No** � additional rear-only orphan at family id |
| **103** | **No** (id absent) | � | **No** � additional rear-only orphan at family id |
| **104** | **No** (id absent) | � | **No** � additional rear-only orphan at family id |

**Clarification:** Front/rear graphs **do not share a global feature-id map**. Rear
17 is a **flat** (face 326, Y ? ?85, 2463 mm�, `reachable_dirs: ["-Z"]`). Front
17 is an unrelated **filleted_open_pocket**. The id-17 �migration� is a **numeric
collision**, not geometry hand-off.

**Geometric twins on front** (same part, different ids) � all **`?Z` only**, also
in front�s gap list, so they do **not** land in front�s 38 under `+Z` filtering:

| Rear drop | Rear geometry | Front twin id | Front faces |
|-----------|---------------|---------------|-------------|
| 102 | torus 323, 1351 mm� | **34** | 274 |
| 103 | plane +X, 332 | **36** | 281 |
| 104 | torus 345, 1487 mm� | **26** (partial) | 277, 296 |

**Additional orphans beyond 36/37:** rear **102, 103, 104** have no front id and
their front twins are also `+Z`-filtered out.

---

## C. contour_surface / flat reachability � is step-4a running?

For rear class-filter drops **17, 36, 37, 102, 103, 104** (all `flat` or
`contour_surface`):

| feature | class | `reachability.verified` | `reachable_dirs` | Points to back (`?Z` / ?Y)? | Empty / no-op? |
|---------|-------|-------------------------|------------------|------------------------------|----------------|
| 17 | flat | `true` | `["-Z"]` | **Yes** � flat normal ?Y, 25% exposed on `?Z` | **No** � fully populated `per_direction` |
| 36 | contour_surface | `true` | `["-Z"]` | **Yes** � 16.7% on `?Z`, 0% on `+Z` | **No** |
| 37 | contour_surface | `true` | `["-Z"]` | **Questionable** � lateral faces; 33% on `?Z` likely edge artifact | **No** � computation ran |
| 102 | contour_surface | `true` | `["-Z"]` | **Yes** � torus at back lip, 58.3% on `?Z` | **No** |
| 103 | contour_surface | `true` | `["-Z"]` | **No** � normal `+X`; `?Z` is false positive | **No** � computation ran |
| 104 | contour_surface | `true` | `["-Z"]` | **Yes** � oblique back band, 25% on `?Z` | **No** |

**Conclusion:** Reachability is **not** silently no-op�ing for `contour_surface` /
`flat`. Every node has `verified: true`, non-empty `reachable_dirs`, and full
`per_direction` occlusion scores. Drops are because the planner requires `+Z` while
these nodes (mostly correctly) verify only `?Z`.

They are **not** �surviving by luck� with empty `reachable_dirs`; they are
**correctly verified as `?Z`-reachable** and then filtered out by the uniform `+Z`
setup filter.

---

## Summary table

| Feature | Classification | One-line recommendation |
|---------|----------------|-------------------------|
| **36** (rear; twin front 25) | Back-side perimeter chamfer; `?Z` verified | **Fix approach-dir mapping** (rear ? `?Z`) |
| **37** (rear; twins front 36/37) | Lateral �X edge; `?Z` reachability suspect | **Add lateral setup** or **drop** as contour artifact |
| **17** (rear flat) | `?Z` flat; not same as front id 17 | **Fix approach-dir mapping** for rear |
| **102�104** (rear only ids) | `?Z` back contours; no front id | **Fix approach-dir mapping**; map twins by geometry not id |

## Findings for downstream fixes (out of scope here)

1. **Planner:** `_primary_setup_approach_dir` must depend on `machining_side` /
   setup descriptor, not hard-coded `+Z` for both panels.
2. **Id scheme:** Multi-setup triage cannot assume feature ids match across panel
   exports; correlate by face geometry or a part-family id map.
3. **Reachability vs lateral faces:** Features with step-2 axis ? opening_axis but
   `?Z` verified (37, 103) need explicit lateral handling or stricter oblique gating
   before assigning `setup_dir` from ray tests alone.

---

## Post-fix validation (per-setup opening axis)

Date: 2026-07-08. After fixing `_primary_setup_approach_dir()` /
`_reachability_dir_for_setup()` to source the descriptor opening axis (+Y) and map
`machining_side` to the reachability frame token (`front` -> `+Z`, `back` -> `-Z`).

### Resolved opening axis per setup (printed by scope diff)

| Setup | `opening_axis` | `opening_axis_vector` | `reachability_dir` |
|-------|----------------|----------------------|--------------------|
| rear  | **+Y** | `[0.0, 1.0, 0.0]` | **-Z** (approach along -Y) |
| front | **+Y** | `[0.0, 1.0, 0.0]` | **+Z** (approach along +Y) |

Not hardcoded world +Z; both panels share descriptor +Y, but opposite reachability
filter dirs per `machining_side`.

### Scope-diff before/after (wrong-axis era vs fixed)

| Setup | Metric | Wrong-axis (+Z both) | Fixed (per-side) | Delta |
|-------|--------|----------------------|------------------|-------|
| rear  | reachability kept | 98 | **36** | -62 (back-side filter; +Z-only features correctly drop) |
| rear  | class-only drops | [17, 36, 37, 102, 103, 104] | **resolved** — no longer in drop list | -6 rear false drops |
| front | reachability kept | 38 | **39** | +1 |
| front | class-only drops | [25, 26, 34, 36, 37] | unchanged gap list | 0 |

Rear kept count fell because the old run incorrectly retained ~62 `+Z`-only features
(pockets, +Y flats, etc.) on the back setup. That was the wrong-axis artifact, not
a regression.

### Orphan resolution table

| Feature / twin | Rear (-Z) | Front (+Z) | Status |
|----------------|-----------|------------|--------|
| **36** (rear perimeter chamfer) | **IN** | n/a (twin **25** on front) | **Resolved** on rear |
| **102** (rear torus 323) | **IN** | n/a (twin **34** on front) | **Resolved** on rear; front twin still -Z-only |
| **103** (rear +X shelf 332) | **IN** | n/a (twin **36** on front) | **Resolved** on rear |
| **104** (rear torus 345) | **IN** | n/a (twin **26** on front) | **Resolved** on rear |
| **17** (rear flat 326) | **IN** | id collision only (front 17 is a pocket) | Rear flat resolved |
| **37** (rear lateral edge) | **IN** (rear -Z) | **OUT** (front 37 is different face, -Z only) | **Remains flagged** — rear assignment is ray-test suspect; not forced across panels |
| front **25, 26, 34** | twins on rear graph | **OUT** | Correct: -Z-only, belong on rear |
| front **36, 37** | different faces than rear 36/37 | **OUT** | **Remain** as front-panel gaps (lateral edge planes) |

**Cross-panel shared-id orphans after fix:** only **[34]** (+Z-only on rear graph;
-Z-only on front graph as feature 34). Previously [36, 37] under wrong-axis +Z filter.

**Front reachability gap (unchanged):** `[25, 26, 34, 36, 37]` — all `-Z`-only on
the front export; correctly excluded from `+Z` front filter. Geometric twins land on
rear via `-Z` filter.

### CamPlan

`examples/cam_plan_96260B.json` regenerates and passes `CamPlan.model_validate`.
Ops: rear 5, front 11 (16 total). Per-setup stats include `opening_axis`, `opening_axis_vector`,
`reachability_dir`.

### Cross-panel feature-id grep (report only)

**`planner.py`:** No code joins or matches features across setups by `feature_id`.
`plan_multi_setups()` plans each graph independently; `_apply_cross_setup_precedence()`
only chains ops by setup order, not by feature id.

**Latent bug elsewhere:** `scripts/plan_sanity_report.py` `check_homolog_overlap()`
assumes the same `feature_id` string across setups denotes the same physical feature
when flagging duplicate `(feature_id, operation_type)` pairs. That assumption is
**false** for split-panel exports (e.g. front id 17 != rear id 17). Out of scope
for this fix; needs a geometry-based homolog map before enabling that gate on
multi-panel parts.
