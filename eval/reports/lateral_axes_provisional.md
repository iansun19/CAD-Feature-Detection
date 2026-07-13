# Lateral approach axes (±X/±Y/±Z) — PROVISIONAL

Status: **implemented, unit-tested on synthetic geometry, NOT validated on any
real side-access part.** Strictly the six cardinal directions; no arbitrary
face-normal orientations, no setup-count minimization.

## Key finding (design)

The prompt's guardrail asked me to stop if reachability or stock geometry was
implicitly *flattened along Z*. It is **not** — the opposite is true, and it
changes the design:

- `reachability._direction_result` / `_Corridor.blocked` cast real 3-D rays
  against the actual OCC solid along an **arbitrary** outward direction `d`. The
  collision math was never Z-specific.
- The "Z-only" limitation lived **only** in `annotate_reachability`, which
  hardcoded the direction set to `{+Z, -Z}` along the opening axis. Likewise
  `annotate_approach_vectors` only snapped candidacy to `±opening_axis`.
- The setup layer was already cardinal-general: `opening_axis` is a discrete
  label (`"+Y"`, etc.), reachability runs along that axis, and the sequencer
  already penalizes setup/approach changes. Only `machining_side` (front/back)
  was binary — it selects the *sign* along the opening axis.

So extending to six cardinals needed **no rewrite of the collision logic**: just
enumerate all six cardinals and call the unmodified primitive once per direction.
Because every reframe is a rigid rotation (proved in tests), transform-and-reuse
is exact. This is verified directly (`test_reframe_matches_direct_and_shares_transform`):
reframing the feature points **and** the part solid by the same cardinal→+Z
rotation and testing a +Z corridor gives the identical verdict to testing the
cardinal corridor directly.

**Stale-stock bug:** structurally impossible in the OCC path here, because
nothing is rotated independently — feature points, the part solid, and any future
stock/fixture solid are all queried in the one part frame along the real cardinal
vector. The invariant that would guard the *transform* path (feature and stock
reframed by the same R) is pinned in `test_axis_frames.StockConsistencyTests`
(with a negative control proving the test has teeth) and in the OCC exactness
test above. Note: reachability today only collides against the **part** solid —
there is no separate stock/fixture solid in the query (pre-existing limitation,
not introduced here).

## New files

| File | Purpose |
|---|---|
| `axis_frames.py` | Pure cardinal-axis reframe: 6 unit vectors, `rotation_to_plus_z` (proper rotation sending a cardinal to +Z), inverse, point/vector transforms, `nearest_cardinal`. Self-verifying (asserts det=1, R·d=+Z). |
| `lateral_axes.py` | PROVISIONAL. `annotate_lateral_candidates` (pure 6-cardinal candidacy from feature axes/normals), `annotate_lateral_reachability` (6-cardinal collision reusing the Z-only primitive verbatim), `approach_vector_for_setup` (machine-frame approach vector per setup). |
| `tests/test_axis_frames.py` | Round-trip, orthonormality/det, rigidity (distances/angles preserved), stock-consistency + negative control. Pure numpy. |
| `tests/test_lateral_axes.py` | Candidacy (pure), OCC reachability of a synthetic block+side-hole, transform-exactness/shared-transform, orientation-aware sequencing + approach-vector helper. |
| `scripts/smoke_lateral_axes.py` | End-to-end smoke test (see output below). |
| `eval/lateral_axes_provisional.md` | This document. |

## Changed files (all additive, backward-compatible)

| File | Change |
|---|---|
| `setup_descriptor.py` | `SetupEntry`/`ResolvedSetup` gain optional `orientation` (a cardinal label). Parsed/serialized/round-tripped. `machining_side` front/back still works unchanged (None orientation → legacy path). `CARDINAL_ORIENTATIONS` + `_parse_orientation`. |
| `machining_context.py` | `SetupContext` gains `orientation` + `orientation_provisional`; populated in `build_setup_context` from the resolved descriptor. |
| `cam_plan_schema.py` | `Setup` gains optional `orientation` + `orientation_provisional` (default None/False → old plans validate). Schema regenerated. |
| `planner.py` | `Setup` emission carries the orientation; `_setup_approach_map` now keys on `orientation` when present (else `opening_axis`), so consecutive ops in differently-oriented setups incur an approach-change cost. No calibrated thresholds touched. |
| `run_cascade.py` | New **opt-in** `--lateral-axes` flag (default OFF). When set, additively annotates lateral candidacy + reachability on export. Calibrated Z-only export is byte-for-byte unchanged when the flag is off. |

## Task-by-task

1. Read/summarized approach_vectors, reachability, setup_descriptor(+schema), setup_generation, operation_bank, score_sequence, sequence_search, planner wiring. ✓
2. **Candidacy** over all 6 cardinals from feature axes/normals (pure geometry). Directional sources (flat normal, opening axis) keep their sign; bores are two-ended. ✓
3. **Axis-permutation transform** `axis_frames.py`, independently unit-tested on synthetic geometry with inverse. ✓
4. **Reachability wrapper** reuses the Z-only primitive unchanged per cardinal; stock/feature shared-transform test added; exactness of reframe proved against the direct call on a real OCC solid. ✓
5. **Setup descriptor** general `orientation` field, front/back kept working. ✓
6. **Feature→ops**: ops carry no axis of their own — approach is the setup's orientation (verified no hardcoded Z in emission); `approach_vector_for_setup` resolves the machine-frame vector per setup. ✓
7. **Sequencing**: setup-change + approach-change cost already existed; now orientation-aware via `_setup_approach_map`. Tested. No setup-count minimization attempted. ✓
8. **Synthetic smoke test** (block + Ø12 side hole through +X): passes end-to-end. ✓
9. **Provisional flags** everywhere: module header warnings, `provisional: True` in every lateral annotation, `orientation_provisional` in descriptor/plan output, opt-in run_cascade flag. Calibrated Z-only thresholds untouched. ✓

## Smoke test output

```
solid: 7 faces | hole faces=[5] top faces=[2]
[1] candidacy: through_hole -> ['+X','-X'];  flat -> ['+Z']
[2] reachability: through_hole reachable ['+X','-X'] (NOT +Z);  flat ['+X','-X','+Y','-Y','+Z']
    ✓ side hole reachable ±X (lateral), not +Z — as expected
[3] setup descriptor round-trips orientation top=+Z side=+X
[4] sequenced ['op_face','op_drill']; approach top=[0,0,1] side=[1,0,0];
    setup_changes=1 approach_changes=1 total=111.0
[5] ✓ valid CamPlan emitted; side setup orientation=+X provisional=True
SMOKE TEST PASSED
```

Run: `env -u VIRTUAL_ENV -u PYTHONPATH \
/Users/iansun19/miniconda3/envs/mlcad/bin/python scripts/smoke_lateral_axes.py`

## What remains UNVALIDATED (needs a real side-access part)

- **Everything about correctness on real geometry.** 96260B has no side-access
  features; the only exercise is the synthetic block. No lateral result has been
  checked against a part whose actual machining used a lateral setup.
- **Candidacy tolerance** (`LATERAL_PARALLEL_TOL_DEG = 15°`, mirrors the Z-only
  value) — uncalibrated for lateral features.
- **Reachability thresholds reused from the Z path** (`EXPOSED_FRACTION_THRESHOLD`,
  ring sampling, swept-tool radius) were calibrated for axial access on a plate;
  their behavior on genuine side pockets/undercuts is unverified. The permissive
  exposed-fraction makes flat faces read reachable from many grazing cardinals
  (seen in the smoke test) — fine for a flat, unknown for real lateral features.
- **Stock/fixture collision:** reachability collides only against the part solid.
  Lateral setups are exactly where fixture/vise collision matters most; that
  model does not exist yet.
- **Front/back → signed-cardinal derivation:** when only `machining_side` (no
  explicit `orientation`) is given, orientation stays None (legacy path). Auto-
  deriving a signed cardinal from machining_side + opening axis is deferred.
- **Setup discovery/minimization:** explicitly out of scope — orientations must be
  supplied (descriptor `orientation`), not discovered. Needs the unified part
  model that does not exist yet ([[cadcam-roadmap]]).
- **run_cascade `--lateral-axes`** produces annotations but they feed nothing
  downstream automatically (no auto lateral-setup generation); consume manually.
