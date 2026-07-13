# Upstream task — `_grow_open_pocket` axial band is hardcoded to Y

**Status:** open (filed while fixing open-pocket over-labeling on part1/part2).
**Severity:** latent correctness bug; currently masked by Predicate A.
**File:** [coaxial_stack_detection.py](../coaxial_stack_detection.py) — `_grow_open_pocket`.

## The bug

`_grow_open_pocket` bands its BFS on a **hardcoded world `centroid[1]` (Y)**
axis, ignoring the actual `opening_axis` passed to the coaxial pass:

```python
seed_y = float(by_index[seed].centroid[1])          # index 1 == Y, always
...
if abs(float(by_index[nb].centroid[1]) - seed_y) > axial_band_mm:
    continue
```

The rest of the coaxial pass is axis-agnostic (`opening_axis` is threaded through
`_face_depth_below_top`, `_is_horizontal_plane`, etc.), but the grow assumes the
96260B convention that the hub opens along +Y. On a part whose hub opens along X
or Z the band compares the wrong coordinate.

## How it was exposed

On part1 (opening axis Z) and part2 (opening axis X) the coaxial pass wrongly
grabbed exterior prism faces; the Y-band happened to select exactly the faces at
`y≈0` (`{5,7,9,10}` on both parts) — a coincidence of those parts' symmetry, not
a correct grow. See the diagnosis in the open-pocket over-labeling fix.

## Why it is currently masked (do NOT rely on this)

The fix added **Predicate A** (`_is_interior_recess_floor`): a floor seed with no
concave boundary edge cannot seed an open pocket, so on part1/part2 the pass now
returns before `_grow_open_pocket` runs at all. The Y-hardcode is therefore
dormant on every current fixture (96260B opens along +Y, so its real hub grow is
correct by accident of convention; part1/part2 never reach the grow). A future
part with a genuine interior recess opening along X or Z would still mis-grow.

## Proposed fix (separate change)

Band along the projection onto the true `opening_axis` instead of `centroid[1]`:

```python
seed_a = float(by_index[seed].centroid @ opening_axis)
...
if abs(float(by_index[nb].centroid @ opening_axis) - seed_a) > axial_band_mm:
    continue
```

`opening_axis` is already available in `_decompose_hub_stack` (the caller) — pass
it into `_grow_open_pocket`. Add a synthetic non-Y-axis hub to the coaxial
self-test to lock this in. Verify the 96260B front/rear goldens stay
byte-identical (their axis is +Y, so the projected band must reproduce the
current result exactly).
