# Open capability: homolog-map-split-panel

Date opened: 2026-07-12. Status: OPEN. No code in this task attempts to fix it.

## The gap

`check_homolog_overlap` (`scripts/plan_sanity_report.py`) detects cross-setup
over-machining by keying on `(feature_id, operation_type)`. That key assumes a
SHARED feature-id space across setups, which holds only when every setup is scoped
from one feature graph.

Split-panel parts (96260B front/rear) export a **separate graph per orientation**,
each with its own local 0-based ids — front id N is not rear id N. Cross-setup id
equality is therefore coincidental, not a physical homolog (it inflated the ratio
to 78% on 96260B; see `orphaned_feature_triage.md`). So when setups are backed by
distinct feature graphs, the gate now correctly **skips the id comparison and emits
a WARN** naming this missing capability. Same-graph multi-setups still get the hard
gate. This skip is legitimate and is kept.

## What is needed

A geometry-based homolog map: match faces across STEP exports by **world-space
position and normal**, not by local id. With that map, cross-setup over-machining
detection can be restored on split-panel parts (front re-cutting a wall the rear
already finished becomes detectable regardless of id namespace).

## Test case

96260B front/rear. The front's 514.6 mm walls are reachable from both graphs; a
world-space homolog map is what would confirm they are the same physical walls the
rear already owns.

## Relationship to setup-ownership-arbitration

Likely the same fix from two directions: the homolog map tells you *which faces are
shared*, and ownership arbitration decides *which setup cuts them*. See
`open_capability_setup_ownership_arbitration.md`.
