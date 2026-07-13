# Open capability: cross-STEP homolog map (for genuine flip-jobs)

Date opened: 2026-07-12. **Rewritten 2026-07-12: 96260B removed as the test case
(it is not one).** Status: OPEN, but low priority — no in-repo part motivates it.

## What was wrong before

This doc originally justified a `check_homolog_overlap` behavior on 96260B: the
front/rear graphs have independent local id spaces, so cross-setup
`(feature_id, operation_type)` equality is coincidental, and the gate "skips the id
comparison and emits a WARN" naming this capability.

That was built on the phantom premise that 96260B is one split-panel part.
**96260B_FRONT and 96260B_REAR are two SEPARATE PARTS.** There is no shared
geometry between two different parts, so there are no homologs to map and nothing
to over-machine across them. Cross-part op-type overlap is a **category error**, not
a finding. Accordingly:

- `check_homolog_overlap` no longer emits even a WARN when the setups are backed by
  distinct feature graphs — it **does not run** (a category error is not a
  finding). It runs the HARD gate only for a true same-graph, one-part multi-setup.
- 96260B is **not** a test case for this capability. Its "514.6 mm walls reachable
  from both graphs" were two parts' separate walls, never one part's walls seen
  from two approaches.

## The residual (genuine) capability

For a REAL flip-job — ONE part, ONE stock, refixtured, whose orientations happen to
be exported as separate cascade graphs (separate local id spaces) — detecting that
setup B re-cuts a face setup A already finished would need a **geometry-based
homolog map**: match faces across the per-orientation exports by world-space
position and normal, not by local id. Only then can same-graph-style
over-machining detection extend to a part whose setups live in different graphs.

## Test case

None in-repo. The former "96260B front/rear" example was a miscategorization and is
withdrawn. This stays OPEN but unmotivated until a genuine multi-graph flip-job
appears; wire the test to that real part when it does.
