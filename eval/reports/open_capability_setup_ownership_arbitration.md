# RETRACTED: setup-ownership-arbitration (was "open capability")

Date opened: 2026-07-12. **Date retracted: 2026-07-12. Status: RETRACTED — not a
real capability gap for 96260B.**

## Why this is retracted

This doc claimed an open capability: "nothing arbitrates which setup *owns*
geometry that both setups can reach," using 96260B as the motivating case. That
framing was wrong at its root.

**96260B_FRONT and 96260B_REAR are two SEPARATE PARTS** — separate stock, separate
jobs, sharing only a name prefix. They were never two setups of one part. The
pipeline had *inferred* one-part-ness from the shared filename prefix and the
FRONT/REAR side-word (`setup_generation.infer_machining_side`, since removed), then
glued the two into a single phantom two-setup "96260B" plan.

Once you accept the fact that they are two parts, "which setup owns the shared
geometry both can reach" **is not a question that exists**:

- There is no shared geometry. Two parts, two stocks. The front part's walls are
  the front part's walls; the rear part's walls are the rear part's. They are not
  "the same physical walls reachable from two approaches."
- The front part's full milling program is **not over-machining**. It is the front
  part's own, complete, correct program. There is nothing for the rear to "already
  own."
- The gate reds this doc cited as "known-red true positives" (front 10 ops vs shop
  setup-2 = 1 op; front milling "unexpected") were **category errors**: comparing
  the front part against the rear part's shop program. They have been removed as
  noise, not preserved as signal. See the deliverable summary and
  `open_capability_homolog_map_split_panel.md`.

## Correction to commit 982abc2

Commit 982abc2 ("Fix facing selection and setup scope; expose setup-ownership
gap") described the front's 10-op program as over-machining and called the
resulting HARD failure "a correctly HARD-failing true positive." **That is wrong.**
The front is a separate part; its 10-op program is its own legitimate program, and
the HARD failure was an artifact of comparing two unrelated parts (the front part's
op count against the rear part's setup-2 facing flip). The record is corrected
here: there was no over-machining and no true positive — only a miscategorization.

## Does any real capability survive?

For a **genuine flip-job** (ONE part, ONE stock, refixtured so different faces face
the spindle in different setups), deciding which setup cuts geometry reachable from
more than one orientation is a real sequencing/fixturing concern. But that is not
96260B, and it is not "arbitration over shared geometry between two parts." If and
when a real multi-setup part motivates it, open a fresh doc scoped to that part —
do not resurrect this one.
