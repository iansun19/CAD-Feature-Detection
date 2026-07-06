# Pocket open/closed — decision memo

**Date:** 2026-07-04  
**Status:** No discriminator is wireable. Do **not** implement `classify_pocket_toolpath`.

This memo records the outcome of diagnostic work (`probe_pocket_open_closed.py` and follow-on read-only analysis on 96260B front/back STEPs). It is a decision record, not a spec for implementation.

---

## Question

Can we classify filleted pockets as **open** vs **closed** (Toolpath: `filleted_open_pocket` vs `filleted_pocket`) from B-rep geometry alone, before wiring into the cascade emit path?

**Answer: No — not with any candidate tested, and not without external ground truth.**

---

## Three candidates tested

### 1. Wall-loop boundary topology — **falsified**

**Rule:** Walk wall-face rim edges in the plane ⊥ opening axis. **Closed** if every rim edge mates wall→wall in-set; **open** if any rim edge exits to a non-wall face.

**Result:** Both parts produce **watertight analytic shells** — zero rim exit edges on all 14 pockets (7 front + 7 back). The predicate calls **everything closed**. It only “matches” the back part degenerately (correct label, wrong reason). It cannot distinguish front-open from back-closed.

**Verdict:** Falsified. Do not wire.

---

### 2. Raw step-plane count (≥ 2 → closed) — **confounded**

**Rule:** Count claimed axial step planes per pocket instance. One step → open; two or more → closed.

**Result:** Scores **14/14** against Toolpath instance counts on the two reference STEPs:

| STEP file | Y extent (mm) | Steps / lobe | Toolpath label |
|-----------|---------------|--------------|----------------|
| Front (`96260B_FRONT_…`) | −54.83 … −21.65 | **1** @ ≈ −40.2 | filleted_open_pocket (×7) |
| Back (`96260B_REAR_…`) | −91.66 … −54.56 | **2** @ ≈ −77.0, −82.1 | filleted_pocket (×7) |

**Confound:** Front and back are **different geometric subsets** of the same plate, split near **Y ≈ −54.8 mm**. The front STEP literally does not contain the second deep step band (−82 mm). Step count tracks **which STEP / which part-side is loaded**, not an intrinsic open/closed property of a lobe invariant across sides.

Grow-pass audit: front “missing” second step is **not** a pipeline artifact (no −82 geometry in file). Back second step is real; grow conflict reassignment preserves 2 steps/pocket. Unclaimed Y ≈ −45.3 planes on front are torus-bounded sculpted-floor ring, not pocket steps.

**Verdict:** Consistent with **n = 2** confounded examples only. Not validated. Do not wire.

---

### 3. Opening-side cap — **falsified**

**Rule:** **Closed** if a claimed axial plane caps the opening between wall rim and part top (toward exterior along +Y).

**Result:** **Opening band is empty** on all 14 pockets — no claimed plane between rim and part top on either part. Predicate → **open** everywhere.

**Disagreement with step-count:** **7/7 back pockets** — step-count says closed, opening-side cap says open. This is the only explicit rule-vs-rule disagreement in current data.

Back “closure” is **deep-side** (two step planes **below** wall rim at −77 / −82), not a cap at the rim toward part top.

**Verdict:** Falsified as Toolpath’s discriminator. Deep-band closure (≥ 2 steps below rim) collapses to step-count on available data — same confound.

---

## STEP-extent confound (interface Y ≈ −54.8)

```
        part top
           │
  Front STEP │  Y ∈ [−54.8, −21.7]   →  1 step / lobe  →  Toolpath: open (front panel)
  ───────────┼── interface ≈ −54.8
  Back STEP  │  Y ∈ [−91.7, −54.6]   →  2 steps / lobe →  Toolpath: closed (back panel)
           │
        part bottom (back)
```

Any rule that uses step count or deep-band plane count without knowing **machining setup / view side** is measuring **file extent**, not pocket geometry.

---

## Taxonomy hypothesis (same 7 lobes, opposite labels)

The spatial pocket pass finds **the same 7 lobes** on both STEPs (spatial clustering, ~9 walls + step planes per lobe). Toolpath ground truth (merged `96260B_front.yaml` per-side breakdown) lists:

- **Front panel:** 7 × `filleted_open_pocket` **and** 7 × `filleted_pocket`
- **Back panel:** 7 × (not listed as filleted pockets on `96260B_plate.yaml`); plus 1 × `open_pocket`

**Hypothesis:** Toolpath’s open/closed split for the filleted lobes is **not a second pocket instance** — it is **the same 7 lobes labeled by machining side / setup**:

- Front-facing access → `filleted_open_pocket`
- Back-facing access → `filleted_pocket` (closed from that side)

If true, open/closed is **setup-dependent** (which side you machine from), not a geometry-intrinsic boolean on a single face set. A geometry-only classifier on one STEP cannot recover both labels without setup context.

**Plain `open_pocket` (GT = 1):** Attributed to **back only** in GT comments. Geometry at Y ≈ −85 … −92 (faces 322, 326, 330, 331, 332 on back STEP) is **absent from front STEP**. Face 277 (front “extra flat”) is **ruled out** — ~20 mm off lobe rim, not a pocket wall boundary. Strict unfilleted-pocket search found **zero** candidates. Likely a **back-panel taxonomy** issue (aggregate label vs central through-hole stack) — not resolved without Toolpath face export.

---

## Cascade impact (unchanged)

`_extract_cascade_instances` still hardcodes `tp_class="filleted_pocket"` for all pockets. Fixing open/closed is **blocked** until a validated discriminator or setup-aware labeling exists. No wiring attempted.

---

## What would unblock this

### (a) Toolpath per-face export → `face_truth` in `96260B_front.yaml`

**Need:** Toolpath (or manual transcription from Feature Details with face IDs) exporting **B-rep face indices** aligned to `analyze_step` / TopologyExplorer order — same index space as `graph.npz`.

**Minimum face sets to label:**

| Instance | Count | What to export | Why |
|----------|-------|----------------|-----|
| Filleted lobe pockets | **7** | Full claimed face set per lobe (walls + step planes + any Toolpath-included floors/fillets) | Disambiguate one label vs two per lobe |
| Plain open pocket | **1** | Face set Toolpath assigns to `open_pocket` on back | Currently unknown; not face 277 |
| Optional central stack | — | 322, 326, 330, 331, 332 if Toolpath groups them with `open_pocket` or another class | Resolves back-only GT=1 |

**Known per-lobe step-plane face IDs (front STEP, July 2026 pocket pass):**

| Lobe | Step face | Step Y (mm) | Faces / lobe (count) |
|------|-----------|-------------|----------------------|
| 0 | 80 | −40.2 | 10 (9 walls + 1 step) |
| 1 | 67 | −40.2 | 10 |
| 2 | 0 | −40.2 | 10 |
| 3 | 54 | −40.2 | 10 |
| 4 | 15 | −40.2 | 10 |
| 5 | 41 | −40.2 | 10 |
| 6 | 28 | −40.2 | 10 |

Full wall+step face lists for each lobe, and all back-STEP lobe sets (11 faces, steps at Y ≈ −77 / −82), **must come from Toolpath export** — not inferred in this memo. Toolpath may also include faces outside the analytic claimed set (sculpted floors, fillet rings, contour).

**Critical question for face_truth:**

For each of the **7 lobes**, does Toolpath assign:

- **One class** (e.g. only `filleted_open_pocket` on front setup), or  
- **Two classes on the same faces** (both `filleted_open_pocket` and `filleted_pocket` in merged GT = 14 instances on two setups, not 14 spatial pockets)?

Export should state **class + face list per instance**, not just instance counts.

**Suggested `face_truth` shape (once export exists):**

```yaml
face_truth:
  filleted_open_pocket:
    - faces: [...]   # lobe 0, front setup
    - ...
  filleted_pocket:
    - faces: [...]   # lobe 0, back setup — SAME lobe or different?
    - ...
  open_pocket:
    - faces: [...]   # back-only instance
```

---

### (b) Setup-dependent vs geometry-intrinsic — confirmation needed

**Need explicit answer from Toolpath behavior (documentation or experiment):**

1. **Setup-dependent:** Open/closed is defined relative to **machining orientation / which side is “open” to the tool**. Same face set carries different feature class from front vs back setup. → Geometry-only rules on a single STEP will always confound with file extent; classifier needs **setup vector** (opening axis + which end is accessible) or **per-setup STEP**, not face count alone.

2. **Geometry-intrinsic:** Open/closed is a **property of the solid** (e.g. blind vs through, capped vs uncapped in a fixed axis frame). → Requires at least **one disagreement case** where step-count and Toolpath disagree, or face_truth proving the discriminating faces. **None exists in current 96260B data.**

**Experiment that would decide:** Load **one combined full-thickness STEP** (if available) or the same lobe in a single model; run Toolpath from front and back setups; compare whether feature class changes while face set is identical.

---

## Decision

| Action | Status |
|--------|--------|
| Implement `classify_pocket_toolpath` | **Do not** |
| Wire open/closed into cascade emit | **Do not** |
| Edit `eval_cascade.py` or GT yaml | **Do not** (until face_truth exists) |
| Next step | Per-face Toolpath export + setup-intrinsic confirmation |

---

## References

- Diagnostic script: `probe_pocket_open_closed.py`
- GT: `eval/gt/96260B_front.yaml`, `eval/gt/96260B_plate.yaml`
- Flats/step-band context: `probe_flats_gap.py` (groups 0/1 = Y ≈ −77 / −82 step bands on back)
