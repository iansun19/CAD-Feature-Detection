#!/usr/bin/env python3
"""
Stage D (OCC/conda): round-trip verification of the annotated STEP.
Re-parse fixtures/step/fixtures/step/nist_ctc_01_annotated.step, read back the 139 ADVANCED_FACE name fields
via the SAME verified face->entity map, and confirm:
  (1) each face's read-back name == predicted class id in the jsonl,
  (2) face count unchanged (139),
  (3) geometry unchanged: area/centroid per node identical to the original Stage-A npz.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python nist_stage_d_verify.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from brep.step_ingest import extract_brep_from_step, load_step_shape, sprops  # noqa: E402
from OCC.Extend.TopologyUtils import TopologyExplorer  # noqa: E402
from OCC.Core.StepRepr import StepRepr_RepresentationItem  # noqa: E402

ANNOTATED = os.path.join(ROOT, "fixtures/step/fixtures/step/nist_ctc_01_annotated.step")
NPZ = os.path.join(ROOT, "nist_ctc_01_stage_a.npz")
JSONL = os.path.join(ROOT, "nist_ctc_01_predictions.jsonl")


def read_names_in_kept_order(path):
    """Same kept-face ordering + offset as Stage A; return (#N, name) per node."""
    shape, treader = load_step_shape(path)
    reader = treader._step_reader_ref
    m = reader.WS().Model()
    nums, names, areas, cents = [], [], [], []
    for face in TopologyExplorer(shape).faces():
        gp = sprops(face)
        area = float(gp.Mass())
        if area < 1e-12:
            continue
        item = treader.EntityFromShapeResult(face, 1)
        nums.append(int(m.Number(item)))
        ri = StepRepr_RepresentationItem.DownCast(item)
        names.append(ri.Name().ToCString() if ri is not None else None)
        c = gp.CentreOfMass()
        areas.append(area)
        cents.append([c.X(), c.Y(), c.Z()])
    return (np.array(nums, np.int64), names,
            np.array(areas, np.float64), np.array(cents, np.float64))


def main():
    print(f"[Stage D] re-parsing annotated STEP: {ANNOTATED}")
    preds = [json.loads(l) for l in open(JSONL)]
    pred_by_node = {r["face_id"]: r for r in preds}
    N_pred = len(preds)

    nums, names, areas, cents = read_names_in_kept_order(ANNOTATED)
    N = len(nums)
    print(f"[Stage D] faces re-parsed = {N} (expected {N_pred})")
    assert N == N_pred, "face count changed on round-trip!"

    # offset from original npz entity_ids (node i -> #N)
    d = np.load(NPZ)
    orig_ent = d["entity_ids"].astype(np.int64)
    offset = int(orig_ent[0] - nums[0])
    ent_ids = nums + offset

    # (1) name read-back matches predicted class id, node by node
    mismatches, ambiguous = [], []
    for i in range(N):
        name = names[i]
        exp = pred_by_node[i]["class_id"]
        if ent_ids[i] != pred_by_node[i]["entity_id"]:
            ambiguous.append((i, int(ent_ids[i]), pred_by_node[i]["entity_id"]))
        if name is None or not name.lstrip("-").isdigit() or int(name) != exp:
            mismatches.append((i, int(ent_ids[i]), name, exp))
    print(f"[Stage D] name-field read-back mismatches: {len(mismatches)}")
    for m in mismatches[:10]:
        print(f"    node {m[0]} #{m[1]}: read '{m[2]}' expected {m[3]}")
    print(f"[Stage D] ambiguous entity-map cases: {len(ambiguous)}")
    for a in ambiguous[:10]:
        print(f"    node {a[0]}: mapped #{a[1]} but jsonl entity #{a[2]}")

    # (2)+(3) geometry unchanged vs original Stage-A npz
    model, _ = extract_brep_from_step(ANNOTATED, require_labels=False)
    da = np.abs(model["area"] - d["area"]).max()
    dc = np.linalg.norm(model["cent"] - d["cent"], axis=1).max()
    print(f"[Stage D] geometry unchanged: max|Δarea|={da:.3e} max‖Δcentroid‖={dc:.3e}")
    assert model["N"] == N, "geometry face count changed"
    assert da < 1e-9 and dc < 1e-9, "geometry drifted on round-trip!"

    ok = (not mismatches) and (not ambiguous)
    print("\n[Stage D] ROUND-TRIP " + ("VERIFIED ✅" if ok else "FAILED ❌") +
          f": {N}/{N} name fields == predictions, geometry byte-stable, "
          f"no ambiguous mappings" if ok else "")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
