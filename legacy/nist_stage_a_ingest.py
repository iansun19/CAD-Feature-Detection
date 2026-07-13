#!/usr/bin/env python3
"""
Stage A (OCC/conda env): NIST CTC-01 STEP -> production B-rep graph + verified
face->ADVANCED_FACE-entity map. Writes an intermediate .npz consumed by Stage B
(torch env). Graph construction reuses the exact production functions in
step_ingest.py (ingest_step_to_pyg == training-time graph build). The ONLY new
code here is file-loading glue and the entity-id map + its verification.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python nist_stage_a_ingest.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from OCC.Core.StepRepr import StepRepr_RepresentationItem  # noqa: E402
from OCC.Extend.TopologyUtils import TopologyExplorer  # noqa: E402

from brep.step_ingest import (  # noqa: E402
    DEFAULT_MIN_FACE_AREA,
    extract_brep_from_step,
    ingest_step_to_pyg,
    load_step_shape,
    sprops,
)

STEP_PATH = os.path.join(ROOT, "fixtures/step/fixtures/step/nist_ctc_01.step")
OUT_NPZ = os.path.join(ROOT, "nist_ctc_01_stage_a.npz")
NUM_SURFACE_TYPES = 6
ANGLE_REDUCE = "median"


def entity_map_in_kept_order(path, min_face_area=DEFAULT_MIN_FACE_AREA):
    """Reproduce extract_brep_from_step's kept-face ordering and, for each kept
    face, record (STEP entity #N, area, centroid). Filtering must mirror
    step_ingest exactly: drop only faces with area < min_face_area."""
    shape, treader = load_step_shape(path)
    reader = treader._step_reader_ref  # pinned by load_step_shape
    model_step = reader.WS().Model()  # StepData_StepModel: Number(ent) -> #N
    topo = TopologyExplorer(shape)
    ent_ids, areas, cents, names = [], [], [], []
    for face in topo.faces():
        gp = sprops(face)
        area = float(gp.Mass())
        if area < min_face_area:
            continue  # identical skip rule to extract_brep_from_step
        item = treader.EntityFromShapeResult(face, 1)
        if item is None:
            ent_ids.append(-1)
            names.append(None)
        else:
            # model.Number(ent) is the internal sequence; the STEP file #N label
            # differs by a constant offset (header/context entities). Return the
            # raw Number here; main() resolves offset -> #N and verifies bijection.
            num = model_step.Number(item)
            ent_ids.append(int(num))
            ri = StepRepr_RepresentationItem.DownCast(item)
            names.append(ri.Name().ToCString() if ri is not None else None)
        c = gp.CentreOfMass()
        areas.append(area)
        cents.append([c.X(), c.Y(), c.Z()])
    return (np.array(ent_ids, np.int64), np.array(areas, np.float64),
            np.array(cents, np.float64), names)


def grep_advanced_face_ids(path):
    """Bare-text list of #N ids that are ADVANCED_FACE entities (ground truth for
    what the entity map must land on)."""
    ids = []
    pat = re.compile(r"#(\d+)\s*=\s*ADVANCED_FACE\(")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                ids.append(int(m.group(1)))
    return ids


def main():
    print(f"[Stage A] STEP file: {STEP_PATH}")
    assert os.path.isfile(STEP_PATH), "fixtures/step/fixtures/step/nist_ctc_01.step missing"

    af_ids = grep_advanced_face_ids(STEP_PATH)
    print(f"[Stage A] ADVANCED_FACE entities in file text: {len(af_ids)}")

    # --- production graph build (identical to training-time construction) ---
    print("[Stage A] building graph via ingest_step_to_pyg (production path)...")
    x, edge_index, edge_attr, stats = ingest_step_to_pyg(
        STEP_PATH, num_surface_types=NUM_SURFACE_TYPES, angle_reduce=ANGLE_REDUCE)
    N = x.shape[0]
    print(f"[Stage A] graph faces (nodes) = {N}")
    print(f"[Stage A] node feat dim = {x.shape[1]}  edge feat dim = {edge_attr.shape[1]}"
          f"  edges = {edge_index.shape[1]}")
    print(f"[Stage A] stats: face_count={stats.face_count} "
          f"zero_area_faces={stats.zero_area_faces} "
          f"surface_type_counts={stats.surface_type_counts}")

    # geometry reference from the production model dict (same call, for verify)
    model, _ = extract_brep_from_step(STEP_PATH, require_labels=False)
    ref_area, ref_cent = model["area"], model["cent"]
    assert model["N"] == N, f"model N {model['N']} != pyg N {N}"

    # --- verified entity map, in kept-face order ---
    raw_nums, areas, cents, names = entity_map_in_kept_order(STEP_PATH)
    print(f"[Stage A] entity-map faces = {len(raw_nums)}")

    # resolve internal Number -> file #N by constant offset, then verify the
    # mapped set is EXACTLY the ADVANCED_FACE #N set (bijection, order-agnostic).
    offset = min(af_ids) - int(raw_nums.min())
    ent_ids = raw_nums + offset
    offsets_all = sorted({int(af_ids[i] - raw_nums[i]) for i in range(len(af_ids))})
    print(f"[Stage A] Number->#N offset = {offset} "
          f"(per-face offsets observed: {offsets_all})")
    assert len(offsets_all) == 1 and offsets_all[0] == offset, (
        "offset not constant across faces — topo order != file order, cannot map")

    # face-count reconciliation
    assert len(ent_ids) == N, f"entity-map faces {len(ent_ids)} != graph nodes {N}"
    if N != len(af_ids):
        print(f"[Stage A] WARNING: graph nodes {N} != ADVANCED_FACE count {len(af_ids)}"
              " — a face was dropped or geometry differs from text.")
    else:
        print(f"[Stage A] face-count match: nodes == ADVANCED_FACE count == {N}")

    # VERIFY positional correspondence entity-order <-> node-order via geometry
    da = np.abs(areas - ref_area)
    dc = np.linalg.norm(cents - ref_cent, axis=1)
    scale = np.linalg.norm(ref_cent.max(0) - ref_cent.min(0)) + 1e-9
    print(f"[Stage A] correspondence residuals: "
          f"max|Δarea|={da.max():.3e} (rel {da.max()/(ref_area.max()+1e-9):.2e}), "
          f"max‖Δcentroid‖={dc.max():.3e} (rel {dc.max()/scale:.2e})")
    assert da.max() < 1e-6 * (ref_area.max() + 1.0), "area mismatch: ordering broke"
    assert dc.max() < 1e-6 * scale, "centroid mismatch: ordering broke"
    print("[Stage A] VERIFIED: entity-map order == graph-node order (geometry bijection)")

    # every mapped entity must be a real ADVANCED_FACE #N, and cover all 139
    bad = [int(e) for e in ent_ids if e not in set(af_ids)]
    assert not bad, f"entity ids not ADVANCED_FACE: {bad[:10]}"
    assert len(set(ent_ids.tolist())) == len(ent_ids), "duplicate entity ids in map"
    if set(ent_ids.tolist()) == set(af_ids):
        print(f"[Stage A] VERIFIED: mapped entity ids are exactly the "
              f"{len(af_ids)} ADVANCED_FACE #N ids (bijection)")
    else:
        missing = set(af_ids) - set(ent_ids.tolist())
        print(f"[Stage A] WARNING: {len(missing)} ADVANCED_FACE ids unmapped: "
              f"{sorted(missing)[:10]}")

    # confirm all name fields empty (leak check, geometry side)
    nonempty = [(int(ent_ids[i]), names[i]) for i in range(N)
                if names[i] not in (None, "")]
    print(f"[Stage A] non-empty ADVANCED_FACE names among mapped faces: {len(nonempty)}"
          + ("" if not nonempty else f"  !! {nonempty[:5]}"))

    np.savez(
        OUT_NPZ,
        x=x.astype(np.float32),
        edge_index=edge_index.astype(np.int64),
        edge_attr=edge_attr.astype(np.float32),
        entity_ids=ent_ids.astype(np.int64),
        area=ref_area.astype(np.float64),
        cent=ref_cent.astype(np.float64),
        # adjacency (undirected canonical pairs) for cluster-coherence in Stage B
        adj_src=edge_index[0].astype(np.int64),
        adj_dst=edge_index[1].astype(np.int64),
    )
    print(f"[Stage A] wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
