"""
diagnose_face_count.py — why do STEP face counts != H5 node counts?

For sampled mismatched models, inspect the STEP side:
  - NumberOfRoots, OneShape() shape type, #solids
  - faces via whole-shape TopologyExplorer (dedup) vs raw TopExp_Explorer (no dedup)
  - faces summed per-solid (counts shared faces once per solid they bound)
  - shared faces (same TShape appearing in >1 solid)
and the H5 side:
  - num_faces, and duplicate (area,centroid,type) rows (split/aux faces?)

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/diagnose_face_count.py
"""
import json
import os
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import (TopAbs_FACE, TopAbs_SOLID, TopAbs_COMPOUND,
                             TopAbs_COMPSOLID, TopAbs_SHELL, TopAbs_SHAPE)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Extend.TopologyUtils import TopologyExplorer

SHTYPE = {TopAbs_COMPOUND: "COMPOUND", TopAbs_COMPSOLID: "COMPSOLID",
          TopAbs_SOLID: "SOLID", TopAbs_SHELL: "SHELL", TopAbs_FACE: "FACE"}


def raw_face_count(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    n = 0
    while exp.More():
        n += 1; exp.Next()
    return n


def main():
    with open("diag/step_candidates.json") as f:
        cands = {c["model_id"]: c for c in json.load(f)}
    for mid in ["4482", "5875", "26677", "33892", "5391", "48827"]:
        if mid not in cands:
            print(f"{mid}: not in candidates json"); continue
        c = cands[mid]
        path = os.path.join("MFCAD++_dataset", "step", c["split"], f"{mid}.step")
        r = STEPControl_Reader()
        if r.ReadFile(path) != IFSelect_RetDone:
            print(f"{mid}: read fail"); continue
        nroots = r.NbRootsForTransfer()
        r.TransferRoots()
        shape = r.OneShape()
        te = TopologyExplorer(shape)
        solids = list(te.solids())
        faces_dedup = list(te.faces())
        n_raw = raw_face_count(shape)
        # faces per solid summed (shared faces counted once per bounding solid)
        per_solid = [len(list(TopologyExplorer(s).faces())) for s in solids]
        # shared faces: appear in raw enumeration more than once vs dedup map
        fmap = TopTools_IndexedMapOfShape()
        from OCC.Core.TopExp import topexp
        topexp.MapShapes(shape, TopAbs_FACE, fmap)
        n_unique = fmap.Size()

        # H5 side
        v1 = np.array(c["v1"]); nh5 = c["num_faces"]
        key = np.round(v1[:, :4], 5)
        _, counts = np.unique(key, axis=0, return_counts=True)
        n_dup_h5 = int(np.sum(counts > 1))

        print(f"\n=== model {mid} ===")
        print(f"  STEP: roots={nroots} oneshape={SHTYPE.get(shape.ShapeType(),'?')} "
              f"solids={len(solids)}")
        print(f"  STEP faces: dedup(TopologyExplorer)={len(faces_dedup)} "
              f"unique(MapShapes)={n_unique} raw(no-dedup)={n_raw} "
              f"sum-per-solid={sum(per_solid)} per_solid={per_solid}")
        print(f"  H5 nodes={nh5}  (dup V_1 [area,centroid] rows: {n_dup_h5})")
        print(f"  => STEP unique {n_unique} vs H5 {nh5} : "
              f"{'MATCH' if n_unique == nh5 else 'MISMATCH'}; "
              f"raw {n_raw} vs H5 {nh5}: {'MATCH' if n_raw == nh5 else 'MISMATCH'}; "
              f"sum-per-solid {sum(per_solid)} vs H5 {nh5}: "
              f"{'MATCH' if sum(per_solid) == nh5 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
