"""
regen_probe.py — can we regenerate a self-consistent graph from the LOCAL .step files?

Answers three things per sampled model id:
  1. LABELS: do local .step files carry per-face labels (STEP entity names), the way
     the generator's read_step_with_labels expects? -> decides if a regenerated
     dataset can have ground truth without the released H5.
  2. COUNT: regenerated face count (= TopologyExplorer faces) vs existing H5 count.
  3. SAME-PART vs DIFFERENT-PART (option D): Chamfer distance between the H5
     normalized-centroid cloud and the STEP normalized-centroid cloud for the SAME id,
     compared against random other ids as a baseline. Small same-id Chamfer => same
     solid (just a different face split); same-id ~ random => genuinely different part.

Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/regen_probe.py
"""
import os
import numpy as np
import h5py

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus)
from OCC.Extend.TopologyUtils import TopologyExplorer
try:
    from OCC.Core.BRepGProp import brepgprop
    def sprops(f):
        p = GProp_GProps(); brepgprop.SurfaceProperties(f, p); return p
except Exception:
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
    def sprops(f):
        p = GProp_GProps(); brepgprop_SurfaceProperties(f, p); return p

H5 = "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
STEP_DIR = "MFCAD++_dataset/step/train"


def read_with_labels(path):
    """Faithful replica of generator read_step_with_labels + per-face geometry."""
    r = STEPControl_Reader()
    r.ReadFile(path)
    r.TransferRoots()
    shape = r.OneShape()
    treader = r.WS().TransferReader()
    faces = list(TopologyExplorer(shape).faces())
    labels, cents = [], []
    n_named = 0
    for fc in faces:
        item = treader.EntityFromShapeResult(fc, 1)
        name = ""
        if item is not None:
            item = StepRepr_RepresentationItem.DownCast(item)
            if item is not None:
                name = item.Name().ToCString()
        if name:
            n_named += 1
        labels.append(name)
        c = sprops(fc).CentreOfMass()
        cents.append([c.X(), c.Y(), c.Z()])
    return np.array(cents), labels, n_named, len(faces)


def mm(a):
    a = np.asarray(a, float)
    return (a - a.min(0)) / (a.max(0) - a.min(0) + 1e-9)


def chamfer(a, b):
    D = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return 0.5 * (D.min(1).mean() + D.min(0).mean())


def main():
    f = h5py.File(H5, "r")
    b0 = f[list(f.keys())[0]]
    cm = [x.decode() if isinstance(x, bytes) else str(x) for x in b0["CAD_model"][()]]
    idx = np.asarray(b0["idx"]); v1all = np.asarray(b0["V_1"])
    laball = np.asarray(b0["labels"]).reshape(-1); base = int(idx[0, 0])

    sample = cm[:10]
    h5cent = {}; h5lab = {}
    print(f"{'id':>7} {'nH5':>4} {'nSTEP':>5} {'STEP labeled':>13} "
          f"{'STEP label set (sample)':<28} {'H5 label set'}")
    for k in range(10):
        mid = cm[k]
        s = int(idx[k, 0]) - base
        e = int(idx[k + 1, 0]) - base if k + 1 < len(idx) else v1all.shape[0]
        h5cent[mid] = mm(v1all[s:e, 1:4])
        h5lab[mid] = sorted(set(laball[s:e].tolist()))
        path = os.path.join(STEP_DIR, f"{mid}.step")
        cents, slabels, n_named, n_faces = read_with_labels(path)
        slabset = sorted(set(x for x in slabels if x))[:6]
        print(f"{mid:>7} {e-s:>4} {n_faces:>5} {n_named:>6}/{n_faces:<5} "
              f"{str(slabset):<28} {h5lab[mid]}")
        # stash STEP centroids for chamfer
        h5cent[mid + "_step"] = mm(cents)

    print("\nsame-part vs different-part (Chamfer of normalized centroid clouds):")
    ids = sample
    for mid in ids:
        same = chamfer(h5cent[mid], h5cent[mid + "_step"])
        rnd = [chamfer(h5cent[mid], h5cent[o + "_step"]) for o in ids if o != mid]
        print(f"  {mid:>7}: same-id={same:.3f}   random-id median={np.median(rnd):.3f} "
              f"min={np.min(rnd):.3f}   {'SAME-PART?' if same < 0.5*np.median(rnd) else 'not distinguishable'}")


if __name__ == "__main__":
    main()
