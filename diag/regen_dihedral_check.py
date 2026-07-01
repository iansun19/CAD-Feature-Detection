"""
regen_dihedral_check.py — DEFINITIVE salvageability test.

Regenerate graphs directly from LOCAL .step files (faces + per-face labels + true
normals + adjacency + convexity sign), exactly as the documented generator does, then
run the concave same-label through_step wall-floor ~90-degree check on legacy STEP
label "8" (rectangular through step; old id 8 -> new class 3 through_step). Here
face<->label<->normal correspondence is exact BY CONSTRUCTION (one STEP read), so
there is no matching step and no normalization artifact.

GO  : median ~90, low std, majority in 80-95 bucket.
Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/regen_dihedral_check.py
"""
import os
import glob
import random
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopAbs import TopAbs_REVERSED, TopAbs_FORWARD
from OCC.Core.TopoDS import topods
from OCC.Core.BRepTools import breptools
from OCC.Extend.TopologyUtils import TopologyExplorer

STEP_DIR = "MFCAD++_dataset/step/train"
# Legacy STEP label for rectangular through step (old id 8 -> new class 3 through_step).
OLD_RECT_THROUGH_STEP = "8"


def normal_at_uv(face, u, v):
    surf = BRepAdaptor_Surface(face, True)
    p = BRepLProp_SLProps(surf, u, v, 1, 1e-6)
    if not p.IsNormalDefined():
        return None
    d = p.Normal()
    n = np.array([d.X(), d.Y(), d.Z()])
    if face.Orientation() == TopAbs_REVERSED:
        n = -n
    return n


def face_mid_normal(face):
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    return normal_at_uv(face, 0.5 * (umin + umax), 0.5 * (vmin + vmax))


def edge_midpnt_tangent(edge):
    res = BRep_Tool.Curve(edge)
    if res is None or len(res) < 3:
        return None, None
    curve, a, b = res[0], res[1], res[2]
    from OCC.Core.gp import gp_Vec
    p = gp_Pnt(0, 0, 0); v = gp_Vec(0, 0, 0)
    curve.D1(0.5 * (a + b), p, v)
    return np.array(p.Coord()), np.array(v.Coord())


def normal_on_face_at_point(xyz, face):
    surface = BRep_Tool.Surface(face)
    sas = ShapeAnalysis_Surface(surface)
    uv = sas.ValueOfUV(gp_Pnt(float(xyz[0]), float(xyz[1]), float(xyz[2])), 0.01)
    props = GeomLProp_SLProps(surface, uv.X(), uv.Y(), 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    d = props.Normal()
    n = np.array([d.X(), d.Y(), d.Z()])
    if face.Orientation() == TopAbs_REVERSED:
        n = -n
    return n


def convexity_sign(edge, faces):
    """Generator edge_dihedral: sign of (n0 x n1).tangent (orientation-aware)."""
    mid, tan = edge_midpnt_tangent(edge)
    if mid is None:
        return 0
    n0 = normal_on_face_at_point(mid, faces[0])
    n1 = normal_on_face_at_point(mid, faces[1])
    if n0 is None or n1 is None:
        return 0
    if edge.Orientation() == TopAbs_FORWARD:
        r = np.dot(np.cross(n0, n1), tan)
    else:
        r = np.dot(np.cross(n1, n0), tan)
    return float(np.sign(r))


def read_part(path):
    r = STEPControl_Reader()
    r.ReadFile(path)
    r.TransferRoots()
    shape = r.OneShape()
    treader = r.WS().TransferReader()
    faces = list(TopologyExplorer(shape).faces())
    fidx = {}
    labels, normals = [], []
    for i, fc in enumerate(faces):
        fidx[fc] = i
        item = treader.EntityFromShapeResult(fc, 1)
        name = ""
        if item is not None:
            item = StepRepr_RepresentationItem.DownCast(item)
            if item is not None:
                name = item.Name().ToCString()
        labels.append(name)
        normals.append(face_mid_normal(fc))
    return shape, faces, fidx, labels, normals


def main():
    random.seed(3)
    files = glob.glob(os.path.join(STEP_DIR, "*.step"))
    random.shuffle(files)

    angles = []
    parts_used = 0
    scanned = 0
    for path in files:
        if parts_used >= 25:
            break
        scanned += 1
        try:
            shape, faces, fidx, labels, normals = read_part(path)
        except Exception:
            continue
        if OLD_RECT_THROUGH_STEP not in labels:
            continue
        topo = TopologyExplorer(shape)
        part_pairs = 0
        for edge in topo.edges():
            efaces = list(topo.faces_from_edge(edge))
            if len(efaces) != 2:
                continue
            i, j = fidx[efaces[0]], fidx[efaces[1]]
            if labels[i] != OLD_RECT_THROUGH_STEP or labels[j] != OLD_RECT_THROUGH_STEP:
                continue
            s = convexity_sign(edge, efaces)
            if s >= 0:                      # concave only (match prior convention)
                continue
            ni, nj = normals[i], normals[j]
            if ni is None or nj is None:
                continue
            cos = np.clip(ni @ nj / (np.linalg.norm(ni) * np.linalg.norm(nj)), -1, 1)
            angles.append(float(np.degrees(np.arccos(cos))))
            part_pairs += 1
        if part_pairs > 0:
            parts_used += 1

    a = np.array(angles)
    print(f"scanned {scanned} STEP files; used {parts_used} parts with concave "
          f"through_step (legacy label {OLD_RECT_THROUGH_STEP}) pairs")
    print(f"concave same-label through_step pairs: n={a.size}")
    if a.size:
        print(f"  mean={a.mean():.1f}  median={np.median(a):.1f}  std={a.std():.1f}")
        bins = [0, 30, 60, 80, 95, 180]
        h, _ = np.histogram(a, bins=bins)
        for i in range(len(h)):
            print(f"   {bins[i]:3d}-{bins[i+1]:3d} deg : {h[i]:4d} ({100*h[i]/a.size:5.1f}%)")
        print(f"\n  GO if median ~90 and 80-95 bucket dominant.")


if __name__ == "__main__":
    main()
