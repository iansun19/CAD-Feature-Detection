"""
step_normal_probe.py — validate TRUE STEP normals on the concave class-8 (90 deg) check.

Reads diag/step_candidates.json (from find_step_candidates.py), and for each model:
  1. parses the STEP B-rep with pythonocc, enumerating faces (TopologyExplorer dedupes).
  2. per face: surface type, area, 3D centroid (GProp), and the OUTWARD normal sampled
     at the UV midpoint (orientation-corrected via TopAbs_REVERSED).
  3. matches STEP faces <-> H5 nodes by per-model MIN-MAX normalized [area,cx,cy,cz]
     (V_1 in the H5 is per-model min-max normalized; we replicate that on the STEP
     side). Reports match quality (nearest-vs-2nd-nearest margin, centroid distance).
  4. for each concave class-8 H5 face pair, computes the angle between the two matched
     STEP-face normals.
  5. cylinder spot-check: angular spread of normals sampled across a curved face.

Run in the pythonocc env:
    /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/step_normal_probe.py
"""
import json
import os
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus)
from OCC.Core.BRepTools import breptools
from OCC.Extend.TopologyUtils import TopologyExplorer

try:
    from OCC.Core.BRepGProp import brepgprop
    def surf_props(face):
        p = GProp_GProps(); brepgprop.SurfaceProperties(face, p); return p
except Exception:
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
    def surf_props(face):
        p = GProp_GProps(); brepgprop_SurfaceProperties(face, p); return p

TYPE_NAME = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder",
             GeomAbs_Cone: "cone", GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus"}


def read_faces(step_path):
    r = STEPControl_Reader()
    if r.ReadFile(step_path) != IFSelect_RetDone:
        return None
    r.TransferRoots()
    shape = r.OneShape()
    faces = list(TopologyExplorer(shape).faces())
    out = []
    for fc in faces:
        surf = BRepAdaptor_Surface(fc, True)
        umin, umax, vmin, vmax = breptools.UVBounds(fc)
        u = 0.5 * (umin + umax); v = 0.5 * (vmin + vmax)
        props = BRepLProp_SLProps(surf, u, v, 1, 1e-6)
        n = np.array([0.0, 0.0, 0.0])
        if props.IsNormalDefined():
            d = props.Normal()
            n = np.array([d.X(), d.Y(), d.Z()])
            if fc.Orientation() == TopAbs_REVERSED:
                n = -n
        gp = surf_props(fc)
        c = gp.CentreOfMass()
        out.append({
            "type": surf.GetType(),
            "area": gp.Mass(),
            "centroid": np.array([c.X(), c.Y(), c.Z()]),
            "normal": n,
            "face": fc,
            "uvb": (umin, umax, vmin, vmax),
        })
    return out, shape


def minmax(a):
    a = np.asarray(a, float)
    lo = a.min(0); hi = a.max(0)
    return (a - lo) / (hi - lo + 1e-6)


def match_h5_to_step(h5_feat_norm, step_feat_norm):
    """For each H5 node return (best_step_idx, best_dist, margin)."""
    res = []
    for hf in h5_feat_norm:
        d = np.linalg.norm(step_feat_norm - hf, axis=1)
        order = np.argsort(d)
        best = int(order[0])
        margin = float(d[order[1]] - d[order[0]]) if len(order) > 1 else 9.9
        res.append((best, float(d[order[0]]), margin))
    return res


def cyl_spread(face, uvb):
    umin, umax, vmin, vmax = uvb
    surf = BRepAdaptor_Surface(face, True)
    ns = []
    for uu in np.linspace(umin, umax, 5)[1:-1] if umax > umin else [0.5*(umin+umax)]:
        for vv in np.linspace(vmin, vmax, 5)[1:-1] if vmax > vmin else [0.5*(vmin+vmax)]:
            p = BRepLProp_SLProps(surf, float(uu), float(vv), 1, 1e-6)
            if p.IsNormalDefined():
                dd = p.Normal(); ns.append([dd.X(), dd.Y(), dd.Z()])
    ns = np.array(ns)
    if len(ns) < 2:
        return 0.0
    ref = ns[0]
    cos = np.clip(ns @ ref, -1, 1)
    return float(np.degrees(np.arccos(cos)).max())


def main():
    with open("diag/step_candidates.json") as f:
        cands = json.load(f)

    all_angles = []            # all pairs
    conf_angles = []           # pairs where BOTH faces matched confidently
    cyl_spreads = []
    DIST_OK = 0.02             # normalized-centroid distance threshold for a trusted match
    print(f"{'model':>7} {'nSTEP':>5} {'nH5':>4} {'conf':>5} {'medDist':>8} "
          f"{'pairs':>5}  pair_angles(deg)  [* = low-confidence match]")
    for c in cands:
        mid = c["model_id"]
        step_path = os.path.join("MFCAD++_dataset", "step", c["split"], f"{mid}.step")
        parsed = read_faces(step_path)
        if parsed is None:
            print(f"  {mid}: STEP read FAILED"); continue
        sfaces, shape = parsed
        v1 = np.array(c["v1"])                       # [N,5] H5 (normalized) area,cx,cy,cz,typecode
        n_h5 = v1.shape[0]; n_step = len(sfaces)

        # features for matching: [area, cx, cy, cz]
        h5_feat = minmax(v1[:, :4])                  # re-min-max (already ~[0,1]); robust
        step_raw = np.array([[s["area"], *s["centroid"]] for s in sfaces])
        step_feat = minmax(step_raw)

        match = match_h5_to_step(h5_feat, step_feat)
        best_idx = [m[0] for m in match]
        dists = np.array([m[1] for m in match])
        conf_cnt = int(np.sum(dists < DIST_OK))

        # pair angles via matched STEP normals
        pa = []
        for (i, j) in c["concave_class8_pairs"]:
            si, sj = best_idx[i], best_idx[j]
            ni, nj = sfaces[si]["normal"], sfaces[sj]["normal"]
            if np.linalg.norm(ni) < 1e-6 or np.linalg.norm(nj) < 1e-6:
                continue
            cos = np.clip(ni @ nj / (np.linalg.norm(ni)*np.linalg.norm(nj)), -1, 1)
            ang = float(np.degrees(np.arccos(cos)))
            confident = dists[i] < DIST_OK and dists[j] < DIST_OK
            all_angles.append(ang)
            if confident:
                conf_angles.append(ang)
            pa.append(f"{ang:.0f}{'' if confident else '*'}")
        print(f"{mid:>7} {n_step:>5} {n_h5:>4} {conf_cnt:>3}/{n_h5:<2} "
              f"{np.median(dists):>8.3f} {len(pa):>5}  {pa}")

        for s in sfaces:
            if s["type"] in (GeomAbs_Cylinder, GeomAbs_Cone):
                cyl_spreads.append(cyl_spread(s["face"], s["uvb"]))

    def report(a, title):
        a = np.array(a)
        print(f"\n=== {title} ===")
        if not a.size:
            print("  (none)"); return
        print(f"  n={a.size}  mean={a.mean():.1f}  median={np.median(a):.1f}  std={a.std():.1f}")
        bins = [0, 30, 60, 80, 95, 180]
        hist, _ = np.histogram(a, bins=bins)
        for i in range(len(hist)):
            print(f"   {bins[i]:3d}-{bins[i+1]:3d} deg : {hist[i]:4d} ({100*hist[i]/a.size:5.1f}%)")

    report(all_angles, "ALL concave class-8 pairs (incl. low-confidence matches)")
    report(conf_angles, "CONFIDENT matches only (both faces centroid-dist < 0.02)")
    cs = np.array(cyl_spreads)
    print(f"\n=== cylinder/cone normal spread spot-check (deg across face) ===")
    if cs.size:
        print(f"  n={cs.size} faces  median spread={np.median(cs):.1f}  "
              f"max={cs.max():.1f}  (expect >> 0 for true curved normals)")
    else:
        print("  no curved faces among these parts")


if __name__ == "__main__":
    main()
