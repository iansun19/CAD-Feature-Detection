"""
regen_dataset.py  (Step A — generation only; does NOT touch dataset.py/model.py/train)

Regenerate the MFCAD++ B-rep graph dataset directly from the local STEP files, so that
node<->face<->normal<->label<->dihedral correspondence is exact by construction.

Output mirrors the released hierarchical H5 B-rep schema so the existing loader reads
it drop-in, PLUS two additions for the dihedral edge-feature work:
  * A_1_values  = true dihedral angle (radians) per adjacency edge (released = const 1.0)
  * V_1 cols 5-8 = exact per-face unit normal (nx,ny,nz) + plane-d (n . centroid)

Per batch group (mirrors released):
  CAD_model [m]    bytes ids
  idx       [m,2]  col0 = cumulative face start (base 0); col1 unused (0)
  V_1       [N,9]  [area,cx,cy,cz] per-model min-max -> [0,1]; type/11; nx;ny;nz;plane_d
  labels    [N]    float32 per-face class (0-24)
  A_1_idx   [E,2]  int32 global face-index pairs (BOTH directions)
  A_1_values[E]    float32 dihedral radians (both directions equal)
  E_1_idx/E_2_idx/E_3_idx  int32 global pairs: convex / concave / smooth(+seam self-loops)
  A_3_idx   [0,2]  empty (mesh pooling disabled; node normals come from V_1 now)
  A_1_shape [2]    [N,N]

Usage:
  python diag/regen_dataset.py --split val   [--bs 256] [--limit N]
"""
import os
import sys
import argparse
import numpy as np
import h5py

sys.path.insert(0, "diag")
from regen_dihedral_check import edge_midpnt_tangent, normal_on_face_at_point

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus)
from OCC.Core.TopAbs import TopAbs_REVERSED, TopAbs_FORWARD
from OCC.Core.BRepTools import breptools
from OCC.Extend.TopologyUtils import TopologyExplorer
try:
    from OCC.Core.BRepGProp import brepgprop
    def sprops(f):
        p = GProp_GProps(); brepgprop.SurfaceProperties(f, p); return p
except Exception:
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
    def sprops(f):
        p = GProp_GProps(); brepgprop_SurfaceProperties(f, p); return p

OCC2CODE = {GeomAbs_Plane: 1, GeomAbs_Cylinder: 2, GeomAbs_Torus: 3,
            GeomAbs_Sphere: 4, GeomAbs_Cone: 5}
OUT_DIR = "MFCAD++_dataset/hierarchical_graphs_regen"
SPLIT_OUT = {"train": "training_MFCAD++.h5", "val": "val_MFCAD++.h5",
             "test": "test_MFCAD++.h5"}


def face_mid_normal(face):
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    surf = BRepAdaptor_Surface(face, True)
    p = BRepLProp_SLProps(surf, 0.5 * (umin + umax), 0.5 * (vmin + vmax), 1, 1e-6)
    if not p.IsNormalDefined():
        return np.array([0.0, 0.0, 0.0])
    d = p.Normal(); n = np.array([d.X(), d.Y(), d.Z()])
    if face.Orientation() == TopAbs_REVERSED:
        n = -n
    nn = np.linalg.norm(n)
    return n / nn if nn > 1e-9 else n


def read_model(path):
    """Return per-face arrays + edge lists for one STEP part, or None on failure."""
    r = STEPControl_Reader()
    r.ReadFile(path)
    r.TransferRoots()
    shape = r.OneShape()
    treader = r.WS().TransferReader()
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    fidx = {f: i for i, f in enumerate(faces)}
    N = len(faces)
    area = np.zeros(N); cent = np.zeros((N, 3)); tcode = np.zeros(N)
    normals = np.zeros((N, 3)); labels = np.full(N, -1, int)
    for f, i in fidx.items():
        gp = sprops(f); c = gp.CentreOfMass()
        area[i] = gp.Mass(); cent[i] = [c.X(), c.Y(), c.Z()]
        tcode[i] = OCC2CODE.get(BRepAdaptor_Surface(f, True).GetType(), 11)
        normals[i] = face_mid_normal(f)
        item = treader.EntityFromShapeResult(f, 1)
        name = ""
        if item is not None:
            item = StepRepr_RepresentationItem.DownCast(item)
            if item is not None:
                name = item.Name().ToCString()
        if name == "" or not name.lstrip("-").isdigit():
            return None                      # missing/garbage label -> drop model
        labels[i] = int(name)
    if (labels < 0).any():
        return None

    # edges -> adjacency (both dirs), convexity bucket, dihedral radians
    A = []; Aval = []; E1 = []; E2 = []; E3 = []
    for edge in topo.edges():
        ef = list(topo.faces_from_edge(edge))
        if len(ef) == 1:                     # seam/boundary -> self-loop in E_3
            i = fidx[ef[0]]; E3.append((i, i)); continue
        if len(ef) != 2:
            continue
        i, j = fidx[ef[0]], fidx[ef[1]]
        if i == j:
            E3.append((i, i)); continue
        mid, tan = edge_midpnt_tangent(edge)
        ang = np.pi; sgn = 0
        if mid is not None:
            n0 = normal_on_face_at_point(mid, ef[0])
            n1 = normal_on_face_at_point(mid, ef[1])
            if n0 is not None and n1 is not None:
                cos = np.clip(n0 @ n1 / (np.linalg.norm(n0) * np.linalg.norm(n1)), -1, 1)
                ang = float(np.arccos(cos))
                r = (np.dot(np.cross(n0, n1), tan) if edge.Orientation() == TopAbs_FORWARD
                     else np.dot(np.cross(n1, n0), tan))
                sgn = int(np.sign(r))
        A.append((i, j)); Aval.append(ang)
        A.append((j, i)); Aval.append(ang)
        bucket = E1 if sgn == 1 else (E2 if sgn == -1 else E3)
        bucket.append((i, j)); bucket.append((j, i))

    plane_d = np.sum(normals * cent, axis=1)         # n . centroid (signed offset)
    return dict(N=N, area=area, cent=cent, tcode=tcode, normals=normals,
                plane_d=plane_d, labels=labels,
                A=np.array(A, np.int64).reshape(-1, 2), Aval=np.array(Aval, np.float32),
                E1=np.array(E1, np.int64).reshape(-1, 2),
                E2=np.array(E2, np.int64).reshape(-1, 2),
                E3=np.array(E3, np.int64).reshape(-1, 2))


def mm01(a):
    a = np.asarray(a, float); lo = a.min(0); hi = a.max(0)
    return (a - lo) / (hi - lo + 1e-9)


def build_V1(m):
    cc = mm01(np.column_stack([m["area"], m["cent"]]))     # [N,4] -> [0,1]
    return np.column_stack([cc, m["tcode"] / 11.0, m["normals"], m["plane_d"]]).astype(np.float32)


def write_group(g, models):
    """models: list of (id, dict). Concatenate with cumulative global offsets."""
    offs = []; cur = 0
    V1 = []; LAB = []; A = []; AV = []; E1 = []; E2 = []; E3 = []; ids = []; idx = []
    for mid, m in models:
        offs.append(cur); idx.append((cur, 0))
        V1.append(build_V1(m)); LAB.append(m["labels"].astype(np.float32))
        for arr, dst in [(m["A"], A), (m["E1"], E1), (m["E2"], E2), (m["E3"], E3)]:
            if arr.size:
                dst.append(arr + cur)
        if m["Aval"].size:
            AV.append(m["Aval"])
        ids.append(mid); cur += m["N"]
    N = cur
    V1 = np.concatenate(V1); LAB = np.concatenate(LAB)

    def cat(lst):
        return (np.concatenate(lst).astype(np.int32) if lst
                else np.zeros((0, 2), np.int32))
    g.create_dataset("V_1", data=V1, compression="lzf")
    g.create_dataset("labels", data=LAB, compression="lzf")
    g.create_dataset("idx", data=np.array(idx, np.int32))
    g.create_dataset("CAD_model", data=np.array(ids, dtype=h5py.string_dtype()))
    g.create_dataset("A_1_idx", data=cat(A), compression="lzf")
    g.create_dataset("A_1_values",
                     data=(np.concatenate(AV).astype(np.float32) if AV
                           else np.zeros((0,), np.float32)), compression="lzf")
    g.create_dataset("E_1_idx", data=cat(E1), compression="lzf")
    g.create_dataset("E_2_idx", data=cat(E2), compression="lzf")
    g.create_dataset("E_3_idx", data=cat(E3), compression="lzf")
    g.create_dataset("A_3_idx", data=np.zeros((0, 2), np.int32))
    g.create_dataset("A_1_shape", data=np.array([N, N], np.int32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    step_dir = f"MFCAD++_dataset/step/{args.split}"
    split_txt = f"MFCAD++_dataset/{args.split}.txt"
    with open(split_txt) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        ids = ids[:args.limit]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, SPLIT_OUT[args.split])
    fout = h5py.File(out_path, "w")

    batch = []; gi = 0; done = 0; fails = 0; faces_tot = 0
    def flush():
        nonlocal batch, gi
        if batch:
            write_group(fout.create_group(f"batch_{gi:05d}"), batch)
            gi += 1; batch = []

    for n, mid in enumerate(ids):
        p = os.path.join(step_dir, f"{mid}.step")
        if not os.path.isfile(p):
            fails += 1; continue
        try:
            m = read_model(p)
        except Exception:
            m = None
        if m is None:
            fails += 1; continue
        batch.append((mid, m)); done += 1; faces_tot += m["N"]
        if len(batch) >= args.bs:
            flush()
        if n % 2000 == 0:
            print(f"  {args.split}: {n}/{len(ids)} done={done} fails={fails}", flush=True)
    flush()
    fout.close()
    print(f"[{args.split}] wrote {out_path}: models={done} fails={fails} "
          f"faces={faces_tot} groups={gi}", flush=True)


if __name__ == "__main__":
    main()
