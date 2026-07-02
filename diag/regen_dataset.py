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
  V_1       [N,11] [area,cx,cy,cz] per-model min-max -> [0,1]; type/11; nx;ny;nz;plane_d;
                   mean_curv; gauss_curv  (cols 9-10 added for curvature node features)
  labels    [N]    float32 per-face class (0-11)
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step_ingest import StepIngestError, build_V1, extract_brep_from_step

OUT_DIR = "MFCAD++_dataset/hierarchical_graphs_regen"
OUT_DIR_12 = "MFCAD++_dataset/hierarchical_graphs_regen_12"
STEP_DIR = "MFCAD++_dataset/step"
STEP_DIR_12 = "MFCAD++_dataset/step_12class"
SPLIT_OUT = {"train": "training_MFCAD++.h5", "val": "val_MFCAD++.h5",
             "test": "test_MFCAD++.h5"}


def read_model(path, *, require_12class=False):
    """Return per-face arrays + edge lists for one STEP part, or None on failure."""
    try:
        m, _stats = extract_brep_from_step(
            path, require_labels=True, require_12class=require_12class)
    except StepIngestError:
        return None
    except Exception:
        return None
    if m is None:
        return None
    if (m["labels"] < 0).any():
        return None
    return m


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
    ap.add_argument(
        "--twelve-class",
        dest="twelve_class",
        action="store_true",
        help="Read step_12class/ (0–11 labels) and write hierarchical_graphs_regen_12/",
    )
    args = ap.parse_args()

    step_root = STEP_DIR_12 if args.twelve_class else STEP_DIR
    step_dir = f"{step_root}/{args.split}"
    out_dir = OUT_DIR_12 if args.twelve_class else OUT_DIR
    split_txt = f"MFCAD++_dataset/{args.split}.txt"
    with open(split_txt) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        ids = ids[:args.limit]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, SPLIT_OUT[args.split])
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
            m = read_model(p, require_12class=args.twelve_class)
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
