"""
regen_subset_B.py  (Step B)

Regenerate a stratified random subset (~800 parts) directly from local STEP files and
validate, before committing to full regeneration:

  (a) CLASS-LEVEL LABEL DISTRIBUTION: regenerated per-face label histogram vs the
      released H5 over the SAME ids. Flag any class present in H5 but never produced
      by regeneration (systematic gap).
  (b) FACE/EDGE COUNT SANITY: distributions, degenerate (0-edge) parts, read failures.
  (c) DIHEDRAL EDGE FEATURE across ALL adjacency edges (not just class-8): angle
      distribution, convex/concave/flat split, NaN/undefined count.

Writes a report to diag/out_B.txt.
Run: /Users/iansun19/miniconda3/envs/mfcadstep/bin/python diag/regen_subset_B.py
"""
import os
import sys
import glob
import random
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brep.taxonomy import NEW_NAMES, NUM_CLASSES

sys.path.insert(0, "diag")
from regen_dihedral_check import (read_part, edge_midpnt_tangent,
                                  normal_on_face_at_point)
from OCC.Core.TopAbs import TopAbs_FORWARD
from OCC.Extend.TopologyUtils import TopologyExplorer

STEP_DIR = "MFCAD++_dataset/step/train"
H5 = "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5"
N = 800
NAMES = [NEW_NAMES[i] for i in range(NUM_CLASSES)]


def regen_edges(shape, faces, fidx, normals):
    """Return list of (i, j, dihedral_deg, convex_sign) for adjacency edges."""
    topo = TopologyExplorer(shape)
    out = []
    for edge in topo.edges():
        ef = list(topo.faces_from_edge(edge))
        if len(ef) != 2:
            continue
        i, j = fidx[ef[0]], fidx[ef[1]]
        mid, tan = edge_midpnt_tangent(edge)
        if mid is None:
            out.append((i, j, np.nan, 0)); continue
        n0 = normal_on_face_at_point(mid, ef[0])
        n1 = normal_on_face_at_point(mid, ef[1])
        if n0 is None or n1 is None:
            out.append((i, j, np.nan, 0)); continue
        cos = np.clip(n0 @ n1 / (np.linalg.norm(n0) * np.linalg.norm(n1)), -1, 1)
        ang = float(np.degrees(np.arccos(cos)))
        if edge.Orientation() == TopAbs_FORWARD:
            s = np.dot(np.cross(n0, n1), tan)
        else:
            s = np.dot(np.cross(n1, n0), tan)
        out.append((i, j, ang, int(np.sign(s))))
    return out


def main():
    log = open("diag/out_B.txt", "w")
    def P(*a):
        print(*a); print(*a, file=log); log.flush()

    random.seed(11)
    files = glob.glob(os.path.join(STEP_DIR, "*.step"))
    random.shuffle(files)

    # released-H5 index for same-id label distribution
    f = h5py.File(H5, "r")
    index = {}
    for bk in f.keys():
        for i, raw in enumerate(f[bk]["CAD_model"][()]):
            pid = raw.decode() if isinstance(raw, bytes) else str(raw)
            index[pid] = (bk, i)

    def h5_labels(pid):
        bk, mi = index[pid]
        b = f[bk]; idx = np.asarray(b["idx"]); lab = np.asarray(b["labels"]).reshape(-1)
        base = int(idx[0, 0]); s = int(idx[mi, 0]) - base
        e = int(idx[mi + 1, 0]) - base if mi + 1 < len(idx) else lab.shape[0]
        return lab[s:e].astype(int)

    regen_hist = np.zeros(NUM_CLASSES, int); h5_hist = np.zeros(NUM_CLASSES, int)
    face_counts = []; edge_counts = []; angles = []; signs = []
    nan_edges = 0; fails = 0; zero_edge = 0; used = 0

    for path in files:
        if used >= N:
            break
        mid = os.path.basename(path)[:-5]
        if mid not in index:
            continue
        try:
            shape, faces, fidx, labels, normals = read_part(path)
        except Exception:
            fails += 1; continue
        ints = [int(x) for x in labels if x != ""]
        if len(ints) != len(faces):
            fails += 1; continue
        used += 1
        for c in ints:
            if 0 <= c < NUM_CLASSES:
                regen_hist[c] += 1
        for c in h5_labels(mid):
            if 0 <= c < NUM_CLASSES:
                h5_hist[c] += 1
        edges = regen_edges(shape, faces, fidx, normals)
        face_counts.append(len(faces)); edge_counts.append(len(edges))
        if len(edges) == 0:
            zero_edge += 1
        for (_, _, ang, s) in edges:
            if np.isnan(ang):
                nan_edges += 1
            else:
                angles.append(ang); signs.append(s)

    P(f"=== Step B: regenerated subset (target {N}) ===")
    P(f"parts used={used}  read/label-mismatch fails={fails}  zero-edge parts={zero_edge}")
    fc = np.array(face_counts); ec = np.array(edge_counts)
    P(f"faces/part: min={fc.min()} median={int(np.median(fc))} max={fc.max()} mean={fc.mean():.1f}")
    P(f"edges/part: min={ec.min()} median={int(np.median(ec))} max={ec.max()} mean={ec.mean():.1f}")

    P("\n(a) class-level label distribution (face-fraction): regen vs released-H5 (same ids)")
    rt = regen_hist.sum(); ht = h5_hist.sum()
    P(f"{'cls':>3} {'name':<22} {'regen%':>7} {'H5%':>7}  flag")
    for c in range(NUM_CLASSES):
        rp = 100 * regen_hist[c] / rt; hp = 100 * h5_hist[c] / ht
        flag = ""
        if h5_hist[c] > 0 and regen_hist[c] == 0:
            flag = "<-- in H5, NEVER in regen (GAP)"
        elif regen_hist[c] > 0 and h5_hist[c] == 0:
            flag = "<-- in regen, not in H5"
        P(f"{c:>3} {NAMES[c]:<22} {rp:6.2f}% {hp:6.2f}%  {flag}")

    a = np.array(angles); s = np.array(signs)
    P(f"\n(c) dihedral edge feature over ALL adjacency edges: n={a.size}  NaN/undefined={nan_edges}")
    P(f"  angle: min={a.min():.1f} median={np.median(a):.1f} max={a.max():.1f}")
    bins = [0, 30, 60, 80, 95, 120, 150, 180]
    h, _ = np.histogram(a, bins=bins)
    for i in range(len(h)):
        P(f"   {bins[i]:3d}-{bins[i+1]:3d} deg : {h[i]:6d} ({100*h[i]/a.size:5.1f}%)")
    P(f"  convex(sign>0)={100*np.mean(s>0):.1f}%  concave(<0)={100*np.mean(s<0):.1f}%  "
      f"flat(0)={100*np.mean(s==0):.1f}%")
    P(f"  near-90 (85-95 deg)={100*np.mean((a>=85)&(a<=95)):.1f}%   "
      f"near-180 (>=175)={100*np.mean(a>=175):.1f}%")
    log.close()


if __name__ == "__main__":
    main()
