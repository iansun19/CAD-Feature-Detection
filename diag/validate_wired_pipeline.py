"""
validate_wired_pipeline.py — Step 3 validation of the wired regen pipeline.

Runs THROUGH the real dataset.py loading path (get_dataset / MFCADPPRegenGraphDataset),
not the standalone investigation scripts, so it catches any wiring/indexing bug between
the validated-in-isolation H5 and what the model will actually train on.

Checks:
  1. one batch per split: node dim, edge dim, NaN/Inf, BRepGNN forward -> valid logits
  2. node_in / edge_in auto-derive from tensor shapes
  3. concave class-8 (rect through step) ~90 deg, reconstructed from the loader's
     OWN edge_attr (cos(dihedral)), per split -> compare to standalone result
  4. median-vs-mean dedup reduction: full-scale comparison of the per-pair angle
     representative; flags whether the choice meaningfully shifts the edge-feature
     distribution, broken down by curved faces / curved-adjacent classes.

Run (must be UNSANDBOXED for torch):
  .venv_pyg/bin/python diag/validate_wired_pipeline.py
"""
import os
import sys
import numpy as np
import h5py
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy.dataset import get_dataset, MFCADPPRegenGraphDataset
from legacy.model import BRepGNN
from torch_geometric.loader import DataLoader

SPLITS = {"train.txt": "training_MFCAD++.h5", "val.txt": "val_MFCAD++.h5",
          "test.txt": "test_MFCAD++.h5"}
REGEN = "MFCAD++_dataset/hierarchical_graphs_regen"
CURVED_CODES = {2, 3, 4, 5}          # cyl, torus, sphere, cone (plane=1, other=11)


def canon(u, v):
    return (u, v) if u < v else (v, u)


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# ---------------------------------------------------------------------------
def check_batches_and_forward(cfg):
    section("1-2. batch dims / NaN-Inf / forward pass (real dataset.py path)")
    results = {}
    for split in SPLITS:
        ds = get_dataset(cfg, split)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        node_in, edge_in = batch.x.shape[1], batch.edge_attr.shape[1]
        x_ok = torch.isfinite(batch.x).all().item()
        e_ok = torch.isfinite(batch.edge_attr).all().item()
        # cos(dihedral) must be in [-1,1]
        cos_col = batch.edge_attr[:, 3]
        cos_ok = bool((cos_col >= -1.0001).all() and (cos_col <= 1.0001).all())
        # onehot rows sum to 1
        oh_ok = bool(torch.allclose(batch.edge_attr[:, :3].sum(1),
                                    torch.ones(batch.edge_attr.shape[0]))) \
            if batch.edge_attr.shape[0] else True

        model = BRepGNN(node_in, edge_in, cfg["hidden_dim"], cfg["num_classes"],
                        cfg["num_layers"], cfg["dropout"])
        model.eval()
        with torch.no_grad():
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
        logits_ok = (logits.shape == (batch.x.shape[0], cfg["num_classes"]) and
                     torch.isfinite(logits).all().item())
        ds._close_h5()
        results[split] = (node_in, edge_in)
        print(f"  [{split:9}] nodes={batch.x.shape[0]:5d} edges={batch.edge_attr.shape[0]:6d} "
              f"node_in={node_in} edge_in={edge_in} "
              f"x_finite={x_ok} e_finite={e_ok} cos_in[-1,1]={cos_ok} onehot_ok={oh_ok} "
              f"logits={tuple(logits.shape)} logits_finite={logits_ok}")
    dims = set(results.values())
    print(f"\n  node_in/edge_in consistent across splits: {dims} "
          f"-> {'OK (auto-derived)' if dims == {(14, 4)} else 'MISMATCH!'}")


# ---------------------------------------------------------------------------
def class8_via_loader(cfg):
    section("3. concave class-8 ~90 deg, reconstructed from loader edge_attr")
    print("(angle recovered as degrees(arccos(edge_attr[:,3])); concave = onehot col0)")
    for split in SPLITS:
        ds = MFCADPPRegenGraphDataset(cfg["data_root"],
                                      cfg.get("h5_dir", "hierarchical_graphs_regen"),
                                      split, cfg["num_surface_types"],
                                      angle_reduce=cfg.get("angle_reduce", "median"))
        degs = []
        for k in range(len(ds)):
            d = ds[k]
            if d.edge_index.numel() == 0:
                continue
            y = d.y.numpy()
            ei = d.edge_index.numpy()
            ea = d.edge_attr.numpy()
            concave = ea[:, 0] > 0.5
            u, v = ei[0], ei[1]
            both8 = (y[u] == 8) & (y[v] == 8)
            sel = concave & both8
            if sel.any():
                cosv = np.clip(ea[sel, 3], -1.0, 1.0)
                degs.extend(np.degrees(np.arccos(cosv)).tolist())
        ds._close_h5()
        a = np.array(degs)
        if a.size:
            in8095 = 100 * np.mean((a >= 80) & (a <= 95))
            print(f"  [{split:9}] n={a.size:6d} (directed) median={np.median(a):.1f} "
                  f"mean={a.mean():.1f} std={a.std():.1f} in80-95={in8095:.1f}%")
        else:
            print(f"  [{split:9}] no concave class-8 edges found")


# ---------------------------------------------------------------------------
def median_vs_mean(cfg):
    section("4. dedup reduction: median (loader default) vs mean — full-scale")
    print(f"  loader angle_reduce = {cfg.get('angle_reduce', 'median')!r}")
    all_diff = []
    shift_curved = shift_planar = 0
    label_hist = {}
    n_multi = n_edges = 0
    THR = 0.05  # cos shift considered "meaningful"
    for split, fname in SPLITS.items():
        with h5py.File(os.path.join(REGEN, fname), "r") as f:
            for bk in f.keys():
                b = f[bk]
                idx = b["idx"][()]
                V1 = b["V_1"][()]
                lab = b["labels"][()].astype(int)
                A1 = b["A_1_idx"][()]
                AV = b["A_1_values"][()]
                base = int(idx[0, 0])
                codes = np.round(V1[:, 4] * 11).astype(int)
                for mi in range(len(idx)):
                    s = int(idx[mi, 0]) - base
                    e = (int(idx[mi + 1, 0]) - base) if mi + 1 < len(idx) else V1.shape[0]
                    m = (A1[:, 0] >= s) & (A1[:, 0] < e) & (A1[:, 1] >= s) & (A1[:, 1] < e)
                    rows = A1[m] - s
                    av = AV[m]
                    per = {}
                    for (u, v), ang in zip(rows.tolist(), av.tolist()):
                        if u == v:
                            continue
                        per.setdefault(canon(u, v), []).append(float(ang))
                    for (u, v), angs in per.items():
                        n_edges += 1
                        if len(angs) <= 2:   # singleton pair (2 = one edge, both dirs)
                            continue
                        n_multi += 1
                        cmed = np.cos(np.median(angs))
                        cmean = np.cos(np.mean(angs))
                        diff = abs(cmed - cmean)
                        all_diff.append(diff)
                        if diff > THR:
                            gu, gv = u + s, v + s
                            if codes[gu] in CURVED_CODES or codes[gv] in CURVED_CODES:
                                shift_curved += 1
                            else:
                                shift_planar += 1
                            for lb in (lab[gu], lab[gv]):
                                label_hist[lb] = label_hist.get(lb, 0) + 1
    all_diff = np.array(all_diff)
    print(f"  unique face pairs (canonical): {n_edges}")
    print(f"  multi-edge pairs (median!=mean possible): {n_multi}")
    if all_diff.size:
        print(f"  |cos_median - cos_mean| over multi-edge pairs: "
              f"max={all_diff.max():.4f} mean={all_diff.mean():.5f} "
              f"median={np.median(all_diff):.5f}")
        for t in (0.001, 0.01, 0.05, 0.1):
            n = int((all_diff > t).sum())
            print(f"    pairs shifting > {t:<5}: {n} "
                  f"({100*n/max(n_edges,1):.4f}% of ALL pairs, "
                  f"{100*n/max(n_multi,1):.3f}% of multi-edge)")
    print(f"\n  pairs shifting > {THR} that touch a CURVED face : {shift_curved}")
    print(f"  pairs shifting > {THR} that are purely PLANAR    : {shift_planar}")
    if label_hist:
        names = load_names(cfg)
        top = sorted(label_hist.items(), key=lambda kv: -kv[1])[:10]
        print(f"  endpoint label classes among >{THR}-shift pairs (top 10):")
        for lb, ct in top:
            print(f"    class {lb:2d} {names.get(lb,'?'):24s}: {ct}")
    tot_shift = shift_curved + shift_planar
    print(f"\n  VERDICT: {tot_shift} pairs ({100*tot_shift/max(n_edges,1):.4f}% of all "
          f"edges) shift cos by >{THR} between median and mean.")


def load_names(cfg):
    path = os.path.join(cfg["data_root"], "feature_labels.txt")
    names = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split(" - ", 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    names[int(parts[0].strip())] = parts[1].strip()
    return names


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"config: h5_format={cfg.get('h5_format')} h5_dir={cfg.get('h5_dir')} "
          f"angle_reduce={cfg.get('angle_reduce')}")
    check_batches_and_forward(cfg)
    class8_via_loader(cfg)
    median_vs_mean(cfg)


if __name__ == "__main__":
    main()
