"""
scan_nan_full.py — diagnose NaN-loss-from-epoch-1 on the regen pipeline.

Part 1 (Step 1): instrument the REAL train.py construction — is loss NaN on the very
                 first forward pass (pre-backward), or only after a gradient step?
Part 2 (Step 2): full-dataset NaN/Inf scan of the raw H5 (V_1, A_1_values) AND of the
                 dataset.py-built x / edge_attr; pinpoint part_id, column, source, rate.
Part 3 (Step 3): per-column min/max/mean/std for the 14 node + 4 edge feature columns.

Run UNSANDBOXED (torch): .venv_pyg/bin/python diag/scan_nan_full.py
"""
import os
import sys
import numpy as np
import h5py
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy.dataset import get_dataset, MFCADPPRegenGraphDataset, build_node_features_regen
from legacy.model import BRepGNN

SPLITS = {"train.txt": "training_MFCAD++.h5", "val.txt": "val_MFCAD++.h5",
          "test.txt": "test_MFCAD++.h5"}
REGEN = "MFCAD++_dataset/hierarchical_graphs_regen"
NODE_COLS = (["onehot0", "onehot1", "onehot2", "onehot3", "onehot4", "onehot5",
              "area", "cent_x", "cent_y", "cent_z", "nx", "ny", "nz", "plane_d"])
EDGE_COLS = ["concave", "convex", "smooth", "cos_dihedral"]


def sec(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


# ---------------------------------------------------------------------------
def step1_instrument(cfg):
    sec("STEP 1: first-forward vs after-step (real train.py construction)")
    from torch_geometric.loader import DataLoader
    torch.manual_seed(cfg.get("seed", 42))
    ds = get_dataset(cfg, "train.txt")
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=0)
    sample = ds[0]
    node_in, edge_in = sample.x.shape[1], sample.edge_attr.shape[1]
    model = BRepGNN(node_in, edge_in, cfg["hidden_dim"], cfg["num_classes"],
                    cfg["num_layers"], cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    model.train()
    it = iter(loader)

    # --- the very first forward, BEFORE any backward/opt step ---
    b0 = next(it)
    with torch.no_grad():
        logits0 = model(b0.x, b0.edge_index, b0.edge_attr)
        loss0 = torch.nn.functional.cross_entropy(logits0, b0.y)
    print(f"  batch0: x_finite={torch.isfinite(b0.x).all().item()} "
          f"edge_finite={torch.isfinite(b0.edge_attr).all().item()} "
          f"logits_finite={torch.isfinite(logits0).all().item()} "
          f"FIRST_FORWARD_loss={loss0.item():.4f} finite={torch.isfinite(loss0).item()}")
    if not torch.isfinite(loss0).item():
        print("  => NaN on FIRST forward, before any backward => FORWARD/DATA issue (Step 2).")
        return "data"

    # --- now actually step and watch when it goes NaN ---
    verdict = "dynamics"
    for step in range(40):
        batch = b0 if step == 0 else next(it, None)
        if batch is None:
            it = iter(loader); batch = next(it)
        opt.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        pre_back_finite = torch.isfinite(loss).item()
        loss.backward()
        gnorm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()
                                        if p.grad is not None]))
        opt.step()
        if step < 6 or not pre_back_finite:
            print(f"  step {step:02d}: loss={loss.item():.4f} finite_preback={pre_back_finite} "
                  f"grad_norm={gnorm.item():.3e} "
                  f"x_finite={torch.isfinite(batch.x).all().item()}")
        if not pre_back_finite:
            print(f"  => loss became NaN at step {step} (finite earlier) "
                  f"=> GRADIENT-DYNAMICS issue (Step 3).")
            verdict = "dynamics"
            break
    else:
        print("  => 40 steps all finite in this seed/order; NaN may need the specific "
              "bad batch (Step 2 scan will find it).")
    return verdict


# ---------------------------------------------------------------------------
def step2_scan_raw(cfg):
    sec("STEP 2a: full-dataset NaN/Inf scan of RAW H5 (V_1, A_1_values)")
    first = None
    tot_models = bad_models = 0
    bad_cols = {}
    bad_a1 = 0
    for split, fname in SPLITS.items():
        with h5py.File(os.path.join(REGEN, fname), "r") as f:
            for bk in f.keys():
                b = f[bk]
                idx = b["idx"][()]
                V1 = b["V_1"][()]
                AV = b["A_1_values"][()]
                A1 = b["A_1_idx"][()]
                ids = b["CAD_model"][()]
                base = int(idx[0, 0])
                for mi in range(len(idx)):
                    s = int(idx[mi, 0]) - base
                    e = (int(idx[mi + 1, 0]) - base) if mi + 1 < len(idx) else V1.shape[0]
                    pid = ids[mi]
                    pid = pid.decode() if isinstance(pid, bytes) else str(pid)
                    tot_models += 1
                    v1 = V1[s:e]
                    bad = ~np.isfinite(v1)
                    if bad.any():
                        bad_models += 1
                        cols = np.where(bad.any(axis=0))[0]
                        for c in cols:
                            bad_cols[int(c)] = bad_cols.get(int(c), 0) + 1
                        if first is None:
                            first = (split, pid, s, e, cols.tolist(), v1)
                    # A_1_values for this model
                    m = (A1[:, 0] >= s) & (A1[:, 0] < e) & (A1[:, 1] >= s) & (A1[:, 1] < e)
                    if m.any() and not np.isfinite(AV[m]).all():
                        bad_a1 += 1
    V1_COLS = ["area", "cx", "cy", "cz", "type/11", "nx", "ny", "nz", "plane_d"]
    print(f"  models scanned: {tot_models}")
    print(f"  models with non-finite V_1: {bad_models} "
          f"({100*bad_models/max(tot_models,1):.4f}%)")
    print(f"  models with non-finite A_1_values: {bad_a1}")
    if bad_cols:
        print("  non-finite V_1 columns (col -> #models):")
        for c in sorted(bad_cols):
            print(f"    col {c} ({V1_COLS[c]:8s}): {bad_cols[c]}")
    if first:
        split, pid, s, e, cols, v1 = first
        print(f"\n  FIRST bad part: split={split} id={pid} faces={e-s} bad_cols={cols}")
        np.set_printoptions(precision=3, suppress=False, linewidth=140)
        for c in cols:
            col = v1[:, c]
            print(f"    col {c} ({V1_COLS[c]}): n_nonfinite={int((~np.isfinite(col)).sum())}/"
                  f"{len(col)}  sample={col[:min(len(col),12)]}")
    return first


def step2_scan_dataset(cfg):
    sec("STEP 2b: full-dataset NaN/Inf scan via dataset.py (x, edge_attr)")
    first = None
    tot = bad_x = bad_e = 0
    xcol_hist = {}
    for split in SPLITS:
        ds = MFCADPPRegenGraphDataset(cfg["data_root"],
                                      cfg.get("h5_dir", REGEN.split("/")[-1]),
                                      split, cfg["num_surface_types"],
                                      angle_reduce=cfg.get("angle_reduce", "median"))
        for k in range(len(ds)):
            d = ds[k]
            tot += 1
            xb = ~torch.isfinite(d.x)
            eb = ~torch.isfinite(d.edge_attr) if d.edge_attr.numel() else torch.zeros(1, dtype=torch.bool)
            if xb.any():
                bad_x += 1
                cols = torch.where(xb.any(0))[0].tolist()
                for c in cols:
                    xcol_hist[c] = xcol_hist.get(c, 0) + 1
                if first is None:
                    first = (split, ds.ids[k], cols)
            if eb.any():
                bad_e += 1
        ds._close_h5()
    print(f"  parts scanned: {tot}")
    print(f"  parts with non-finite x       : {bad_x} ({100*bad_x/max(tot,1):.4f}%)")
    print(f"  parts with non-finite edge_attr: {bad_e}")
    if xcol_hist:
        print("  non-finite x columns (col -> #parts):")
        for c in sorted(xcol_hist):
            print(f"    col {c:2d} ({NODE_COLS[c]:8s}): {xcol_hist[c]}")
    if first:
        print(f"  FIRST bad part via loader: split={first[0]} id={first[1]} cols={first[2]}")
    return first, bad_x, tot


# ---------------------------------------------------------------------------
def step3_scale(cfg, n_sample=4000):
    sec("STEP 3: per-column scale of node (14) + edge (4) features")
    ds = MFCADPPRegenGraphDataset(cfg["data_root"],
                                  cfg.get("h5_dir", REGEN.split("/")[-1]),
                                  "train.txt", cfg["num_surface_types"],
                                  angle_reduce=cfg.get("angle_reduce", "median"))
    xs, es = [], []
    for k in range(min(n_sample, len(ds))):
        d = ds[k]
        xs.append(d.x.numpy())
        if d.edge_attr.numel():
            es.append(d.edge_attr.numpy())
    ds._close_h5()
    X = np.concatenate(xs, 0)
    E = np.concatenate(es, 0)
    finite_X = X[np.isfinite(X).all(1)]
    print(f"  node features over {X.shape[0]} faces (finite rows={finite_X.shape[0]}):")
    for i, name in enumerate(NODE_COLS):
        col = X[:, i]; fc = col[np.isfinite(col)]
        print(f"    {i:2d} {name:8s} min={fc.min():9.3f} max={fc.max():9.3f} "
              f"mean={fc.mean():8.3f} std={fc.std():8.3f} nonfinite={int((~np.isfinite(col)).sum())}")
    print(f"  edge features over {E.shape[0]} edges:")
    for i, name in enumerate(EDGE_COLS):
        col = E[:, i]; fc = col[np.isfinite(col)]
        print(f"    {i:2d} {name:12s} min={fc.min():9.3f} max={fc.max():9.3f} "
              f"mean={fc.mean():8.3f} std={fc.std():8.3f} nonfinite={int((~np.isfinite(col)).sum())}")


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"config: h5_format={cfg.get('h5_format')} dir={cfg.get('h5_dir')} "
          f"lr={cfg.get('lr')} bs={cfg.get('batch_size')}")
    step2_scan_raw(cfg)
    step2_scan_dataset(cfg)
    step3_scale(cfg)
    step1_instrument(cfg)


if __name__ == "__main__":
    main()
