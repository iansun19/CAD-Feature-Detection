"""
sanity_mesh_feats.py — validate the pooled mesh node features BEFORE training.

Loads one real batch through the PyG DataLoader and checks:
  1. node feature width == EXPECTED_DIM (10 base + 1 pooled plane-d = 11)
  2. no NaN / Inf anywhere in x
  3. the pooled plane-d channel (last col) is standardized (mean ~0, std ~1, finite)
  4. one forward pass through BRepGNN succeeds at the new input width (model
     auto-sizes node_in from x.shape[1]; this proves nothing is hardcoded)

Does NOT train. Run the actual training run yourself once this looks good.

    python diag/sanity_mesh_feats.py
"""
import os
import sys

import torch
import yaml
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import get_dataset
from model import BRepGNN

EXPECTED_DIM = 11          # 6 surface-type one-hot + 1 area + 3 centroid + 1 plane-d


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ds = get_dataset(cfg, "train.txt")
    n = min(cfg.get("batch_size", 64), len(ds))
    subset = [ds[i] for i in range(n)]
    batch = next(iter(DataLoader(subset, batch_size=n, shuffle=False)))

    x = batch.x
    print(f"loaded batch: {batch.num_graphs} graphs, {x.shape[0]} nodes, "
          f"feature dim = {x.shape[1]}")

    ok = True

    # 1. width
    if x.shape[1] != EXPECTED_DIM:
        print(f"  FAIL: feature dim {x.shape[1]} != expected {EXPECTED_DIM}")
        ok = False
    else:
        print(f"  OK: feature dim == {EXPECTED_DIM}")

    # 2. finite
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).sum().item()
        print(f"  FAIL: {bad} non-finite (NaN/Inf) values in x")
        ok = False
    else:
        print("  OK: no NaN / Inf in node features")

    # 3. pooled plane-d channel (last col): standardized per-part, must be finite
    d = x[:, -1]
    print(f"  pooled-d channel: mean={d.mean():.3f} std={d.std():.3f} "
          f"min={d.min():.3f} max={d.max():.3f}")
    if torch.isfinite(d).all() and d.std() > 1e-3:
        print("  OK: plane-d channel present, finite, and non-degenerate")
    else:
        print("  FAIL: plane-d channel is non-finite or degenerate (std ~ 0)")
        ok = False

    # 4. forward pass at the new width
    try:
        model = BRepGNN(x.shape[1], batch.edge_attr.shape[1], cfg["hidden_dim"],
                        cfg["num_classes"], cfg["num_layers"], cfg["dropout"])
        model.eval()
        with torch.no_grad():
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
        assert logits.shape == (x.shape[0], cfg["num_classes"])
        print(f"  OK: forward pass -> logits {tuple(logits.shape)} "
              f"(node_in auto-sized to {x.shape[1]}, edge_in={batch.edge_attr.shape[1]})")
    except Exception as e:
        print(f"  FAIL: forward pass errored: {e}")
        ok = False

    print("\nRESULT:", "ALL CHECKS PASSED — safe to train." if ok
          else "CHECKS FAILED — do NOT train yet.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
