"""
overfit_check.py — RUN THIS BEFORE train.py.

Trains on ~20 parts for a few hundred steps. A correct pipeline will drive train
accuracy to ~100% within a minute or two. If it plateaus far below that, your data
loader is broken (wrong field names, misaligned labels, bad edge_index) — fix that
BEFORE wasting a night on the full run. This is your tripwire.
"""

import yaml
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from legacy.dataset import get_dataset
from legacy.device import resolve_device
from legacy.model import BRepGNN
from legacy.setup_data import check_data


def _pyg_wheel_status():
    parts = []
    for name in ("torch_scatter", "torch_sparse"):
        try:
            __import__(name)
            parts.append(f"{name}: installed")
        except ImportError:
            parts.append(f"{name}: not installed")
    return ", ".join(parts) + " (OK — PyG fallbacks work without them)"


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = resolve_device(cfg.get("device", "auto"))
    print(f"device={device}")
    print(f"pyg wheels: {_pyg_wheel_status()}")

    if not check_data(cfg):
        raise SystemExit(1)

    full = get_dataset(cfg, "train.txt")
    subset = [full[i] for i in range(min(20, len(full)))]
    loader = DataLoader(subset, batch_size=20, shuffle=True)

    s = subset[0]
    model = BRepGNN(s.x.shape[1], s.edge_attr.shape[1], cfg["hidden_dim"],
                    cfg["num_classes"], cfg["num_layers"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step in range(300):
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = F.cross_entropy(logits, batch.y)
            loss.backward(); opt.step()
        if step % 25 == 0:
            acc = (logits.argmax(1) == batch.y).float().mean().item()
            print(f"step {step:03d}  loss {loss.item():.4f}  train_acc {acc:.4f}")
    print("\nIf train_acc is near 1.0 -> pipeline OK, proceed to train.py.")
    print("If it's stuck low -> data loader bug; do NOT run the full job yet.")


if __name__ == "__main__":
    main()
