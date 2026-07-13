"""
repro_train_nan.py — run the REAL train.py loop (device/seed/lr/AdamW/BN identical)
for a few epochs and catch the FIRST batch whose loss goes non-finite, with full
diagnostics on that batch (input finiteness, label range, pre-step grad norm).

Run UNSANDBOXED: .venv_pyg/bin/python diag/repro_train_nan.py [max_steps] [num_workers]
"""
import os
import sys
import yaml
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_geometric.loader import DataLoader
from legacy.dataset import get_dataset
from legacy.device import resolve_device, set_seed
from legacy.model import BRepGNN

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"device={device} max_steps={MAX_STEPS} num_workers={NW} "
          f"lr={cfg['lr']} bs={cfg['batch_size']}")

    train_ds = get_dataset(cfg, "train.txt")
    ld = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=NW, persistent_workers=(NW > 0),
                    pin_memory=(device.type == "cuda"))
    sample = train_ds[0]
    node_in, edge_in = sample.x.shape[1], sample.edge_attr.shape[1]
    train_ds._close_h5()
    model = BRepGNN(node_in, edge_in, cfg["hidden_dim"], cfg["num_classes"],
                    cfg["num_layers"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    model.train()

    step = 0
    worst_loss = 0.0
    worst_grad = 0.0
    done = False
    for epoch in range(cfg["epochs"]):
        if done:
            break
        for batch in ld:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = F.cross_entropy(logits, batch.y)
            lf = torch.isfinite(loss).item()
            worst_loss = max(worst_loss, loss.item() if lf else float("inf"))
            if not lf:
                print(f"\n!!! NON-FINITE loss at step {step} (epoch {epoch}) loss={loss.item()}")
                print(f"    x_finite={torch.isfinite(batch.x).all().item()} "
                      f"edge_finite={torch.isfinite(batch.edge_attr).all().item()} "
                      f"logits_finite={torch.isfinite(logits).all().item()}")
                print(f"    y_min={int(batch.y.min())} y_max={int(batch.y.max())} "
                      f"nodes={batch.x.shape[0]} edges={batch.edge_attr.shape[0]}")
                # is it the inputs or the weights already?
                wf = all(torch.isfinite(p).all().item() for p in model.parameters())
                print(f"    model_params_finite_BEFORE_thisstep={wf}")
                done = True
                break
            loss.backward()
            gnorm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()
                                            if p.grad is not None])).item()
            worst_grad = max(worst_grad, gnorm)
            opt.step()
            if step % 100 == 0:
                acc = (logits.argmax(1) == batch.y).float().mean().item()
                print(f"  step {step:04d} ep{epoch} loss={loss.item():.4f} "
                      f"acc={acc:.3f} grad_norm={gnorm:.3e}")
            step += 1
            if step >= MAX_STEPS:
                done = True
                break
    print(f"\nfinished at step {step}. worst_finite_loss={worst_loss:.4f} "
          f"max_grad_norm={worst_grad:.3e}")
    if worst_loss != float("inf"):
        print("NO NaN encountered in this run.")


if __name__ == "__main__":
    main()
