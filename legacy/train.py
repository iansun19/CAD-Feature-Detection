"""
train.py — overnight training entry point.

Usage:
    python train.py                  # uses config.yaml
Outputs to runs/<timestamp>/:
    best_model.pt   — weights at best val accuracy
    log.txt         — per-epoch metrics (read this in the morning)
    config_used.yaml
"""

import env_bootstrap  # noqa: F401

import os
import time
import datetime
import yaml
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# Workers share sample tensors via shared memory. The default 'file_descriptor'
# strategy keeps one open FD per shared tensor; with persistent workers these
# accumulate across epochs until we blow past `ulimit -n` ("Too many open
# files"). 'file_system' names shared segments instead, using no per-tensor FD.
torch.multiprocessing.set_sharing_strategy("file_system")

from legacy.dataset import get_dataset
from legacy.device import resolve_device, set_seed
from legacy.model import BRepGNN
from brep.taxonomy import NUM_CLASSES


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        correct += (logits.argmax(1) == batch.y).sum().item()
        total += batch.y.numel()
    return correct / max(total, 1)


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        # free throughput on Ampere+ : allow TF32 matmuls and autotune kernels
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(cfg["out_dir"], stamp)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    logf = open(os.path.join(out, "log.txt"), "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line); logf.write(line + "\n"); logf.flush()

    log(f"device={device}  out={out}")

    train_ds = get_dataset(cfg, "train.txt")
    val_ds = get_dataset(cfg, "val.txt")
    test_ds = get_dataset(cfg, "test.txt")
    log(f"sizes train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    persistent = cfg["num_workers"] > 0
    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=cfg["num_workers"],
                          persistent_workers=persistent,
                          pin_memory=(device.type == "cuda"))
    val_ld = DataLoader(val_ds, batch_size=cfg["batch_size"],
                        num_workers=cfg["num_workers"],
                        persistent_workers=persistent)
    test_ld = DataLoader(test_ds, batch_size=cfg["batch_size"],
                         num_workers=cfg["num_workers"],
                         persistent_workers=persistent)

    # infer feature dims from one sample (close H5 so workers can pickle the dataset)
    sample = train_ds[0]
    node_in, edge_in = sample.x.shape[1], sample.edge_attr.shape[1]
    train_ds._close_h5()
    val_ds._close_h5()
    test_ds._close_h5()
    log(f"node_in={node_in} edge_in={edge_in}")

    model = BRepGNN(node_in, edge_in, cfg["hidden_dim"], cfg["num_classes"],
                    cfg["num_layers"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=4)

    best_val, best_epoch, bad = 0.0, -1, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        run_loss = run_acc = nb = 0
        for batch in train_ld:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            assert int(batch.y.min()) >= 0 and int(batch.y.max()) < NUM_CLASSES
            loss = F.cross_entropy(logits, batch.y)
            loss.backward()
            opt.step()
            run_loss += loss.item(); run_acc += accuracy(logits, batch.y); nb += 1
        val_acc = evaluate(model, val_ld, device)
        sched.step(val_acc)
        # MPS caches a buffer per distinct allocation size; variable-sized graph
        # batches make the pool grow unbounded against unified memory until the
        # machine swaps. Drain it each epoch to keep epoch time flat.
        if device.type == "mps":
            torch.mps.empty_cache()
        log(f"epoch {epoch:03d} | loss {run_loss/nb:.4f} | "
            f"train_acc {run_acc/nb:.4f} | val_acc {val_acc:.4f} | "
            f"{time.time()-t0:.1f}s")

        if val_acc > best_val:
            best_val, best_epoch, bad = val_acc, epoch, 0
            torch.save(model.state_dict(), os.path.join(out, "best_model.pt"))
            log(f"  ^ new best, saved.")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"early stop (no val gain in {cfg['patience']} epochs).")
                break

    # final test with best weights
    model.load_state_dict(torch.load(os.path.join(out, "best_model.pt"),
                                     map_location=device))
    test_acc = evaluate(model, test_ld, device)
    log(f"DONE. best_val={best_val:.4f}@{best_epoch} test_acc={test_acc:.4f}")
    logf.close()


if __name__ == "__main__":
    main()
