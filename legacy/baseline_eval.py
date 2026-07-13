"""
baseline_eval.py — rule-based baselines to compare against the GNN.

Three baselines (increasing fairness):
  1. Majority class — always predict the most common label in train.
  2. Per-surface-type majority — predict train majority label per surface type
     (cols 0:num_surface_types of data.x); no graph structure.
  3. k-NN on node features (optional) — nearest train face in L2-normalized
     node-feature space; tests per-face geometry without message passing.

Metrics match evaluate.py: overall accuracy + macro-F1 over classes present
in the test ground truth.

Usage:
    python baseline_eval.py
    python baseline_eval.py --knn --device cuda
    python baseline_eval.py --run runs/20260630_042957   # add GNN row to summary
    python baseline_eval.py --config config.yaml
"""

import argparse
import glob
import os

import numpy as np
import torch
import yaml

from legacy.dataset import _cache_path
from legacy.device import resolve_device, set_seed
from legacy.evaluate import load_class_names, per_class_metrics
from brep.taxonomy import NUM_CLASSES


SURFACE_NAMES = ("plane", "cylinder", "cone", "sphere", "torus", "other")


def latest_run(out_dir):
    runs = sorted(glob.glob(os.path.join(out_dir, "*")))
    runs = [r for r in runs if os.path.isfile(os.path.join(r, "per_class_metrics.csv"))]
    if not runs:
        return None
    return runs[-1]


def load_gnn_metrics(run_dir):
    """Read overall acc + macro-F1 from a prior evaluate.py run."""
    import csv

    path = os.path.join(run_dir, "per_class_metrics.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    supports = np.array([int(r["support"]) for r in rows], dtype=np.float64)
    f1s = np.array([float(r["f1"]) for r in rows], dtype=np.float64)
    recalls = np.array([float(r["recall"]) for r in rows], dtype=np.float64)
    present = supports > 0
    macro_f1 = f1s[present].mean()
    # weighted accuracy = sum(support * recall) / sum(support)
    acc = (supports * recalls).sum() / max(supports.sum(), 1)
    return acc, macro_f1


def load_cache(cfg, split_file):
    path = _cache_path(cfg, split_file)
    if not os.path.isfile(path):
        raise SystemExit(
            f"cache not found: {path}\n"
            "Run train.py or evaluate.py once to build it, or check config.yaml paths.")
    print(f"  loading {path}")
    return torch.load(path, weights_only=False)


def stack_faces(data_list, num_surface_types):
    """Return (surface_type [N], labels [N], features [N, D]) for all faces."""
    st_parts, y_parts, x_parts = [], [], []
    for g in data_list:
        st_parts.append(g.x[:, :num_surface_types].argmax(dim=1))
        y_parts.append(g.y)
        x_parts.append(g.x)
    return torch.cat(st_parts), torch.cat(y_parts), torch.cat(x_parts)


def score_predictions(y_true, y_pred, num_classes):
    """Accuracy + macro-F1, same definitions as evaluate.py."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    _, _, f1, support = per_class_metrics(cm)
    acc = (y_true == y_pred).mean()
    present = support > 0
    macro_f1 = f1[present].mean() if present.any() else 0.0
    return acc, macro_f1, int(present.sum())


def baseline_majority_class(y_train, y_test, num_classes):
    counts = np.bincount(y_train.numpy(), minlength=num_classes)
    majority = int(counts.argmax())
    preds = np.full(len(y_test), majority, dtype=np.int64)
    acc, macro_f1, n_present = score_predictions(y_test.numpy(), preds, num_classes)
    return majority, int(counts[majority]), acc, macro_f1, n_present


def baseline_surface_type_majority(st_train, y_train, st_test, y_test, num_classes,
                                   num_surface_types, global_fallback):
    surface_majority = np.full(num_surface_types, global_fallback, dtype=np.int64)
    seen_in_train = np.zeros(num_surface_types, dtype=bool)
    for s in range(num_surface_types):
        mask = (st_train == s).numpy()
        if mask.any():
            seen_in_train[s] = True
            surface_majority[s] = np.bincount(
                y_train.numpy()[mask], minlength=num_classes).argmax()

    st_np = st_test.numpy()
    preds = surface_majority[st_np]
    fallback_used = int((~seen_in_train[st_np]).sum())
    acc, macro_f1, n_present = score_predictions(y_test.numpy(), preds, num_classes)
    return surface_majority, seen_in_train, acc, macro_f1, n_present, fallback_used


@torch.no_grad()
def baseline_knn(X_train, y_train, X_test, y_test, k, device, num_classes,
                 batch_size=8192):
    """k-NN via cosine similarity (L2-normalized dot product) on GPU/CPU."""
    eps = 1e-8
    X_train = X_train / X_train.norm(dim=1, keepdim=True).clamp_min(eps)
    X_test = X_test / X_test.norm(dim=1, keepdim=True).clamp_min(eps)
    X_train_d = X_train.to(device)
    y_train_d = y_train.to(device)

    preds = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size].to(device)
        sim = batch @ X_train_d.T
        nn_idx = sim.topk(k, dim=1).indices
        nn_labels = y_train_d[nn_idx]
        votes = torch.zeros(batch.size(0), num_classes, device=device)
        ones = torch.ones(batch.size(0), k, device=device)
        votes.scatter_add_(1, nn_labels, ones)
        preds.append(votes.argmax(dim=1).cpu())
    y_pred = torch.cat(preds).numpy()
    return score_predictions(y_test.numpy(), y_pred, num_classes)


def print_summary_table(rows):
    print(f"\n{'=' * 62}")
    print("SUMMARY")
    print(f"{'=' * 62}")
    print(f"{'Model':<40}{'Test Acc':<12}{'Macro-F1':<12}")
    print("-" * 62)
    for name, acc, f1 in rows:
        print(f"{name:<40}{acc:<12.4f}{f1:<12.4f}")


def main():
    ap = argparse.ArgumentParser(description="Rule-based baselines vs GNN")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", default=None,
                    help="run dir with per_class_metrics.csv; default = latest")
    ap.add_argument("--no-gnn", action="store_true",
                    help="skip loading GNN numbers for comparison table")
    ap.add_argument("--knn", action="store_true", help="run k-NN baseline (slower)")
    ap.add_argument("--k", type=int, default=5, help="k for k-NN baseline")
    ap.add_argument("--max-train-faces", type=int, default=None,
                    help="subsample train faces for k-NN (default: all)")
    ap.add_argument("--batch-size", type=int, default=8192,
                    help="test batch size for k-NN matmul")
    ap.add_argument("--device", default=None,
                    help="device for k-NN (default: config device or auto)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))

    num_classes = cfg["num_classes"]
    assert num_classes == NUM_CLASSES
    num_st = cfg["num_surface_types"]
    names = load_class_names(cfg["data_root"], num_classes)
    device = resolve_device(args.device or cfg.get("device", "auto"))

    print(f"device={device}  num_classes={num_classes}  num_surface_types={num_st}")
    print("Loading cached graphs...")
    train_data = load_cache(cfg, "train.txt")
    test_data = load_cache(cfg, "test.txt")

    st_tr, y_tr, X_tr = stack_faces(train_data, num_st)
    st_te, y_te, X_te = stack_faces(test_data, num_st)
    print(f"train faces = {len(y_tr):,}   test faces = {len(y_te):,}   "
          f"node dim = {X_tr.shape[1]}")

    summary_rows = []

    # ---- Baseline 1 ----
    maj_id, maj_support, acc1, f1_1, n1 = baseline_majority_class(
        y_tr, y_te, num_classes)
    print(f"\n{'=' * 62}")
    print("BASELINE 1: Majority class (always predict train majority)")
    print(f"{'=' * 62}")
    print(f"majority class = {maj_id} ({names[maj_id]!r})  "
          f"train support = {maj_support:,}")
    print(f"test accuracy  = {acc1:.4f}")
    print(f"test macro-F1  = {f1_1:.4f}  (classes present in test: {n1})")
    summary_rows.append(("Baseline 1 (majority class)", acc1, f1_1))

    # ---- Baseline 2 ----
    _, seen_st, acc2, f1_2, n2, fallback = baseline_surface_type_majority(
        st_tr, y_tr, st_te, y_te, num_classes, num_st, global_fallback=maj_id)
    print(f"\n{'=' * 62}")
    print("BASELINE 2: Per-surface-type majority (no graph)")
    print(f"{'=' * 62}")
    print("Train majority label per surface type:")
    for s in range(num_st):
        if not seen_st[s]:
            print(f"  {SURFACE_NAMES[s]:<10} (id={s}): (unseen in train, "
                  f"fallback={maj_id})")
            continue
        mask = (st_tr == s).numpy()
        labels = y_tr.numpy()[mask]
        top = int(np.bincount(labels, minlength=num_classes).argmax())
        purity = (labels == top).mean()
        print(f"  {SURFACE_NAMES[s]:<10} (id={s}): label={top:>2} ({names[top]!r})  "
              f"purity={purity:.3f}  n={mask.sum():,}")
    print(f"test accuracy  = {acc2:.4f}")
    print(f"test macro-F1  = {f1_2:.4f}  (classes present in test: {n2})")
    if fallback:
        print(f"(note: {fallback} test faces used global majority fallback)")
    summary_rows.append(("Baseline 2 (per-surface-type)", acc2, f1_2))

    # ---- Baseline 3 (optional) ----
    if args.knn:
        X_knn, y_knn = X_tr, y_tr
        if args.max_train_faces and len(y_tr) > args.max_train_faces:
            rng = torch.Generator().manual_seed(cfg.get("seed", 42))
            idx = torch.randperm(len(y_tr), generator=rng)[:args.max_train_faces]
            X_knn, y_knn = X_tr[idx], y_tr[idx]
            print(f"\n[k-NN] subsampled train to {len(y_knn):,} faces")
        print(f"\n{'=' * 62}")
        print(f"BASELINE 3: k-NN (k={args.k}, node dim={X_tr.shape[1]})")
        print(f"{'=' * 62}")
        acc3, f1_3, n3 = baseline_knn(
            X_knn, y_knn, X_te, y_te, args.k, device, num_classes,
            batch_size=args.batch_size)
        print(f"train faces used = {len(y_knn):,}   device = {device}")
        print(f"test accuracy  = {acc3:.4f}")
        print(f"test macro-F1  = {f1_3:.4f}  (classes present in test: {n3})")
        summary_rows.append((f"Baseline 3 (k-NN k={args.k})", acc3, f1_3))

    # ---- GNN comparison row ----
    if not args.no_gnn:
        run_dir = args.run or latest_run(cfg["out_dir"])
        if run_dir and os.path.isfile(os.path.join(run_dir, "per_class_metrics.csv")):
            gnn_acc, gnn_f1 = load_gnn_metrics(run_dir)
            print(f"\nGNN metrics from {run_dir}")
            summary_rows.append(("GNN (from evaluate.py)", gnn_acc, gnn_f1))
        else:
            print(f"\n(no GNN run found under {cfg['out_dir']}/; "
                  "pass --run or run evaluate.py first)")

    print_summary_table(summary_rows)

    if summary_rows and summary_rows[-1][0].startswith("GNN"):
        gnn_acc = summary_rows[-1][1]
        print(f"\nGNN accuracy lift over baseline 2: {gnn_acc - acc2:+.4f} "
              f"({100 * (gnn_acc - acc2):+.2f} pp)")


if __name__ == "__main__":
    main()
