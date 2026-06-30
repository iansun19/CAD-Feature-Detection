"""
setup_data.py — verify MFCAD++ is present before training.

Usage:
    python setup_data.py          # check config.yaml paths
    python setup_data.py --inspect  # also print H5 top-level keys if found
"""

import argparse
import os
import sys

import yaml

from dataset import SPLIT_H5

DOWNLOAD_URL = (
    "https://pure.qub.ac.uk/en/datasets/"
    "mfcad-dataset-dataset-for-paper-hierarchical-cadnet-learning-from/"
)

REQUIRED = ("train.txt", "val.txt", "test.txt")


def _h5_paths(cfg):
    root = cfg["data_root"]
    fmt = cfg.get("h5_format", "mfcadpp")
    if fmt in ("mfcadpp", "mfcadpp_regen"):
        default_dir = "hierarchical_graphs_regen" if fmt == "mfcadpp_regen" else "hierarchical_graphs"
        h5_dir = os.path.join(root, cfg.get("h5_dir", default_dir))
        return {name: os.path.join(h5_dir, fname) for name, fname in SPLIT_H5.items()}
    return {"graphs": os.path.join(root, cfg["h5_path"])}


def check_data(cfg, inspect_h5=False):
    root = cfg["data_root"]
    abs_root = os.path.abspath(root)
    issues = []

    if not os.path.isdir(root):
        issues.append(f"data_root missing: {abs_root}")
    else:
        if not os.listdir(root):
            issues.append(f"data_root is empty: {abs_root}")
        for name in REQUIRED:
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                issues.append(f"missing split file: {name}")

    for label, path in _h5_paths(cfg).items():
        if not os.path.isfile(path):
            issues.append(f"missing H5 ({label}): {os.path.abspath(path)}")

    if issues:
        print("MFCAD++ data is not ready:\n")
        for item in issues:
            print(f"  - {item}")
        print(
            f"\nDownload (~1.5 GB) from:\n  {DOWNLOAD_URL}\n"
            f"\nUnzip into the repo as:\n  MFCAD++_dataset/\n"
            "\nExpected layout:\n"
            "  MFCAD++_dataset/train.txt\n"
            "  MFCAD++_dataset/val.txt\n"
            "  MFCAD++_dataset/test.txt\n"
            "  MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5\n"
            "  MFCAD++_dataset/hierarchical_graphs/val_MFCAD++.h5\n"
            "  MFCAD++_dataset/hierarchical_graphs/test_MFCAD++.h5"
        )
        return False

    print(f"OK: data_root={abs_root}")
    for split_name, path in _h5_paths(cfg).items():
        print(f"    {split_name} -> {os.path.abspath(path)}")
    for name in REQUIRED:
        path = os.path.join(root, name)
        with open(path) as f:
            n = sum(1 for line in f if line.strip())
        print(f"    {name}: {n} part ids")

    if inspect_h5:
        from dataset import inspect_h5
        paths = _h5_paths(cfg)
        first = paths.get("train.txt") or next(iter(paths.values()))
        print()
        inspect_h5(first)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="print H5 structure")
    args = ap.parse_args()
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    ok = check_data(cfg, inspect_h5=args.inspect)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
