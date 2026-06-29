"""
Scan MFCAD++ batched H5 files for empty or invalid B-rep face ranges.

MFCAD++ idx layout (per batch group):
  idx[i, 0] — global B-rep face start for CAD_model[i] (local = value - idx[0, 0])
  idx[i, 1] — global mesh (V_2) bound; NOT the B-rep slice end
  B-rep end for model i = idx[i+1, 0] (or len(V_1) for the last model in the batch)

The common loader bug is treating idx[i] as [start, end) into V_1, which yields
empty slices once idx[i, 0] >= len(V_1) (most models after the first batch).

Usage:
    python diag/check_empty.py /path/to/training_MFCAD++.h5
    python diag/check_empty.py /path/to/training_MFCAD++.h5 /path/to/train.txt
"""

import argparse
import sys

import h5py
import numpy as np


def _pid(raw_id):
    return raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)


def _brep_bounds(idx_arr, model_idx, v1_len):
    """Correct per-model B-rep row range within a batch."""
    base = int(idx_arr[0, 0])
    start = int(idx_arr[model_idx, 0]) - base
    if model_idx + 1 < len(idx_arr):
        end = int(idx_arr[model_idx + 1, 0]) - base
    else:
        end = v1_len
    return start, end


def scan_h5(h5_path, split_ids=None):
    degenerate = []
    naive_empty = []
    misaligned = []
    total_models = 0
    total_batches = 0

    with h5py.File(h5_path, "r") as f:
        for batch_key in f.keys():
            total_batches += 1
            batch = f[batch_key]
            cad = batch["CAD_model"][()]
            idx_arr = batch["idx"][()]
            v1_len = batch["V_1"].shape[0]
            n_cad = len(cad)
            n_idx = len(idx_arr)
            if n_cad != n_idx:
                misaligned.append((batch_key, n_cad, n_idx))

            for i, raw_id in enumerate(cad):
                total_models += 1
                pid = _pid(raw_id)
                start, end = _brep_bounds(idx_arr, i, v1_len)
                if end <= start or start < 0 or end > v1_len:
                    degenerate.append({
                        "batch": batch_key,
                        "model_idx": i,
                        "id": pid,
                        "start": start,
                        "end": end,
                        "n_faces": end - start,
                    })

                # Naive bug: start, end = idx[i] used directly on V_1
                naive_s, naive_e = int(idx_arr[i, 0]), int(idx_arr[i, 1])
                actual = max(0, min(naive_e, v1_len) - naive_s)
                if naive_s >= v1_len or actual == 0:
                    naive_empty.append({
                        "batch": batch_key,
                        "model_idx": i,
                        "id": pid,
                        "naive_start": naive_s,
                        "naive_end": naive_e,
                        "v1_len": v1_len,
                    })

    split_set = set(split_ids) if split_ids else None
    degenerate_in_split = []
    if split_set is not None:
        id_to_entry = {d["id"]: d for d in degenerate}
        for pid in split_ids:
            if pid in id_to_entry:
                degenerate_in_split.append(id_to_entry[pid])

    return {
        "h5_path": h5_path,
        "total_batches": total_batches,
        "total_models": total_models,
        "misaligned_batches": misaligned,
        "degenerate": degenerate,
        "naive_empty": naive_empty,
        "degenerate_in_split": degenerate_in_split,
        "split_ids_requested": len(split_ids) if split_ids else None,
    }


def print_report(report):
    print(f"H5: {report['h5_path']}")
    print(f"  batches: {report['total_batches']}")
    print(f"  models:  {report['total_models']}")

    mis = report["misaligned_batches"]
    if mis:
        print(f"\n  MISALIGNED idx vs CAD_model length in {len(mis)} batch(es):")
        for batch_key, n_cad, n_idx in mis[:10]:
            print(f"    {batch_key}: CAD_model={n_cad}, idx={n_idx}")
        if len(mis) > 10:
            print(f"    ... and {len(mis) - 10} more")
    else:
        print("  idx/CAD_model lengths: aligned in all batches")

    deg = report["degenerate"]
    print(f"\n  degenerate B-rep ranges (correct idx parsing): {len(deg)}")
    if deg:
        print("  examples:")
        for d in deg[:20]:
            print(
                f"    {d['id']!r}  batch={d['batch']}  "
                f"model_idx={d['model_idx']}  start={d['start']}  end={d['end']}"
            )
        if len(deg) > 20:
            print(f"    ... and {len(deg) - 20} more")

    naive = report["naive_empty"]
    print(f"\n  empty slices from naive idx[i] -> V_1[start:end]: {len(naive)}")
    if naive:
        print("  (this is the dataset.py bug — idx[i,1] is mesh bound, not B-rep end)")
        print("  examples:")
        for d in naive[:10]:
            print(
                f"    {d['id']!r}  batch={d['batch']}  "
                f"naive=[{d['naive_start']},{d['naive_end']})  V_1 len={d['v1_len']}"
            )
        if len(naive) > 10:
            print(f"    ... and {len(naive) - 10} more")

    if report["split_ids_requested"] is not None:
        n = report["split_ids_requested"]
        in_split = report["degenerate_in_split"]
        print(f"\n  split file: {n} ids")
        print(f"  degenerate ids present in split: {len(in_split)}")
        if in_split:
            print("  split examples:")
            for d in in_split[:10]:
                print(f"    {d['id']!r}  batch={d['batch']}  start={d['start']}  end={d['end']}")


def main():
    ap = argparse.ArgumentParser(description="Find empty/invalid face ranges in MFCAD++ H5")
    ap.add_argument("h5_path", help="path to e.g. training_MFCAD++.h5")
    ap.add_argument("split_file", nargs="?", help="optional train.txt / val.txt / test.txt")
    args = ap.parse_args()

    split_ids = None
    if args.split_file:
        with open(args.split_file) as f:
            split_ids = [line.strip() for line in f if line.strip()]

    report = scan_h5(args.h5_path, split_ids)
    print_report(report)
    sys.exit(1 if report["degenerate"] or report["naive_empty"] else 0)


if __name__ == "__main__":
    main()
