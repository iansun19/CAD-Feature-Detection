#!/usr/bin/env python3
"""
Ingest a directory of STEP files into unlabeled B-rep H5 graphs + per-file report.

Graphs mirror diag/regen_dataset.py H5 layout (V_1, A_1_idx, E_* buckets) but omit
labels — suitable for a later labeling pass. Also logs structured per-file stats.

Usage:
  python scripts/ingest_step.py --input-dir path/to/steps --output-h5 out/unlabeled.h5
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from brep.step_ingest import (  # noqa: E402
    StepIngestError,
    StepIngestStats,
    build_V1,
    extract_brep_from_step,
)

logger = logging.getLogger("step_ingest")


def _write_unlabeled_group(g, models):
    """models: list of (id, dict). Same schema as regen write_group minus labels."""
    cur = 0
    V1, A, AV, E1, E2, E3, ids, idx = [], [], [], [], [], [], [], []
    for mid, m in models:
        idx.append((cur, 0))
        V1.append(build_V1(m))
        for arr, dst in [(m["A"], A), (m["E1"], E1), (m["E2"], E2), (m["E3"], E3)]:
            if arr.size:
                dst.append(arr + cur)
        if m["Aval"].size:
            AV.append(m["Aval"])
        ids.append(mid)
        cur += m["N"]
    N = cur
    V1 = np.concatenate(V1)

    def cat(lst):
        return (
            np.concatenate(lst).astype(np.int32) if lst else np.zeros((0, 2), np.int32)
        )

    g.create_dataset("V_1", data=V1, compression="lzf")
    g.create_dataset("idx", data=np.array(idx, np.int32))
    g.create_dataset("CAD_model", data=np.array(ids, dtype=h5py.string_dtype()))
    g.create_dataset("A_1_idx", data=cat(A), compression="lzf")
    g.create_dataset(
        "A_1_values",
        data=(np.concatenate(AV).astype(np.float32) if AV else np.zeros((0,), np.float32)),
        compression="lzf",
    )
    g.create_dataset("E_1_idx", data=cat(E1), compression="lzf")
    g.create_dataset("E_2_idx", data=cat(E2), compression="lzf")
    g.create_dataset("E_3_idx", data=cat(E3), compression="lzf")
    g.create_dataset("A_3_idx", data=np.zeros((0, 2), np.int32))
    g.create_dataset("A_1_shape", data=np.array([N, N], np.int32))


def _collect_step_files(input_dir: str) -> list[str]:
    paths = []
    for name in sorted(os.listdir(input_dir)):
        low = name.lower()
        if low.endswith(".step") or low.endswith(".stp"):
            paths.append(os.path.join(input_dir, name))
    return paths


def _print_summary_table(rows: list[dict]) -> None:
    if not rows:
        print("No files processed.")
        return
    cols = StepIngestStats("", ).csv_fieldnames()
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser(description="Ingest STEP files to unlabeled B-rep H5")
    ap.add_argument("--input-dir", required=True, help="Directory of .step/.stp files")
    ap.add_argument("--output-h5", required=True, help="Output H5 path")
    ap.add_argument("--report-csv", default="", help="Optional CSV report path")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="Process at most N files")
    ap.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"input-dir not found: {input_dir}")

    paths = _collect_step_files(input_dir)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no STEP files in {input_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_h5)) or ".", exist_ok=True)
    fout = h5py.File(args.output_h5, "w")

    batch = []
    gi = 0
    report_rows = []
    ok = fail = 0

    def flush():
        nonlocal batch, gi
        if batch:
            _write_unlabeled_group(fout.create_group(f"batch_{gi:05d}"), batch)
            gi += 1
            batch = []

    for path in paths:
        mid = os.path.splitext(os.path.basename(path))[0]
        stats = StepIngestStats(filename=os.path.basename(path))
        try:
            model, stats = extract_brep_from_step(path, require_labels=False, stats=stats)
            batch.append((mid, model))
            ok += 1
        except StepIngestError as exc:
            stats.error = str(exc)
            stats.success = False
            fail += 1
            logger.error("ingest failed: %s", exc)
        except Exception as exc:
            stats.error = f"{type(exc).__name__}: {exc}"
            stats.success = False
            fail += 1
            logger.exception("unexpected ingest failure for %s", path)

        report_rows.append(stats.csv_row())
        if stats.success and len(batch) >= args.batch_size:
            flush()

    flush()
    fout.close()

    if args.report_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_csv)) or ".", exist_ok=True)
        with open(args.report_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=StepIngestStats("", ).csv_fieldnames())
            w.writeheader()
            w.writerows(report_rows)

    print(f"\nWrote {args.output_h5}: models={ok} failed={fail} batches={gi}")
    _print_summary_table(report_rows)

    # Aggregate fallback counts across successful files.
    ok_rows = [r for r in report_rows if r["success"]]
    if ok_rows:
        n = len(ok_rows)
        keys = [
            "surface_type_other", "undefined_normals", "zero_area_faces",
            "boundary_edges", "non_manifold_edges", "self_loop_edges",
            "convexity_undetermined", "no_adjacent_faces", "partial_skip_boundary",
        ]
        print("\nFallback rates (files with count>0 / total ok):")
        for k in keys:
            hit = sum(1 for r in ok_rows if int(r[k]) > 0)
            total = sum(int(r[k]) for r in ok_rows)
            print(f"  {k:28s}: {hit}/{n} files ({100*hit/n:.1f}%), total events={total}")


if __name__ == "__main__":
    main()
