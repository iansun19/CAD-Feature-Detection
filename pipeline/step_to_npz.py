#!/usr/bin/env python3
"""CLI: STEP file -> graph.npz (same tensors as run_pipeline ingest)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ingest import ingest_step_to_npz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="STEP -> graph.npz for inference")
    ap.add_argument("step", type=Path)
    ap.add_argument("out_npz", type=Path)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    if not args.step.is_file():
        raise SystemExit(f"STEP not found: {args.step}")

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    x, edge_index, edge_attr = ingest_step_to_npz(args.step, args.out_npz, cfg)
    print(
        f"wrote {args.out_npz}  faces={x.shape[0]}  edges={edge_index.shape[1]}"
    )


if __name__ == "__main__":
    main()
