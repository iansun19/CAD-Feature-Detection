#!/usr/bin/env python3
"""
run_pipeline.py — Phase 4 unified pipeline: STEP (or MFCAD++ part) -> output bundle.

One command produces a directory with:
  feature_graph.json      — feature instances, edges, params (when STEP available)
  face_predictions.jsonl  — per-face class + feature_id + confidence
  viewer.html             — interactive numbered-face viewer (optional, default on)
  manifest.json           — paths, checkpoint, summary metadata
  pipeline.log            — run log

Usage:
    python run_pipeline.py --step path/to/part.step
    python run_pipeline.py --step path/to/part.step --out-dir out/my_part
    python run_pipeline.py --part-id 29000
    python run_pipeline.py --step-dir incoming/ --batch

Requires the unified conda env (torch + pythonocc-core + PyG):
  conda env create -f environment.yml && conda activate mlcad
"""
from __future__ import annotations

import env_bootstrap  # noqa: F401, E402

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from device import resolve_device  # noqa: E402
from feature_graph import write_feature_graph  # noqa: E402
from feature_params import default_step_path, enrich_graph_with_params  # noqa: E402
from infer_feature_graph import (  # noqa: E402
    default_run_dir,
    infer_from_tensors,
    load_model,
    load_part_from_dataset,
)
from pipeline.core import (  # noqa: E402
    build_manifest,
    infer_graph,
    utc_now,
    write_face_predictions,
    write_manifest,
)
from pipeline.ingest import ingest_step_to_npz, require_occ  # noqa: E402


class _Tee:
    """Write to log file and stdout."""

    def __init__(self, log_path: Path):
        self._log = open(log_path, "w")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._log.write(data)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        self._log.close()


def _build_viewer(step_path: Path, graph_path: Path, html_path: Path, part_id: str) -> None:
    require_occ("viewer build")
    from feature_graph_viewer.build import DEFAULT_TEMPLATE, build_viewer

    build_viewer(
        part_id=part_id,
        graph_path=graph_path,
        step_path=step_path,
        output_path=html_path,
        template_path=DEFAULT_TEMPLATE,
        open_browser=False,
    )


def _enrich_graph(graph_path: Path, step_path: Path) -> None:
    require_occ("param enrichment")
    with open(graph_path) as f:
        graph = json.load(f)
    enrich_graph_with_params(graph, step_path)
    write_feature_graph(str(graph_path), graph)


def run_one(
    *,
    part_id: str,
    out_dir: Path,
    cfg: dict,
    ckpt: Path,
    device,
    step_path: Path | None,
    npz_path: Path | None,
    split_file: str,
    with_params: bool,
    with_viewer: bool,
    from_part_id: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    started = utc_now()

    try:
        entity_ids = None
        if from_part_id:
            print(f"[pipeline] loading H5 graph for part-id={part_id}")
            data = load_part_from_dataset(cfg, part_id, split_file)
            x = data.x.cpu().numpy()
            edge_index = data.edge_index.cpu().numpy()
            edge_attr = data.edge_attr.cpu().numpy()
            if step_path is None:
                step_path = default_step_path(part_id)
        elif npz_path is not None:
            print(f"[pipeline] loading graph tensors from {npz_path}")
            d = np.load(npz_path)
            x = d["x"]
            edge_index = d["edge_index"]
            edge_attr = d["edge_attr"]
            entity_ids = d["entity_ids"] if "entity_ids" in d else None
            if step_path is None and "step_path" in d:
                sp = d["step_path"]
                step_path = Path(str(sp.item() if hasattr(sp, "item") else sp))
        else:
            raise RuntimeError("internal error: no graph source")

        print(f"[pipeline] device={device}  checkpoint={ckpt}")
        model, head_w = load_model(str(ckpt), cfg, device)
        pred, conf = infer_from_tensors(model, x, edge_index, edge_attr, head_w, device)

        graph = infer_graph(
            pred, conf, edge_index,
            part_id=part_id,
            entity_ids=entity_ids,
        )

        graph_path = out_dir / "feature_graph.json"
        pred_path = out_dir / "face_predictions.jsonl"
        write_feature_graph(str(graph_path), graph)
        write_face_predictions(pred_path, pred, conf, edge_index, graph, entity_ids)
        print(f"[pipeline] wrote {graph_path.name}  {pred_path.name}")
        print(f"[pipeline]   {graph.get('n_features')} features, {graph.get('n_edges')} edges")

        params_ok = False
        if with_params:
            if step_path is None or not step_path.is_file():
                print(f"[pipeline] warning: params skipped (no STEP at {step_path})")
            else:
                print(f"[pipeline] enriching params from {step_path}")
                _enrich_graph(graph_path, step_path)
                with open(graph_path) as f:
                    graph = json.load(f)
                params_ok = True

        viewer_ok = False
        if with_viewer:
            if step_path is None or not step_path.is_file():
                print(f"[pipeline] warning: viewer skipped (no STEP at {step_path})")
            else:
                html_path = out_dir / "viewer.html"
                print(f"[pipeline] building viewer -> {html_path.name}")
                _build_viewer(step_path, graph_path, html_path, part_id)
                viewer_ok = True

        finished = utc_now()
        manifest = build_manifest(
            part_id=part_id,
            out_dir=out_dir,
            step_path=step_path if step_path and step_path.is_file() else None,
            ckpt=ckpt,
            graph=graph,
            started_at=started,
            finished_at=finished,
            params_enabled=params_ok,
            viewer_enabled=viewer_ok,
        )
        write_manifest(out_dir / "manifest.json", manifest)
        print(f"[pipeline] done -> {out_dir.resolve()}")
        print(f"[pipeline] manifest: {out_dir / 'manifest.json'}")
        return manifest
    finally:
        sys.stdout = tee._stdout
        tee.close()


def main():
    ap = argparse.ArgumentParser(
        description="Unified STEP/part -> feature graph pipeline (Phase 4)",
    )
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--step", type=Path, help="input STEP file")
    src.add_argument("--part-id", help="MFCAD++ part id (uses H5 graph + STEP for params)")
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="output directory (default: pipeline_out/<part_id>/)",
    )
    ap.add_argument("--batch", action="store_true",
                    help="with --step-dir: process every *.step in the directory")
    ap.add_argument("--step-dir", type=Path, default=None,
                    help="directory of STEP files for --batch")
    ap.add_argument("--split", default="test.txt", help="split file for --part-id")
    ap.add_argument("--step-file", type=Path, default=None,
                    help="explicit STEP for --part-id (overrides default test path)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", default=None, help="checkpoint run dir")
    ap.add_argument(
        "--with-params", action=argparse.BooleanOptionalAction, default=True,
        help="attach OCC geometry params (default: on)",
    )
    ap.add_argument(
        "--viewer", action=argparse.BooleanOptionalAction, default=True,
        help="build viewer.html (default: on)",
    )
    args = ap.parse_args()

    if args.batch:
        if args.step is not None or args.part_id is not None:
            raise SystemExit("--batch cannot be combined with --step or --part-id")
    elif args.step is None and args.part_id is None:
        raise SystemExit("one of --step or --part-id is required (or use --batch with --step-dir)")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = resolve_device(cfg.get("device", "auto"))
    run_dir = args.run or default_run_dir(cfg["out_dir"])
    ckpt = Path(run_dir) / "best_model.pt"
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    if args.batch:
        step_dir = args.step_dir
        if step_dir is None or not step_dir.is_dir():
            raise SystemExit("--batch requires --step-dir pointing to a directory")
        steps = sorted(step_dir.glob("*.step")) + sorted(step_dir.glob("*.STEP"))
        if not steps:
            raise SystemExit(f"no STEP files in {step_dir}")
        base_out = args.out_dir or Path("pipeline_out")
        errors = []
        for step_path in steps:
            part_id = step_path.stem
            out_dir = base_out / part_id
            npz_path = out_dir / "graph.npz"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                require_occ("STEP batch ingest")
                ingest_step_to_npz(step_path, npz_path, cfg)
                run_one(
                    part_id=part_id,
                    out_dir=out_dir,
                    cfg=cfg,
                    ckpt=ckpt,
                    device=device,
                    step_path=step_path,
                    npz_path=npz_path,
                    split_file=args.split,
                    with_params=args.with_params,
                    with_viewer=args.viewer,
                    from_part_id=False,
                )
            except Exception as exc:
                errors.append((part_id, str(exc)))
                print(f"[pipeline] FAILED {part_id}: {exc}", file=sys.__stdout__)
        if errors:
            raise SystemExit(f"{len(errors)} part(s) failed: {[e[0] for e in errors]}")
        return

    if args.step is not None:
        if not args.step.is_file():
            raise SystemExit(f"STEP not found: {args.step}")
        part_id = args.step.stem
        out_dir = args.out_dir or Path("pipeline_out") / part_id
        npz_path = out_dir / "graph.npz"
        out_dir.mkdir(parents=True, exist_ok=True)
        require_occ("STEP ingest")
        ingest_step_to_npz(args.step, npz_path, cfg)
        run_one(
            part_id=part_id,
            out_dir=out_dir,
            cfg=cfg,
            ckpt=ckpt,
            device=device,
            step_path=args.step.resolve(),
            npz_path=npz_path,
            split_file=args.split,
            with_params=args.with_params,
            with_viewer=args.viewer,
            from_part_id=False,
        )
        return

    part_id = args.part_id
    out_dir = args.out_dir or Path("pipeline_out") / part_id
    step_path = args.step_file or default_step_path(part_id)
    run_one(
        part_id=part_id,
        out_dir=out_dir,
        cfg=cfg,
        ckpt=ckpt,
        device=device,
        step_path=step_path,
        npz_path=None,
        split_file=args.split,
        with_params=args.with_params,
        with_viewer=args.viewer,
        from_part_id=True,
    )


if __name__ == "__main__":
    main()
