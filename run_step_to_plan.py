#!/usr/bin/env python
"""End-to-end pipeline: a raw STEP file -> a CamPlan JSON of machining operations.

Chains the two general halves of the CAM path into one command for an ARBITRARY
part (no 96260B / fish_mold hardcoding):

  1. Perception   run_cascade.main() ingests the face graph straight from the
                  STEP (no pre-built graph.npz needed) and writes, under
                  --export-dir:
                      feature_graph_cascade.json
                      setup_descriptor.yaml   (single generated setup)
  2. Context      build_context_v0() derives the stock envelope from the STEP's
                  B-rep extents and auto-discovers the generated setup
                  descriptor sitting next to the feature graph.
  3. Planning     planner.plan() -> CamPlan -> JSON.

Requires the conda ``mlcad`` env (pythonocc / OCC); the repo .venv lacks OCC.

    ~/miniconda3/envs/mlcad/bin/python run_step_to_plan.py <part.step>

Single-setup only: a lone cascade run generates exactly one setup, which is what
the v0 planner consumes. For the 96260B rear+front two-setup plan use
``planner.py --multi-setup``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _default_export_dir(step_path: Path) -> Path:
    stem = step_path.stem.replace(" ", "_").split(".")[0]
    return REPO_ROOT / "pipeline_out" / stem


_CARDINAL_AXES = {
    "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
}


def _parse_opening_axis(spec: str) -> tuple[float, float, float]:
    """Parse '+Y' / '-Z' / '0,1,0' into a unit opening-axis vector."""
    key = spec.strip().upper()
    if key in _CARDINAL_AXES:
        return _CARDINAL_AXES[key]
    parts = spec.replace(" ", "").split(",")
    if len(parts) != 3:
        raise ValueError(
            f"--opening-axis must be one of {sorted(_CARDINAL_AXES)} or 'x,y,z'; got {spec!r}"
        )
    x, y, z = (float(p) for p in parts)
    norm = (x * x + y * y + z * z) ** 0.5
    if norm <= 1e-12:
        raise ValueError(f"--opening-axis vector is zero: {spec!r}")
    return (x / norm, y / norm, z / norm)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full STEP -> CamPlan JSON pipeline for an arbitrary part.",
    )
    parser.add_argument("step", type=Path, help="input STEP/STP CAD file")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output CamPlan JSON (default: examples/cam_plan_<stem>.json)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="cascade artifact dir (default: pipeline_out/<stem>)",
    )
    parser.add_argument(
        "--material",
        default="aluminum",
        help="workpiece material for feed/speed preset selection (default: aluminum)",
    )
    parser.add_argument(
        "--tool-source",
        choices=("hardcoded", "directory", "supabase"),
        default="hardcoded",
        help="tool catalog source (default: hardcoded, self-contained)",
    )
    parser.add_argument(
        "--machining-side",
        choices=("front", "back"),
        default=None,
        help="passed through to the cascade (front->open, back->closed pockets)",
    )
    parser.add_argument(
        "--opening-axis",
        default=None,
        help=(
            "pin the setup opening axis (a shop decision): +X|-X|+Y|-Y|+Z|-Z or "
            "'x,y,z'. Overrides the cascade's auto-detected axis with an explicit "
            "one before planning. Required when geometry cannot resolve the axis."
        ),
    )
    parser.add_argument(
        "--setup-descriptor",
        type=Path,
        default=None,
        help="optional setup descriptor YAML passed through to the cascade",
    )
    parser.add_argument(
        "--seq-search",
        choices=("none", "greedy", "beam"),
        default="beam",
        help="operation sequencing strategy (default: beam)",
    )
    parser.add_argument(
        "--seq-beam-width",
        type=int,
        default=5,
        help="beam width for --seq-search beam (default: 5)",
    )
    parser.add_argument(
        "--skip-cascade",
        action="store_true",
        help="reuse an existing feature_graph_cascade.json in --export-dir",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    step_path = args.step
    if not step_path.is_file():
        print(f"STEP not found: {step_path}", file=sys.stderr)
        return 2

    export_dir = args.export_dir or _default_export_dir(step_path)
    graph_path = export_dir / "feature_graph_cascade.json"
    setup_yaml = export_dir / "setup_descriptor.yaml"

    # --- Stage 1: perception (STEP -> cascade feature graph + setup descriptor) ---
    if not args.skip_cascade:
        import run_cascade

        cascade_argv = [str(step_path), "--export-dir", str(export_dir)]
        if args.machining_side is not None:
            cascade_argv += ["--machining-side", args.machining_side]
        if args.setup_descriptor is not None:
            cascade_argv += ["--setup-descriptor", str(args.setup_descriptor)]
        rc = run_cascade.main(cascade_argv)
        if rc != 0:
            print(f"cascade failed (exit {rc})", file=sys.stderr)
            return rc

    if not graph_path.is_file():
        print(f"feature graph missing: {graph_path}", file=sys.stderr)
        return 1

    # --- Stage 2: machining context (stock from STEP extents + generated setup) ---
    from machining_context import build_context_v0

    # Patch shop inputs into the generated descriptor before planning. Reachability
    # scoping needs two facts the cascade cannot derive from geometry alone:
    #   --opening-axis   an explicit (authoritative) axis -> planner treats it as
    #                    resolved (escape hatch for undetermined parts, or an
    #                    override when the shop disagrees with auto-detect).
    #   --machining-side front|back -> which side faces the spindle; sets the
    #                    reachability direction. Split-panel STEP names (FRONT/
    #                    REAR) are inferred by the cascade; other parts need it.
    generated_descriptor = None
    if args.opening_axis is not None or args.machining_side is not None:
        from setup_descriptor import OpeningAxisSpec, load_setup_descriptor

        generated_descriptor = load_setup_descriptor(setup_yaml)
        vec = _parse_opening_axis(args.opening_axis) if args.opening_axis else None
        for entry in generated_descriptor.setups.values():
            if vec is not None:
                entry.opening_axis = OpeningAxisSpec(mode="explicit", vector=vec)
            if args.machining_side is not None:
                entry.machining_side = args.machining_side
        if vec is not None:
            print(f"Opening axis pinned (explicit): {vec}")
        if args.machining_side is not None:
            print(f"Machining side pinned: {args.machining_side}")

    ctx = build_context_v0(
        step_path,                       # stock envelope derived from B-rep extents
        setup_yaml,                      # only consulted for setups_source="authored"
        graph_path,
        material=args.material,
        tool_source=args.tool_source,
        setups_source="generated",       # auto-discovers export_dir/setup_descriptor.yaml
        generated_descriptor=generated_descriptor,
        generated_descriptor_path=setup_yaml if setup_yaml.is_file() else None,
    )

    # --- Stage 3: planning (features -> operations -> CamPlan JSON) ---
    from cam_plan_schema import write_cam_plan
    from planner import _print_summary, plan

    cam_plan = plan(
        graph_path,
        ctx,
        seq_search=args.seq_search,
        seq_beam_width=args.seq_beam_width,
    )

    out_path = args.out or (REPO_ROOT / "examples" / f"cam_plan_{step_path.stem.replace(' ', '_').split('.')[0]}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_cam_plan(out_path, cam_plan)
    print(f"\nWrote {out_path}")
    _print_summary(cam_plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
