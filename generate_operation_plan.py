"""Generate an operation/toolpath plan from a saved mold's features.

Round-trip: load a mold's features from Supabase -> reconstruct the
feature_graph_cascade.json the cascade would have written -> build a
MachiningContext -> run the existing rule-based planner (planner.plan) ->
emit a CamPlan as inspectable JSON.

Two modes:
  planner (default) - full pipeline: real feature->op mapping, tool selection,
                      handbook feeds/speeds, reachability, sequencing.
  lightweight       - config-only placeholder ops from feature_operation_map,
                      for inspecting the mapping without tools/stock. Emits
                      operation records with tool/feeds-speeds as placeholders.

CLI:
    python generate_operation_plan.py <mold_id> [--out plan.json]
        [--mode planner|lightweight] [--tool-source hardcoded|directory|supabase]
        [--material 6061] [--oversize-mm 2.0]
        [--extents-json '{"min":[...],"max":[...]}'] [--setup-yaml path.yaml]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import ground_truth_store as gts
import feature_operation_map as fmap


def _write_temp_json(data: Mapping[str, Any], suffix: str) -> Path:
    fd = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    with fd as fh:
        json.dump(data, fh, indent=2)
    return Path(fd.name)


def _write_temp_yaml(descriptor: Mapping[str, Any]) -> Path:
    import yaml

    fd = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    with fd as fh:
        yaml.safe_dump(dict(descriptor), fh, sort_keys=False)
    return Path(fd.name)


def generate_lightweight_plan(
    mold_id: str, *, client: Any | None = None
) -> dict[str, Any]:
    """Config-only placeholder plan (no MachiningContext needed)."""
    mold = gts.load_mold(mold_id, client=client)
    features = gts.load_mold_features(mold_id, client=client)
    operations: list[dict[str, Any]] = []
    for feat in features:
        operations.extend(fmap.lightweight_operations_for(feat))
    return {
        "mode": "lightweight",
        "source_part": mold.get("name"),
        "mold_id": mold_id,
        "detection_version": mold.get("detection_version"),
        "n_features": len(features),
        "n_operations": len(operations),
        "operations": operations,
    }


def generate_planner_plan(
    mold_id: str,
    *,
    extents: Mapping[str, Any] | None = None,
    setup_yaml: str | Path | None = None,
    material: str | None = None,
    oversize_mm: float = 2.0,
    tool_source: str = "hardcoded",
    seq_search: str = "beam",
    setup_id: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Full plan via the existing planner. Returns the CamPlan as a JSON dict."""
    # Deferred imports: these pull in the heavy planning stack (pydantic models,
    # geometry). Keeping them here lets --mode lightweight run without them.
    from machining_context import build_context_v0
    from planner import plan as run_planner

    graph, stored_descriptor, stored_extents = gts.mold_planning_inputs(
        mold_id, client=client
    )
    extents = extents or stored_extents
    if extents is None:
        raise ValueError(
            "stock extents unavailable: pass --extents-json (or store `extents` "
            "at insert time). build_context_v0 needs a part AABB."
        )

    graph_path = _write_temp_json(graph, "_feature_graph_cascade.json")

    # Setup descriptor: prefer an explicit YAML, else the one captured at insert,
    # else fall back to descriptor generation from the graph.
    temp_paths = [graph_path]
    if setup_yaml is not None:
        setup_yaml_path = Path(setup_yaml)
        setups_source = "authored"
    elif stored_descriptor is not None:
        setup_yaml_path = _write_temp_yaml(stored_descriptor)
        temp_paths.append(setup_yaml_path)
        setups_source = "authored"
    else:
        # No descriptor available: let the context layer generate one from the graph.
        setup_yaml_path = graph_path  # unused for generation but must be a real path
        setups_source = "generated"

    try:
        context = build_context_v0(
            dict(extents),
            setup_yaml_path,
            graph_path,
            oversize_mm=oversize_mm,
            material=material,
            tool_source=tool_source,  # type: ignore[arg-type]
            setups_source=setups_source,  # type: ignore[arg-type]
            setup_id=setup_id,
        )
        cam_plan = run_planner(graph_path, context, seq_search=seq_search)  # type: ignore[arg-type]
        result = cam_plan.model_dump(mode="json")
    finally:
        for p in temp_paths:
            try:
                p.unlink()
            except OSError:
                pass

    result.setdefault("metadata", {})["mold_id"] = mold_id
    return result


def generate_plan(
    mold_id: str,
    *,
    mode: str = "planner",
    client: Any | None = None,
    **planner_kwargs: Any,
) -> dict[str, Any]:
    """Dispatch to the planner or lightweight generator."""
    if mode == "lightweight":
        return generate_lightweight_plan(mold_id, client=client)
    if mode == "planner":
        return generate_planner_plan(mold_id, client=client, **planner_kwargs)
    raise ValueError(f"unknown mode: {mode!r} (expected 'planner' or 'lightweight')")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mold_id", help="uuid of the mold to plan")
    p.add_argument("--out", help="write plan JSON here (default: stdout)")
    p.add_argument("--mode", choices=["planner", "lightweight"], default="planner")
    p.add_argument(
        "--tool-source",
        choices=["hardcoded", "directory", "supabase"],
        default="hardcoded",
        help="tool catalog source for the planner (default: built-in 8-tool set)",
    )
    p.add_argument("--material", default=None, help="work material for preset selection")
    p.add_argument("--oversize-mm", type=float, default=2.0)
    p.add_argument("--seq-search", default="beam")
    p.add_argument(
        "--extents-json",
        default=None,
        help='part AABB, e.g. \'{"min":[0,0,0],"max":[100,60,40]}\'',
    )
    p.add_argument("--setup-yaml", default=None, help="explicit setup descriptor YAML")
    p.add_argument("--setup-id", default=None, help="which setup in the descriptor to plan")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    kwargs: dict[str, Any] = {}
    if args.mode == "planner":
        extents = json.loads(args.extents_json) if args.extents_json else None
        kwargs = {
            "extents": extents,
            "setup_yaml": args.setup_yaml,
            "material": args.material,
            "oversize_mm": args.oversize_mm,
            "tool_source": args.tool_source,
            "seq_search": args.seq_search,
            "setup_id": args.setup_id,
        }
    plan = generate_plan(args.mold_id, mode=args.mode, **kwargs)
    text = json.dumps(plan, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {plan.get('n_operations', len(plan.get('operations', [])))} "
              f"operations to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
