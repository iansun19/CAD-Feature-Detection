"""One-shot GT -> live CamPlan: persist, verify round-trip, plan, report.

Orchestration wrapper only. Reuses (does not modify):
  * env_bootstrap        - loads repo-root .env at import (SUPABASE_URL etc.)
  * ground_truth_store   - insert / reconstruct against Supabase
  * generate_operation_plan.generate_planner_plan - the planner path

Steps: preflight env + tool catalog -> persist GT (idempotent on the
(name, detection_version) baseline) -> verify a live round-trip is byte-identical
-> run the planner -> write CamPlan + print a coverage summary that always lists
features outside planner scope (e.g. inner_fillet).

    python seed_and_plan.py --machining-side back
    python seed_and_plan.py --gt-path other.json --setup-id rear \
        --detection-version cascade-v6 --tool-source supabase --out plan.json --force

Run with the mlcad env python: ~/miniconda3/envs/mlcad/bin/python seed_and_plan.py ...
"""
from __future__ import annotations

# Import first: env_bootstrap loads the repo-root .env into os.environ at import
# time (existing shell vars win), so module-level Supabase client init downstream
# never reads empty vars. This is the repo's own mechanism -- no new one added.
import env_bootstrap  # noqa: F401  (side effect: load .env)

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import planning.ground_truth_store as gts
import generate_operation_plan as gop

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GT = REPO_ROOT / "pipeline_out" / "fish_mold_cascade" / "feature_graph_cascade.json"


class PreflightError(RuntimeError):
    """User-facing failure raised before any deep planner traceback."""


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def check_env() -> None:
    """Confirm the service-role credentials needed for writes are present."""
    missing = [
        name
        for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        raise PreflightError(
            "missing required env var(s): "
            + ", ".join(missing)
            + f"\n  Set them in {REPO_ROOT / '.env'} (loaded automatically) or export them. "
            "SUPABASE_SERVICE_ROLE_KEY is required for RLS-protected writes."
        )


def check_tool_catalog(client: Any, tool_source: str) -> None:
    """Warn loudly if planning with --tool-source supabase but the catalog is empty."""
    if tool_source != "supabase":
        return
    try:
        resp = client.table("tools").select("*", count="exact").limit(1).execute()
        count = int(resp.count or 0)
    except Exception as exc:  # pragma: no cover - network/permission dependent
        print(f"WARNING: could not count the Supabase tools table: {exc}", file=sys.stderr)
        return
    if count == 0:
        print(
            "WARNING: Supabase `tools` table is EMPTY. --tool-source supabase will "
            "produce misleadingly thin plans (no fitting tools). Seed it with "
            "scripts/ingest_tool_libraries.py or use --tool-source hardcoded.",
            file=sys.stderr,
        )
    else:
        print(f"Preflight: Supabase tools catalog has {count} tools.")


# --------------------------------------------------------------------------- #
# Setup descriptor / extents (planning inputs)
# --------------------------------------------------------------------------- #
def load_sibling_descriptor(gt_path: Path) -> dict[str, Any] | None:
    """Load setup_descriptor.yaml sitting next to the GT graph, if present."""
    candidate = gt_path.parent / "setup_descriptor.yaml"
    if not candidate.is_file():
        return None
    import yaml

    return yaml.safe_load(candidate.read_text(encoding="utf-8"))


def resolve_step_path(descriptor: Mapping[str, Any] | None, setup_id: str) -> Path | None:
    """Find the STEP referenced by the chosen setup (for extents computation)."""
    refs: list[str] = []
    if descriptor:
        entry = (descriptor.get("setups") or {}).get(setup_id) or {}
        if entry.get("part_step"):
            refs.append(str(entry["part_step"]))
    for ref in refs:
        for base in (REPO_ROOT, Path.cwd()):
            p = base / ref
            if p.is_file():
                return p
        if Path(ref).is_file():
            return Path(ref)
    return None


def compute_extents(step_path: Path) -> dict[str, list[float]]:
    """Part AABB via the existing OCC-backed helper (mlcad env required)."""
    from planning.machining_context import _envelope_corners

    mn, mx = _envelope_corners(str(step_path))
    return {"min": list(mn), "max": list(mx)}


def graph_needs_reachability_setup(graph: Mapping[str, Any]) -> bool:
    """True if the planner will engage reachability scoping (needs machining_side)."""
    from planner import _graph_has_verified_reachability

    return _graph_has_verified_reachability(graph)


def guard_planning_inputs(
    graph: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None,
    setup_id: str,
    machining_side: str | None,
) -> None:
    """Fail early, with actionable messages, on the fish_mold-shaped requirements."""
    if not graph_needs_reachability_setup(graph):
        return  # no reachability -> class-scope planning, no side/opening-axis needed
    part = str(graph.get("part_id") or "this part")
    if descriptor is None:
        raise PreflightError(
            f"{part}: graph has verified reachability, so the planner needs a setup "
            "descriptor with an opening axis, but no setup_descriptor.yaml was found "
            "next to the GT graph."
        )
    setups = descriptor.get("setups") or {}
    if setup_id not in setups:
        available = ", ".join(sorted(setups)) or "(none)"
        raise PreflightError(
            f"{part}: --setup-id '{setup_id}' is not in the descriptor "
            f"(available: {available}). fish_mold requires --setup-id rear; the "
            "'default' setup has no opening axis."
        )
    if not machining_side:
        raise PreflightError(
            f"{part}: graph uses verified reachability, so the planner needs "
            "--machining-side (front|back). For fish_mold's rear setup use "
            "--machining-side back."
        )


# --------------------------------------------------------------------------- #
# Persist / verify / plan
# --------------------------------------------------------------------------- #
def persist(
    client: Any,
    graph: Mapping[str, Any],
    *,
    name: str,
    detection_version: str,
    step_file_ref: str | None,
    descriptor: Mapping[str, Any] | None,
    extents: Mapping[str, Any] | None,
    force: bool,
) -> tuple[str, bool]:
    """Insert (or reuse) a mold. Returns (mold_id, reused)."""
    existing = gts.find_mold(name, detection_version, client=client)
    if existing and not force:
        return str(existing["id"]), True
    if existing and force:
        client.table("molds").delete().eq("id", existing["id"]).execute()
    mold_id = gts.insert_mold_with_features(
        graph,
        name=name,
        detection_version=detection_version,
        step_file_ref=step_file_ref,
        setup_descriptor=descriptor,
        extents=extents,
        client=client,
    )
    return mold_id, False


def verify_round_trip(client: Any, mold_id: str, graph: Mapping[str, Any]) -> None:
    """Reconstruct from the live DB and assert node-identical to the source graph."""
    recon = gts.reconstruct_feature_graph(mold_id, client=client)
    src_nodes = list(graph.get("nodes") or [])
    got_nodes = list(recon.get("nodes") or [])
    if got_nodes == src_nodes:
        return
    # Build an actionable diff summary rather than a bare assert.
    lines = [
        f"round-trip mismatch for mold {mold_id}: "
        f"{len(src_nodes)} source nodes vs {len(got_nodes)} reconstructed."
    ]
    for i, src in enumerate(src_nodes):
        got = got_nodes[i] if i < len(got_nodes) else None
        if got != src:
            fid = src.get("feature_id")
            if got is None:
                lines.append(f"  first missing at index {i} (feature_id={fid})")
            else:
                diff_keys = sorted(
                    k for k in set(src) | set(got) if src.get(k) != got.get(k)
                )
                lines.append(
                    f"  first differing node index {i} (feature_id={fid}); "
                    f"keys differ: {diff_keys}"
                )
            break
    raise PreflightError(
        "\n".join(lines)
        + "\n  This usually means the live write did not persist as sent "
        "(check SUPABASE_SERVICE_ROLE_KEY / RLS write policy)."
    )


def run_planner(
    client: Any,
    mold_id: str,
    *,
    extents: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None,
    setup_id: str,
    machining_side: str | None,
    tool_source: str,
    material: str | None,
) -> dict[str, Any]:
    """Drive generate_operation_plan.generate_planner_plan via a temp setup YAML
    carrying the machining_side process input (keeps planner/persistence untouched)."""
    temp_yaml: Path | None = None
    try:
        setup_yaml_arg: str | None = None
        if descriptor is not None and machining_side is not None:
            import yaml

            patched = json.loads(json.dumps(descriptor))  # deep copy
            patched.setdefault("setups", {}).setdefault(setup_id, {})[
                "machining_side"
            ] = machining_side
            fd = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            )
            with fd as fh:
                yaml.safe_dump(patched, fh, sort_keys=False)
            temp_yaml = Path(fd.name)
            setup_yaml_arg = str(temp_yaml)

        return gop.generate_planner_plan(
            mold_id,
            extents=dict(extents),
            setup_yaml=setup_yaml_arg,
            setup_id=setup_id,
            tool_source=tool_source,
            material=material,
            client=client,
        )
    finally:
        if temp_yaml is not None:
            try:
                temp_yaml.unlink()
            except OSError:
                pass


def coverage_gaps(
    graph: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (unmapped_by_class, uncovered_in_plan).

    unmapped_by_class : GT feature whose class the planner has no op mapping for
                        (class not in planner scope). This is the "no rule exists"
                        gap. Empty for fish_mold in the current planner.
    uncovered_in_plan : GT feature not referenced by any produced operation. Superset
                        that also captures reachability/setup drops. Each carries a
                        `reason` so a class-unmapped gap is not confused with a
                        this-setup-only drop. inner_fillet shows up here (dropped by
                        rear-setup reachability), so the gap is never silent.
    """
    from planner import PLANNER_FEATURE_CLASSES

    covered: set[str] = set()
    for op in plan.get("operations") or []:
        for ref in op.get("feature_refs") or []:
            covered.add(str(ref))

    unmapped_by_class: list[dict[str, Any]] = []
    uncovered_in_plan: list[dict[str, Any]] = []
    for node in graph.get("nodes") or []:
        fid = node.get("feature_id")
        cls = node.get("class_name")
        in_scope = cls in PLANNER_FEATURE_CLASSES
        if not in_scope:
            unmapped_by_class.append({"feature_id": fid, "class_name": cls})
        if str(fid) not in covered:
            uncovered_in_plan.append(
                {
                    "feature_id": fid,
                    "class_name": cls,
                    "reason": "class-unmapped" if not in_scope else "no-op-in-plan (setup/reachability scope)",
                }
            )
    return unmapped_by_class, uncovered_in_plan


def report(graph: Mapping[str, Any], plan: Mapping[str, Any], out_path: Path) -> None:
    ops = list(plan.get("operations") or [])
    out_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"\nWrote CamPlan -> {out_path}")
    print(f"Operations: {len(ops)}")
    print("Sequenced op types: " + " -> ".join(o.get("operation", "?") for o in ops))

    unmapped_by_class, uncovered = coverage_gaps(graph, plan)
    n_features = len(graph.get("nodes") or [])
    print(
        f"\nCoverage: {n_features - len(uncovered)}/{n_features} GT features covered "
        f"by >=1 operation."
    )
    # Class-unmapped is the real "no rule exists" gap -> enumerate fully (should be
    # small/empty). It is also the one worth acting on in the mapping config.
    if unmapped_by_class:
        print(f"Unmapped by class (no planner rule) [{len(unmapped_by_class)}]:")
        for f in unmapped_by_class:
            print(f"  feature_id={f['feature_id']}  class_name={f['class_name']}")
    else:
        print("Unmapped by class (no planner rule): none — every GT class is in planner scope.")

    # Not-covered is dominated by setup/reachability scope drops (expected for a
    # single setup) -> collapse to a per-(class, reason) summary with counts and a
    # capped feature_id sample, so the signal is scannable instead of a wall of ids.
    if uncovered:
        print(f"Not covered in this plan [{len(uncovered)}], by class:")
        groups: dict[tuple[str, str], list[Any]] = {}
        for f in uncovered:
            groups.setdefault((str(f["class_name"]), f["reason"]), []).append(f["feature_id"])
        for (cls, reason), ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            ids_sorted = sorted(ids, key=lambda x: (x is None, x))
            shown = ", ".join(str(i) for i in ids_sorted[:10])
            more = f", +{len(ids_sorted) - 10} more" if len(ids_sorted) > 10 else ""
            print(f"  {cls} x{len(ids_sorted)}  reason={reason}")
            print(f"      feature_ids: {shown}{more}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    check_env()

    gt_path = Path(args.gt_path)
    if not gt_path.is_file():
        raise PreflightError(f"GT graph not found: {gt_path}")
    graph = json.loads(gt_path.read_text(encoding="utf-8"))

    name = str(graph.get("part_id") or gt_path.stem)
    detection_version = args.detection_version or f"cascade-v{graph.get('schema_version', '0')}"
    descriptor = load_sibling_descriptor(gt_path)

    # Fail early on missing planning prerequisites (no deep tracebacks).
    guard_planning_inputs(graph, descriptor, args.setup_id, args.machining_side)

    step_path = resolve_step_path(descriptor, args.setup_id)
    if step_path is None:
        raise PreflightError(
            f"could not locate the STEP file for setup '{args.setup_id}' (needed for "
            "stock extents). Expected the descriptor's part_step to resolve under the "
            "repo root."
        )
    extents = compute_extents(step_path)

    sb = gts.create_supabase_client(client)
    check_tool_catalog(sb, args.tool_source)

    mold_id, reused = persist(
        sb,
        graph,
        name=name,
        detection_version=detection_version,
        step_file_ref=step_path.name,
        descriptor=descriptor,
        extents=extents,
        force=args.force,
    )
    print(
        f"{'Reused existing' if reused else 'Inserted'} mold_id={mold_id} "
        f"(name={name!r}, detection_version={detection_version!r})"
    )

    verify_round_trip(sb, mold_id, graph)
    print("Round-trip: OK (reconstructed graph is node-identical to source)")

    plan = run_planner(
        sb,
        mold_id,
        extents=extents,
        descriptor=descriptor,
        setup_id=args.setup_id,
        machining_side=args.machining_side,
        tool_source=args.tool_source,
        material=args.material,
    )
    report(graph, plan, Path(args.out))
    return plan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-path", default=str(DEFAULT_GT), help="validated GT feature graph JSON")
    p.add_argument("--detection-version", default=None, help="baseline tag (default: cascade-v<schema_version>)")
    p.add_argument("--setup-id", default="rear", help="setup in the descriptor to plan (default: rear)")
    p.add_argument("--machining-side", default=None, help="front|back process input (fish_mold rear -> back)")
    p.add_argument("--tool-source", default="supabase", choices=["hardcoded", "directory", "supabase"])
    p.add_argument("--material", default=None, help="work material for preset selection")
    p.add_argument("--out", default="plan.json", help="CamPlan output path")
    p.add_argument("--force", action="store_true", help="replace an existing (name, detection_version) mold")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    try:
        run(args)
    except PreflightError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
