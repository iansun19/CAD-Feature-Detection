#!/usr/bin/env python3
"""Resolve rear-only KEPT_FRONT-token features against front-graph twins + CamPlan.

Read-only. Produces eval/kept_front_resolution_96260B.md; exits 1 if any ORPHAN.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from feature_graph_viewer.geometry import triangulate_step_part  # noqa: E402
from planner import _feature_reachable_for_setup  # noqa: E402

REAR_GRAPH = REPO_ROOT / "pipeline_out/96260B_rear/feature_graph_cascade.json"
FRONT_GRAPH = REPO_ROOT / "pipeline_out/96260B_front/feature_graph_cascade.json"
REAR_STEP = REPO_ROOT / "fixtures/step/96260B_rear.stp"
FRONT_STEP = REPO_ROOT / "fixtures/step/96260B_front.stp"
PLAN_PATH = REPO_ROOT / "examples/cam_plan_96260B_rear.json"
REPORT_PATH = REPO_ROOT / "eval/kept_front_resolution_96260B.md"

AREA_TOL = 0.08
XZ_TOL_MM = 3.0
NORM_DOT_MIN = 0.85


@dataclass(frozen=True)
class FeatureGeom:
    feature_id: str
    class_name: str
    area: float | None
    surf: tuple[tuple[str, int], ...]
    centroid: tuple[float, float, float]
    normal: tuple[float, float, float]


@dataclass(frozen=True)
class Resolution:
    rear_id: str
    bucket: str
    front_twin_id: str | None
    op_id: str | None
    evidence: str


def _load_graph(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _face_map(step_path: Path) -> dict[int, dict[str, Any]]:
    faces = triangulate_step_part(step_path)
    return {int(f["face_index"]): f for f in faces}


def _feature_geom(
    node: Mapping[str, Any],
    face_map: Mapping[int, Mapping[str, Any]],
) -> FeatureGeom | None:
    params = node.get("params") or {}
    face_ids = params.get("face_indices") or node.get("face_ids") or []
    centroids: list[list[float]] = []
    normals: list[list[float]] = []
    for raw in face_ids:
        face = face_map.get(int(raw))
        if face is None:
            continue
        centroids.append(face["centroid"])
        normals.append(face["normal"])
    if not centroids:
        return None

    cx = sum(c[0] for c in centroids) / len(centroids)
    cy = sum(c[1] for c in centroids) / len(centroids)
    cz = sum(c[2] for c in centroids) / len(centroids)
    nx = sum(n[0] for n in normals) / len(normals)
    ny = sum(n[1] for n in normals) / len(normals)
    nz = sum(n[2] for n in normals) / len(normals)
    norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0

    surf_items = tuple(sorted((params.get("surface_type_histogram") or {}).items()))
    return FeatureGeom(
        feature_id=str(node["feature_id"]),
        class_name=str(node.get("class_name", "")),
        area=params.get("total_area"),
        surf=surf_items,
        centroid=(cx, cy, cz),
        normal=(nx / norm, ny / norm, nz / norm),
    )


def _enumerate_target_rear_ids(
    rear_graph: Mapping[str, Any],
    front_graph: Mapping[str, Any],
) -> list[str]:
    """Rear-only, +Z-reachable, -Z-unreachable features (the false KEPT_FRONT set)."""
    rear_ids = {str(n["feature_id"]) for n in rear_graph["nodes"]}
    front_ids = {str(n["feature_id"]) for n in front_graph["nodes"]}
    rear_only = sorted(rear_ids - front_ids, key=lambda x: int(x))
    nodes = {str(n["feature_id"]): n for n in rear_graph["nodes"]}

    out: list[str] = []
    for fid in rear_only:
        node = nodes[fid]
        if _feature_reachable_for_setup(node, "+Z") and not _feature_reachable_for_setup(
            node, "-Z"
        ):
            out.append(fid)
    return out


def _match_score(rear: FeatureGeom, front: FeatureGeom) -> float | None:
    if rear.class_name != front.class_name:
        return None
    if rear.surf != front.surf:
        return None
    if rear.area is None or front.area is None:
        return None
    rel = abs(rear.area - front.area) / max(rear.area, front.area)
    if rel > AREA_TOL:
        return None

    rx, rz = rear.centroid[0], rear.centroid[2]
    fx, fz = front.centroid[0], front.centroid[2]
    if math.hypot(rx - fx, rz - fz) > XZ_TOL_MM:
        return None

    fn = front.normal
    dots = (
        sum(a * b for a, b in zip(rear.normal, fn)),
        sum(a * b for a, b in zip(rear.normal, (-fn[0], -fn[1], -fn[2]))),
    )
    dot = max(dots)
    if dot < NORM_DOT_MIN:
        return None
    return dot


def _one_to_one_twins(
    target_ids: list[str],
    rear_geoms: dict[str, FeatureGeom],
    front_geoms: dict[str, FeatureGeom],
) -> dict[str, str]:
    pairs: list[tuple[float, str, str]] = []
    for rid in target_ids:
        rg = rear_geoms.get(rid)
        if rg is None:
            continue
        for fid, fg in front_geoms.items():
            score = _match_score(rg, fg)
            if score is not None:
                pairs.append((score, rid, fid))
    pairs.sort(reverse=True)

    used_rear: set[str] = set()
    used_front: set[str] = set()
    mapping: dict[str, str] = {}
    for _score, rid, fid in pairs:
        if rid in used_rear or fid in used_front:
            continue
        used_rear.add(rid)
        used_front.add(fid)
        mapping[rid] = fid
    return mapping


def _front_planned_ops(plan: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for op in plan.get("operations", []):
        if op.get("setup_id") != "front":
            continue
        for ref in op.get("feature_refs", []):
            out[str(ref)] = str(op["op_id"])
    return out


def _orphan_reason(
    rear: FeatureGeom,
    front_geoms: dict[str, FeatureGeom],
    front_planned: Mapping[str, str],
    twin_id: str | None,
) -> str:
    if twin_id is None:
        # Summarize nearest rejection cause.
        best: tuple[float, str, str] | None = None
        for fid, fg in front_geoms.items():
            if rear.class_name != fg.class_name:
                continue
            score = 0.0
            reasons: list[str] = []
            if rear.surf != fg.surf:
                reasons.append(f"surface_type rear={rear.surf} front={fg.surf}")
            if rear.area and fg.area:
                rel = abs(rear.area - fg.area) / max(rear.area, fg.area)
                if rel > AREA_TOL:
                    reasons.append(f"area_rel={rel:.3f}>{AREA_TOL}")
                else:
                    score += 1.0 - rel
            rx, rz = rear.centroid[0], rear.centroid[2]
            fx, fz = fg.centroid[0], fg.centroid[2]
            xz = math.hypot(rx - fx, rz - fz)
            if xz > XZ_TOL_MM:
                reasons.append(f"xz_dist={xz:.2f}mm>{XZ_TOL_MM}")
            else:
                score += 1.0
            if reasons:
                if best is None or score > best[0]:
                    best = (score, fid, "; ".join(reasons))
        if best is None:
            return "no front-graph contour candidate; rear-only geometry on split panel"
        return f"no 1:1 twin; nearest front {best[1]} rejected: {best[2]}"

    op_id = front_planned.get(twin_id)
    if op_id is None:
        return f"geometric twin front:{twin_id} exists but is in no front-setup CamPlan op"
    return f"unexpected orphan with twin front:{twin_id}"


def _resolve_all(
    target_ids: list[str],
    rear_geoms: dict[str, FeatureGeom],
    front_geoms: dict[str, FeatureGeom],
    twin_map: dict[str, str],
    front_planned: Mapping[str, str],
) -> list[Resolution]:
    rows: list[Resolution] = []
    for rid in target_ids:
        twin = twin_map.get(rid)
        if twin is not None:
            op_id = front_planned.get(twin)
            if op_id is not None:
                rows.append(
                    Resolution(
                        rear_id=rid,
                        bucket="COVERED_VIA_FRONT_TWIN",
                        front_twin_id=twin,
                        op_id=op_id,
                        evidence=(
                            f"twin front:{twin} area/norm/xz match; "
                            f"CamPlan {op_id} feature_refs includes '{twin}'"
                        ),
                    )
                )
                continue
            rows.append(
                Resolution(
                    rear_id=rid,
                    bucket="ORPHAN",
                    front_twin_id=twin,
                    op_id=None,
                    evidence=_orphan_reason(
                        rear_geoms[rid], front_geoms, front_planned, twin
                    ),
                )
            )
            continue

        rg = rear_geoms.get(rid)
        evidence = (
            _orphan_reason(rg, front_geoms, front_planned, None)
            if rg is not None
            else "missing STEP face geometry"
        )
        rows.append(
            Resolution(
                rear_id=rid,
                bucket="ORPHAN",
                front_twin_id=None,
                op_id=None,
                evidence=evidence,
            )
        )
    return rows


def _render_report(
    *,
    target_ids: list[str],
    rows: list[Resolution],
    twin_map: dict[str, str],
    covered: int,
    orphan: int,
    sanity_notes: list[str],
) -> str:
    lines: list[str] = []
    verdict = "PASS" if orphan == 0 else "FAIL"
    lines.append("# KEPT_FRONT resolution -- 96260B")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append(f"## VERDICT: **{verdict}**")
    if orphan > 0:
        lines.append("")
        lines.append(
            f"The prior coverage audit PASS was **wrong**: {orphan} rear features were "
            "labeled KEPT_FRONT from a +Z reachability token alone, but no front-setup "
            "operation machines them (no planned front twin)."
        )
    lines.append("")
    lines.append("## Enumeration")
    lines.append("")
    lines.append(
        f"Target set: rear-graph features that are **rear-only ids**, **+Z-only** "
        f"reachable (`+Z` yes, `-Z` no), and absent from the front graph node set."
    )
    lines.append("")
    lines.append(f"- **Count: {len(target_ids)}**")
    lines.append(f"- **Ids:** {', '.join(target_ids)}")
    lines.append("")
    lines.append("## Matching method")
    lines.append("")
    lines.append(
        "Twin detection uses STEP face centroids/normals (world frame), per feature:"
    )
    lines.append(f"- same `class_name`")
    lines.append(f"- same `surface_type_histogram`")
    lines.append(f"- `total_area` within {AREA_TOL:.0%}")
    lines.append(f"- centroid X/Z within {XZ_TOL_MM} mm (split-panel Y offset ignored)")
    lines.append(f"- normal dot >= {NORM_DOT_MIN} (Y-flip allowed)")
    lines.append("- greedy 1:1 assignment on match score (no shared front twin)")
    lines.append(
        "- **COVERED** only if the matched front id appears in a `setup_id: front` "
        "operation in `examples/cam_plan_96260B_rear.json`"
    )
    lines.append("")
    lines.append("## Tally")
    lines.append("")
    lines.append(f"- **COVERED_VIA_FRONT_TWIN:** {covered}")
    lines.append(f"- **ORPHAN:** {orphan}")
    lines.append("")
    lines.append("## Per-feature resolution")
    lines.append("")
    lines.append("| rear_id | bucket | front_twin | op_id | evidence |")
    lines.append("|---------|--------|------------|-------|----------|")
    for row in rows:
        lines.append(
            f"| {row.rear_id} | {row.bucket} | {row.front_twin_id or '-'} | "
            f"{row.op_id or '-'} | {row.evidence} |"
        )
    lines.append("")
    lines.append("## Sanity cross-checks")
    lines.append("")
    for note in sanity_notes:
        lines.append(f"- {note}")
    lines.append("")
    if twin_map:
        lines.append("### Twin map (1:1)")
        lines.append("")
        for rid, fid in sorted(twin_map.items(), key=lambda x: int(x[0])):
            lines.append(f"- rear {rid} -> front {fid}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    rear_graph = _load_graph(REAR_GRAPH)
    front_graph = _load_graph(FRONT_GRAPH)
    plan = _load_graph(PLAN_PATH)

    target_ids = _enumerate_target_rear_ids(rear_graph, front_graph)

    rear_face_map = _face_map(REAR_STEP)
    front_face_map = _face_map(FRONT_STEP)

    rear_geoms: dict[str, FeatureGeom] = {}
    for node in rear_graph["nodes"]:
        geom = _feature_geom(node, rear_face_map)
        if geom is not None:
            rear_geoms[geom.feature_id] = geom

    front_geoms: dict[str, FeatureGeom] = {}
    for node in front_graph["nodes"]:
        if node.get("class_name") != "contour_surface":
            continue
        geom = _feature_geom(node, front_face_map)
        if geom is not None:
            front_geoms[geom.feature_id] = geom

    twin_map = _one_to_one_twins(target_ids, rear_geoms, front_geoms)
    front_planned = _front_planned_ops(plan)
    rows = _resolve_all(target_ids, rear_geoms, front_geoms, twin_map, front_planned)

    covered = sum(1 for r in rows if r.bucket == "COVERED_VIA_FRONT_TWIN")
    orphan = sum(1 for r in rows if r.bucket == "ORPHAN")

    sanity_notes: list[str] = []
    front_twin_counts = Counter(twin_map.values())
    dup_twins = [fid for fid, n in front_twin_counts.items() if n > 1]
    sanity_notes.append(
        f"1:1 twin assignment: {len(twin_map)} pairs; "
        f"duplicate front twins: {dup_twins or 'none'}"
    )

    plan = _load_graph(PLAN_PATH)
    op130 = next(
        (op for op in plan["operations"] if op.get("op_id") == "OP130"),
        None,
    )
    if op130 is not None:
        op130_refs = {str(x) for x in op130.get("feature_refs", [])}
        covered_twins = {r.front_twin_id for r in rows if r.bucket == "COVERED_VIA_FRONT_TWIN"}
        missing = sorted(covered_twins - op130_refs)
        sanity_notes.append(
            f"CamPlan OP130 (front surface_finish) refs: {sorted(op130_refs, key=int)}"
        )
        if missing:
            sanity_notes.append(f"WARNING: covered twins missing from OP130: {missing}")
        else:
            sanity_notes.append(
                "All COVERED twins are present in OP130 feature_refs (verified in JSON)."
            )

    orphan_ids = [r.rear_id for r in rows if r.bucket == "ORPHAN"]
    sanity_notes.append(
        f"ORPHAN ids ({len(orphan_ids)}): {', '.join(orphan_ids)}"
    )

    report = _render_report(
        target_ids=target_ids,
        rows=rows,
        twin_map=twin_map,
        covered=covered,
        orphan=orphan,
        sanity_notes=sanity_notes,
    )
    REPORT_PATH.write_text(report)
    print(f"Wrote {REPORT_PATH}")
    print(f"Enumerated: {len(target_ids)} | COVERED: {covered} | ORPHAN: {orphan}")
    print(f"VERDICT: {'PASS' if orphan == 0 else 'FAIL'}")
    return 0 if orphan == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
