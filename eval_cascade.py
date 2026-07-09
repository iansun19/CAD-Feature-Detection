"""eval_cascade.py — cascade vs Toolpath trust evaluation harness.

Measures how much to trust the Toolpath-aligned cascade beyond the reference plate.
Scores discrete classes (pocket/hole/flat) via attribute matching; residual classes
(contour/fillet/profile/wall) via face-coverage only. Reference plate gets rigorous
face-level IoU against hand-established face truth.

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python eval_cascade.py
  /Users/iansun19/miniconda3/envs/mlcad/bin/python eval_cascade.py eval/gt/96260B_plate.yaml
  /Users/iansun19/miniconda3/envs/mlcad/bin/python eval_cascade.py --gt-dir eval/gt
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_instances import face_iou
from run_cascade import run_cascade, _load_edges
from pocket_detection import PocketDetectionConfig, pocket_config_from_setup_dict

MM_PER_IN = 25.4
REPO_ROOT = Path(__file__).resolve().parent

# Toolpath vocabulary
DISCRETE_TP_CLASSES = (
    "filleted_pocket",
    "filleted_open_pocket",
    "through_hole",
    "filleted_blind_hole",
    "flat",
    "open_pocket",
)
RESIDUAL_TP_CLASSES = (
    "contour_surface",
    "wall",
)
WIRED_TP_CLASSES = ("outer_fillet", "wall", "profile")
ALL_TP_COUNT_CLASSES = DISCRETE_TP_CLASSES + WIRED_TP_CLASSES + RESIDUAL_TP_CLASSES

CASCADE_KIND_TO_TP: dict[str, str] = {
    "pocket": "filleted_pocket",
    "through_hole": "through_hole",
    "blind_hole": "filleted_blind_hole",
    "flat": "flat",
    "outer_fillet": "outer_fillet",
}

# Attribute-match tolerances (inches)
TOL_DEPTH_IN = 0.02
TOL_FILLET_IN = 0.01
TOL_DIAMETER_IN = 0.01
IOU_WARN_THRESHOLD = 0.95
SLIVER_AREA_MM2 = 0.01


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML required for eval_cascade.py (pip install pyyaml)."
        ) from exc
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"GT yaml root must be a mapping: {path}")
    return data


def _resolve_path(raw: str, gt_path: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (REPO_ROOT, gt_path.parent):
        candidate = (base / p).resolve()
        if candidate.is_file():
            return candidate
    return (REPO_ROOT / p).resolve()


def _parse_diameter_from_wall_key(key: str) -> float | None:
    m = re.search(r"([\d.]+)", key.replace("⌀", "").replace("in", ""))
    return float(m.group(1)) if m else None


def _primary_wall_diameter_in(wall_diameters: dict[str, int]) -> float | None:
    """Return the dominant (highest-count) wall diameter in inches."""
    best_d: float | None = None
    best_count = -1
    for k, count in wall_diameters.items():
        d = _parse_diameter_from_wall_key(k)
        if d is None:
            continue
        if count > best_count or (count == best_count and best_d is not None and d < best_d):
            best_d = d
            best_count = count
    return best_d


def _norm_bool(val: Any) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return bool(val)


@dataclass
class CascadeInstance:
    tp_class: str
    face_indices: frozenset[int]
    depth_below_top_in: float | None = None
    fillet_radius_in: float | None = None
    three_d_surface: bool | None = None
    diameter_in: float | None = None
    feature_depth_in: float | None = None
    ld: float | None = None
    cascade_kind: str = ""


@dataclass
class HardDimResult:
    name: str
    cascade_val: Any
    gt_val: Any
    ok: bool
    delta: float | None = None


@dataclass
class FeatureMatch:
    cascade_idx: int
    gt_idx: int | None
    matched: bool
    hard_dims: list[HardDimResult] = field(default_factory=list)
    info: dict[str, float | None] = field(default_factory=dict)


@dataclass
class PartReport:
    part_id: str
    gt_path: Path
    n_faces: int
    cascade_completed: bool
    unclaimed_faces: int
    count_rows: list[dict[str, Any]]
    count_divergences: list[tuple[str, int, int]]
    zero_pass_flags: list[str]
    brep_flags: dict[str, int]
    discrete_match: dict[str, dict[str, Any]]
    residual_note: str
    residual_coverage: float | None
    iou_rows: list[dict[str, Any]]
    iou_mean_by_class: dict[str, float]
    iou_below_threshold: list[tuple[str, int, float]]
    error: str | None = None


def _extract_cascade_instances(
    pocket_result,
    hole_result,
    coaxial_result,
    flat_result,
    outer_fillet_result,
    wall_result,
    profile_result,
    residual_result,
) -> tuple[list[CascadeInstance], int, set[int]]:
    """Return discrete instances, residual instance count, all residual face indices."""
    out: list[CascadeInstance] = []

    for feat in pocket_result.features:
        out.append(CascadeInstance(
            tp_class=feat.toolpath_class,
            cascade_kind="pocket",
            face_indices=frozenset(feat.face_indices),
            depth_below_top_in=(
                feat.depth_below_top_mm / MM_PER_IN
                if feat.depth_below_top_mm is not None else None
            ),
            fillet_radius_in=(
                feat.fillet_radius_mm / MM_PER_IN
                if feat.fillet_radius_mm is not None else None
            ),
            three_d_surface=feat.surface_3d,
            diameter_in=_primary_wall_diameter_in(feat.wall_diameters),
        ))

    for feat in hole_result.features:
        tp = CASCADE_KIND_TO_TP.get(feat.kind, feat.kind)
        out.append(CascadeInstance(
            tp_class=tp,
            cascade_kind=feat.kind,
            face_indices=frozenset(feat.face_indices),
            diameter_in=feat.nominal_diameter / MM_PER_IN,
            depth_below_top_in=(
                feat.depth / MM_PER_IN if feat.depth is not None else None
            ),
            three_d_surface=False,
        ))

    for feat in coaxial_result.features:
        out.append(CascadeInstance(
            tp_class=feat.toolpath_class,
            cascade_kind=feat.kind,
            face_indices=frozenset(feat.face_indices),
            three_d_surface=False,
        ))

    for feat in flat_result.features:
        out.append(CascadeInstance(
            tp_class="flat",
            cascade_kind="flat",
            face_indices=frozenset(feat.face_indices),
            three_d_surface=False,
        ))

    for feat in outer_fillet_result.features:
        out.append(CascadeInstance(
            tp_class="outer_fillet",
            cascade_kind="outer_fillet",
            face_indices=frozenset(feat.face_indices),
            three_d_surface=False,
        ))

    for feat in wall_result.features:
        out.append(CascadeInstance(
            tp_class="wall",
            cascade_kind="wall",
            face_indices=frozenset(feat.face_indices),
            diameter_in=feat.nominal_diameter_mm / MM_PER_IN,
            three_d_surface=False,
        ))

    for feat in profile_result.features:
        out.append(CascadeInstance(
            tp_class="profile",
            cascade_kind="profile",
            face_indices=frozenset(feat.face_indices),
            diameter_in=feat.nominal_diameter_mm / MM_PER_IN,
            three_d_surface=False,
        ))

    residual_faces: set[int] = set()
    for feat in residual_result.features:
        out.append(CascadeInstance(
            tp_class="contour_surface",
            cascade_kind=feat.kind,
            face_indices=frozenset(feat.face_indices),
            three_d_surface=feat.params.get("3D_surface"),
        ))
        residual_faces |= feat.face_indices

    return out, len(residual_result.features), residual_faces


# Toolpath class_name -> taxonomy class_id (for cascade feature graph export).
CASCADE_TP_CLASS_ID: dict[str, int] = {
    "filleted_pocket": 6,
    "filleted_open_pocket": 6,
    "open_pocket": 6,
    "pocket": 6,
    "filleted_blind_hole": 5,
    "through_hole": 0,
    "flat": 11,
    "outer_fillet": 10,
    "inner_fillet": 10,
    "wall": -1,
    "profile": -1,
    "contour_surface": -1,
}


def build_cascade_feature_graph(
    part_id: str,
    n_faces: int,
    pocket_result,
    hole_result,
    coaxial_result,
    flat_result,
    outer_fillet_result,
    wall_result,
    profile_result,
    residual_result,
    edge_index: np.ndarray,
    faces: Sequence[Any] | None = None,
    opening_axis: Sequence[float] | None = None,
    *,
    inner_fillet_result: Any | None = None,
) -> dict[str, Any]:
    """Build feature_graph_cascade.json from cascade pass results.

    When ``faces`` and ``opening_axis`` are supplied, each node is annotated with
    a 3-axis tool-approach direction (``node['approach']``) and the graph gains an
    ``approach_frame`` block; ``schema_version`` bumps to 3. Callers that omit
    them get the byte-identical v2 output.
    """
    from feature_graph import feature_adjacency

    nodes: list[dict[str, Any]] = []
    face_to_feature: dict[int, int] = {}

    def _append_node(class_name: str, face_ids: set[int] | frozenset[int], params: dict) -> None:
        fid = len(nodes)
        ids = sorted(int(i) for i in face_ids)
        for i in ids:
            face_to_feature[i] = fid
        nodes.append({
            "feature_id": fid,
            "class_id": CASCADE_TP_CLASS_ID.get(class_name, -1),
            "class_name": class_name,
            "face_ids": ids,
            "n_faces": len(ids),
            "mean_confidence": 1.0,
            "params": params,
        })

    if inner_fillet_result is not None:
        for feat in inner_fillet_result.features:
            _append_node("inner_fillet", feat.face_indices, feat.to_dict())

    for feat in pocket_result.features:
        _append_node(feat.toolpath_class, feat.face_indices, feat.to_dict())

    for feat in hole_result.features:
        tp = CASCADE_KIND_TO_TP.get(feat.kind, feat.kind)
        _append_node(tp, feat.face_indices, feat.to_dict())

    for feat in coaxial_result.features:
        _append_node(feat.toolpath_class, feat.face_indices, feat.to_dict())

    for feat in flat_result.features:
        _append_node("flat", feat.face_indices, feat.to_dict())

    for feat in outer_fillet_result.features:
        _append_node("outer_fillet", feat.face_indices, feat.to_dict())

    for feat in wall_result.features:
        _append_node("wall", feat.face_indices, feat.to_dict())

    for feat in profile_result.features:
        _append_node("profile", feat.face_indices, feat.to_dict())

    for feat in residual_result.features:
        _append_node("contour_surface", feat.face_indices, feat.to_dict())

    # Feature–feature adjacency from shared B-rep edges.
    class _Inst:
        def __init__(self, face_ids):
            self.face_ids = face_ids

    instances = [_Inst(n["face_ids"]) for n in nodes]
    edges = feature_adjacency(instances, edge_index, n_faces)

    graph: dict[str, Any] = {
        "schema_version": 2,
        "part_id": part_id,
        "source": "cascade",
        "n_faces": n_faces,
        "n_features": len(nodes),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }

    if faces is not None and opening_axis is not None:
        from approach_vectors import annotate_approach_vectors

        graph["approach_frame"] = annotate_approach_vectors(
            nodes, faces=faces, opening_axis=opening_axis,
        )
        graph["schema_version"] = 3

    return graph


def _count_by_tp(instances: Sequence[CascadeInstance]) -> dict[str, int]:
    counts = {c: 0 for c in ALL_TP_COUNT_CLASSES}
    for inst in instances:
        if inst.tp_class in counts:
            counts[inst.tp_class] += 1
    return counts


def _residual_gt_sum(counts: dict[str, Any]) -> int:
    return int(counts.get("contour_surface", 0) or 0)


def _compare_hard_dims(
    cascade: CascadeInstance,
    gt_feat: dict[str, Any],
) -> tuple[list[HardDimResult], bool]:
    """Compare HARD match dimensions; return (dim_results, all_ok)."""
    dims: list[HardDimResult] = []
    all_ok = True

    gt_depth = gt_feat.get("depth_below_top")
    if gt_depth is not None:
        c_val = cascade.depth_below_top_in
        ok = (
            c_val is not None
            and abs(c_val - float(gt_depth)) <= TOL_DEPTH_IN
        )
        delta = (c_val - float(gt_depth)) if c_val is not None else None
        dims.append(HardDimResult("depth_below_top", c_val, float(gt_depth), ok, delta))
        all_ok &= ok

    gt_fillet = gt_feat.get("fillet_radius")
    if gt_fillet is not None:
        c_val = cascade.fillet_radius_in
        ok = (
            c_val is not None
            and abs(c_val - float(gt_fillet)) <= TOL_FILLET_IN
        )
        delta = (c_val - float(gt_fillet)) if c_val is not None else None
        dims.append(HardDimResult("fillet_radius", c_val, float(gt_fillet), ok, delta))
        all_ok &= ok

    gt_3d = gt_feat.get("3d_surface")
    if gt_3d is not None:
        c_val = cascade.three_d_surface
        gt_bool = _norm_bool(gt_3d)
        ok = c_val is not None and gt_bool is not None and c_val == gt_bool
        dims.append(HardDimResult("3d_surface", c_val, gt_bool, ok))
        all_ok &= ok

    gt_dia = gt_feat.get("diameter")
    if gt_dia is not None:
        c_val = cascade.diameter_in
        ok = (
            c_val is not None
            and abs(c_val - float(gt_dia)) <= TOL_DIAMETER_IN
        )
        delta = (c_val - float(gt_dia)) if c_val is not None else None
        dims.append(HardDimResult("diameter", c_val, float(gt_dia), ok, delta))
        all_ok &= ok

    return dims, all_ok


def _attribute_distance(cascade: CascadeInstance, gt_feat: dict[str, Any]) -> float:
    """Lower is better; inf if any present hard dim is out of tolerance."""
    dist = 0.0
    n_terms = 0

    gt_depth = gt_feat.get("depth_below_top")
    if gt_depth is not None and cascade.depth_below_top_in is not None:
        d = abs(cascade.depth_below_top_in - float(gt_depth))
        if d > TOL_DEPTH_IN:
            return float("inf")
        dist += d / max(TOL_DEPTH_IN, 1e-9)
        n_terms += 1

    gt_fillet = gt_feat.get("fillet_radius")
    if gt_fillet is not None and cascade.fillet_radius_in is not None:
        d = abs(cascade.fillet_radius_in - float(gt_fillet))
        if d > TOL_FILLET_IN:
            return float("inf")
        dist += d / max(TOL_FILLET_IN, 1e-9)
        n_terms += 1

    gt_3d = gt_feat.get("3d_surface")
    if gt_3d is not None and cascade.three_d_surface is not None:
        if cascade.three_d_surface != _norm_bool(gt_3d):
            return float("inf")
        n_terms += 1

    gt_dia = gt_feat.get("diameter")
    if gt_dia is not None and cascade.diameter_in is not None:
        d = abs(cascade.diameter_in - float(gt_dia))
        if d > TOL_DIAMETER_IN:
            return float("inf")
        dist += d / max(TOL_DIAMETER_IN, 1e-9)
        n_terms += 1

    return dist / max(n_terms, 1)


def _match_discrete_attributes(
    cascade_instances: Sequence[CascadeInstance],
    gt_features: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Greedy per-class attribute matching for discrete Toolpath classes.

    GT feature entries are templates — one panel row may represent many identical
    cascade instances (e.g. 7 near-identical pockets). Each cascade instance is
    matched independently against the closest compatible GT template.
    """
    by_class: dict[str, list[CascadeInstance]] = {}
    for inst in cascade_instances:
        by_class.setdefault(inst.tp_class, []).append(inst)

    gt_by_class: dict[str, list[dict[str, Any]]] = {}
    for feat in gt_features:
        cls = feat.get("class")
        if cls:
            gt_by_class.setdefault(str(cls), []).append(feat)

    results: dict[str, dict[str, Any]] = {}
    for cls in DISCRETE_TP_CLASSES:
        c_list = by_class.get(cls, [])
        g_list = gt_by_class.get(cls, [])
        if not g_list:
            results[cls] = {
                "mode": "counts_only",
                "cascade_n": len(c_list),
                "gt_n": None,
                "matches": [],
            }
            continue

        matches: list[FeatureMatch] = []
        matched_gt_indices: set[int] = set()

        for ci, c_inst in enumerate(c_list):
            best_gi: int | None = None
            best_dist = float("inf")
            best_dims: list[HardDimResult] = []
            best_ok = False

            for gi, g_feat in enumerate(g_list):
                dist = _attribute_distance(c_inst, g_feat)
                dims, ok = _compare_hard_dims(c_inst, g_feat)
                if dist < best_dist:
                    best_dist = dist
                    best_gi = gi
                    best_dims = dims
                    best_ok = ok and dist < float("inf")

            info: dict[str, float | None] = {}
            if best_gi is not None:
                g_feat = g_list[best_gi]
                if g_feat.get("feature_depth") is not None and c_inst.feature_depth_in is not None:
                    info["feature_depth_delta_in"] = (
                        c_inst.feature_depth_in - float(g_feat["feature_depth"])
                    )
                if g_feat.get("ld") is not None and c_inst.ld is not None:
                    info["ld_delta"] = c_inst.ld - float(g_feat["ld"])
                if best_ok:
                    matched_gt_indices.add(best_gi)

            matches.append(FeatureMatch(
                cascade_idx=ci,
                gt_idx=best_gi,
                matched=best_ok,
                hard_dims=best_dims,
                info=info,
            ))

        matched_n = sum(1 for m in matches if m.matched)
        cascade_only = len(c_list) - matched_n
        # gt-only: template rows with no matching cascade instance at all
        gt_only = sum(
            1 for gi, g_feat in enumerate(g_list)
            if gi not in matched_gt_indices
            and not any(
                _attribute_distance(c, g_feat) < float("inf")
                and _compare_hard_dims(c, g_feat)[1]
                for c in c_list
            )
        )

        results[cls] = {
            "mode": "attribute",
            "cascade_n": len(c_list),
            "gt_n": len(g_list),
            "matched": matched_n,
            "cascade_only": cascade_only,
            "gt_only": gt_only,
            "match_rate": matched_n / max(len(c_list), 1),
            "matches": matches,
        }

    return results


def _brep_quality_flags(faces, pocket_result, hole_result, flat_result) -> dict[str, int]:
    radius_types = {"cylinder", "sphere", "torus"}
    orphan_sculpted_in_pocket_claims = 0
    bspline_in_discrete = sum(
        1 for i in (hole_result.claimed_faces | flat_result.claimed_faces)
        if faces[i].surface_type == "bspline"
    )
    for feat in pocket_result.features:
        expected_sculpted = feat.floor_count + feat.sphere_count
        claimed_sculpted = sum(
            1 for i in feat.face_indices
            if faces[i].surface_type in ("bspline", "bezier", "sphere")
        )
        orphan_sculpted_in_pocket_claims += max(0, claimed_sculpted - expected_sculpted)
        claimed_bspline = sum(
            1 for i in feat.face_indices
            if faces[i].surface_type == "bspline"
        )
        bspline_in_discrete += max(0, claimed_bspline - feat.floor_count)
    zero_radius = sum(
        1 for f in faces
        if f.surface_type in radius_types
        and f.radius is not None
        and float(f.radius) <= 1e-9
    )
    extraction_fail = sum(
        1 for f in faces
        if f.surface_type in radius_types and f.radius is None
    )
    sliver = sum(1 for f in faces if float(f.area) < SLIVER_AREA_MM2)

    return {
        "orphan_sculpted_in_pocket_claims": orphan_sculpted_in_pocket_claims,
        "bspline_in_discrete_claims": bspline_in_discrete,
        "zero_radius_faces": zero_radius,
        "radius_extraction_failures": extraction_fail,
        "sliver_faces": sliver,
    }


def _residual_coverage(
    residual_faces: set[int],
    face_truth: dict[str, Any] | None,
) -> tuple[str, float | None]:
    if not face_truth:
        return "coarse by design — count-level only (no face_truth for residual classes)", None

    gt_contour_faces: set[int] = set()
    for cls in RESIDUAL_TP_CLASSES:
        for entry in face_truth.get(cls, []) or []:
            gt_contour_faces |= set(entry.get("faces") or [])

    if not gt_contour_faces:
        return "coarse by design — no residual face_truth provided", None

    covered = gt_contour_faces & residual_faces
    frac = len(covered) / max(len(gt_contour_faces), 1)
    note = (
        f"residual pool covers {len(covered)}/{len(gt_contour_faces)} "
        f"gt-residual faces ({frac:.1%})"
    )
    return note, frac


def _face_iou_baseline(
    cascade_instances: Sequence[CascadeInstance],
    face_truth: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float], list[tuple[str, int, float]]]:
    """Per-feature IoU between cascade claims and hand-established face truth."""
    rows: list[dict[str, Any]] = []
    ious_by_class: dict[str, list[float]] = {}
    below: list[tuple[str, int, float]] = []

    for cls, gt_entries in face_truth.items():
        gt_sets = [frozenset(e.get("faces") or []) for e in (gt_entries or [])]
        c_list = [i for i in cascade_instances if i.tp_class == cls]
        used_c: set[int] = set()

        for gi, gt_faces in enumerate(gt_sets):
            best_ci: int | None = None
            best_iou = -1.0
            for ci, c_inst in enumerate(c_list):
                if ci in used_c:
                    continue
                iou = face_iou(c_inst.face_indices, gt_faces)
                if iou > best_iou:
                    best_iou = iou
                    best_ci = ci

            if best_ci is not None and best_iou > 0:
                used_c.add(best_ci)
                c_faces = c_list[best_ci].face_indices
            else:
                c_faces = frozenset()
                best_iou = 0.0

            rows.append({
                "class": cls,
                "gt_idx": gi,
                "iou": best_iou,
                "cascade_n_faces": len(c_faces),
                "gt_n_faces": len(gt_faces),
                "intersection": len(c_faces & gt_faces),
            })
            ious_by_class.setdefault(cls, []).append(best_iou)
            if best_iou < IOU_WARN_THRESHOLD:
                below.append((cls, gi, best_iou))

    mean_by_class = {
        cls: float(np.mean(vals)) if vals else 0.0
        for cls, vals in ious_by_class.items()
    }
    return rows, mean_by_class, below


def eval_part(gt_path: Path) -> PartReport:
    gt = _load_yaml(gt_path)
    part_id = str(gt.get("part_id") or gt_path.stem)
    step_path = _resolve_path(str(gt["part_step"]), gt_path)
    graph_path = _resolve_path(str(gt["graph_npz"]), gt_path)
    counts_gt = gt.get("counts") or {}
    gt_features = gt.get("features") or []
    face_truth = gt.get("face_truth")

    if not step_path.is_file():
        return PartReport(
            part_id=part_id, gt_path=gt_path, n_faces=0,
            cascade_completed=False, unclaimed_faces=-1,
            count_rows=[], count_divergences=[], zero_pass_flags=[],
            brep_flags={}, discrete_match={}, residual_note="",
            residual_coverage=None, iou_rows=[], iou_mean_by_class={},
            iou_below_threshold=[], error=f"STEP not found: {step_path}",
        )

    edge_index, edge_attr = _load_edges(
        graph_path if graph_path.is_file() else None,
        step_path,
    )
    pocket_config = PocketDetectionConfig(setup=pocket_config_from_setup_dict(gt))
    import logging
    prev_level = logging.root.level
    logging.root.setLevel(logging.ERROR)
    try:
        faces, pk, hl, cx, fl, of, wl, pr, rs, if_ = run_cascade(
            step_path, edge_index, edge_attr, pocket_config=pocket_config,
        )
    finally:
        logging.root.setLevel(prev_level)
    n_faces = len(faces)

    cascade_instances, residual_n, residual_faces = _extract_cascade_instances(
        pk, hl, cx, fl, of, wl, pr, rs,
    )
    cascade_counts = _count_by_tp(cascade_instances)

    total_claimed = len(
        pk.claimed_faces | hl.claimed_faces | cx.claimed_faces | fl.claimed_faces
        | of.claimed_faces | wl.claimed_faces | pr.claimed_faces | rs.claimed_faces
    )
    unclaimed = n_faces - total_claimed
    cascade_completed = unclaimed == 0

    count_rows: list[dict[str, Any]] = []
    divergences: list[tuple[str, int, int]] = []

    for cls in DISCRETE_TP_CLASSES:
        c_n = cascade_counts.get(cls, 0)
        g_n = int(counts_gt.get(cls, 0) or 0)
        flag = c_n != g_n
        count_rows.append({
            "class": cls,
            "cascade": c_n,
            "gt": g_n,
            "flag": flag,
            "mode": "discrete",
        })
        if flag:
            divergences.append((cls, c_n, g_n))

    residual_gt = _residual_gt_sum(counts_gt)
    count_rows.append({
        "class": "residual_bucket",
        "cascade": residual_n,
        "gt": residual_gt,
        "flag": False,
        "mode": "residual_info",
        "note": "cascade << Toolpath expected; informational only",
    })

    for cls in WIRED_TP_CLASSES + RESIDUAL_TP_CLASSES:
        g_n = int(counts_gt.get(cls, 0) or 0)
        c_n = cascade_counts.get(cls, 0) if cls in WIRED_TP_CLASSES else "—"
        flag = cls in WIRED_TP_CLASSES and c_n != g_n
        if g_n or cls in WIRED_TP_CLASSES:
            count_rows.append({
                "class": cls,
                "cascade": c_n,
                "gt": g_n,
                "flag": flag,
                "mode": "wired" if cls in WIRED_TP_CLASSES else "residual_gt_component",
            })
            if flag:
                divergences.append((cls, int(c_n), g_n))

    zero_pass_flags: list[str] = []
    pass_expectations = [
        ("filleted_pocket", sum(1 for f in pk.features if f.toolpath_class == "filleted_pocket"),
         int(counts_gt.get("filleted_pocket", 0) or 0)),
        ("filleted_open_pocket", sum(1 for f in pk.features if f.toolpath_class == "filleted_open_pocket"),
         int(counts_gt.get("filleted_open_pocket", 0) or 0)),
        ("through_hole", sum(1 for f in hl.features if f.kind == "through_hole"),
         int(counts_gt.get("through_hole", 0) or 0)),
        ("filleted_blind_hole", sum(1 for f in hl.features if f.kind == "blind_hole"),
         int(counts_gt.get("filleted_blind_hole", 0) or 0)),
        ("open_pocket", sum(1 for f in cx.features if f.kind == "open_pocket"),
         int(counts_gt.get("open_pocket", 0) or 0)),
        ("flat", len(fl.features), int(counts_gt.get("flat", 0) or 0)),
        ("outer_fillet", len(of.features),
         int(counts_gt.get("outer_fillet", 0) or 0)),
    ]
    for name, found, expected in pass_expectations:
        if expected > 0 and found == 0:
            zero_pass_flags.append(f"{name}: expected {expected}, pass found 0")

    brep_flags = _brep_quality_flags(faces, pk, hl, fl)
    discrete_match = _match_discrete_attributes(cascade_instances, gt_features)
    residual_note, residual_coverage = _residual_coverage(residual_faces, face_truth)

    iou_rows: list[dict[str, Any]] = []
    iou_mean: dict[str, float] = {}
    iou_below: list[tuple[str, int, float]] = []
    if face_truth:
        iou_rows, iou_mean, iou_below = _face_iou_baseline(cascade_instances, face_truth)

    return PartReport(
        part_id=part_id,
        gt_path=gt_path,
        n_faces=n_faces,
        cascade_completed=cascade_completed,
        unclaimed_faces=unclaimed,
        count_rows=count_rows,
        count_divergences=divergences,
        zero_pass_flags=zero_pass_flags,
        brep_flags=brep_flags,
        discrete_match=discrete_match,
        residual_note=residual_note,
        residual_coverage=residual_coverage,
        iou_rows=iou_rows,
        iou_mean_by_class=iou_mean,
        iou_below_threshold=iou_below,
    )


def _fmt_bool(v: bool | None) -> str:
    if v is None:
        return "—"
    return "yes" if v else "no"


def _print_part_report(r: PartReport) -> None:
    print("=" * 78)
    print(f"PART: {r.part_id}  ({r.gt_path.name})")
    if r.error:
        print(f"  ERROR: {r.error}")
        return

    print(f"  faces: {r.n_faces}   cascade complete: {_fmt_bool(r.cascade_completed)}"
          f"   unclaimed: {r.unclaimed_faces}")
    if r.zero_pass_flags:
        print(f"  ZERO-PASS FLAGS: {', '.join(r.zero_pass_flags)}")

    print("\n  COUNT METRICS (cascade vs Toolpath gt):")
    hdr = f"  {'class':22s} {'cascade':>8s} {'gt':>8s} {'flag':>6s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in r.count_rows:
        gt_val = row["gt"]
        gt_s = str(gt_val) if gt_val != "—" else "—"
        flag_s = "YES" if row.get("flag") else ("info" if row.get("mode") == "residual_info" else "no")
        print(f"  {row['class']:22s} {row['cascade']:>8} {gt_s:>8s} {flag_s:>6s}")
        if row.get("note"):
            print(f"    ({row['note']})")

    print("\n  B-REP QUALITY FLAGS:")
    for k, v in r.brep_flags.items():
        print(f"    {k}: {v}")

    print(f"\n  RESIDUAL: {r.residual_note}")
    if r.residual_coverage is not None:
        print(f"    coverage fraction: {r.residual_coverage:.1%}")

    has_attr = any(
        m.get("mode") == "attribute" for m in r.discrete_match.values()
    )
    if has_attr:
        print("\n  DISCRETE ATTRIBUTE MATCHING:")
        for cls, m in r.discrete_match.items():
            if m.get("mode") != "attribute":
                continue
            print(f"    {cls}: matched {m['matched']}/{m['cascade_n']} "
                  f"(cascade-only={m['cascade_only']}, gt-only={m['gt_only']}, "
                  f"rate={m['match_rate']:.0%})")
            for fm in m.get("matches") or []:
                dims = ", ".join(
                    f"{d.name}={'ok' if d.ok else 'FAIL'}"
                    + (f" Δ={d.delta:+.4f}in" if d.delta is not None else "")
                    for d in fm.hard_dims
                )
                status = "MATCH" if fm.matched else "miss"
                info = fm.info
                info_s = ""
                if info.get("feature_depth_delta_in") is not None:
                    info_s += f" feature_depth_Δ={info['feature_depth_delta_in']:+.4f}in"
                if info.get("ld_delta") is not None:
                    info_s += f" ld_Δ={info['ld_delta']:+.4f}"
                print(f"      [{status}] cascade#{fm.cascade_idx} ↔ gt#{fm.gt_idx}: {dims}{info_s}")

    if r.iou_rows:
        print("\n  REFERENCE FACE-LEVEL IoU:")
        for cls, mean_iou in r.iou_mean_by_class.items():
            print(f"    {cls}: mean IoU = {mean_iou:.4f}")
        if r.iou_below_threshold:
            print("    BELOW 0.95:")
            for cls, gi, iou in r.iou_below_threshold:
                print(f"      {cls} #{gi}: IoU={iou:.4f}")
        else:
            print("    all features ≥ 0.95")


def _aggregate_verdict(reports: Sequence[PartReport]) -> None:
    ok = [r for r in reports if not r.error]
    if not ok:
        return

    print("\n" + "=" * 78)
    print("AGGREGATE")
    print("=" * 78)

    all_divergences: list[tuple[str, str, int, int]] = []
    class_totals: dict[str, dict[str, int]] = {}
    match_rates: dict[str, list[float]] = {}
    brep_correlates: list[tuple[str, bool, dict[str, int]]] = []

    for r in ok:
        failed = bool(r.count_divergences or r.zero_pass_flags or r.iou_below_threshold)
        brep_correlates.append((r.part_id, failed, r.brep_flags))
        for cls, c_n, g_n in r.count_divergences:
            all_divergences.append((r.part_id, cls, c_n, g_n))
            bucket = class_totals.setdefault(cls, {"cascade": 0, "gt": 0, "parts": 0})
            bucket["cascade"] += c_n
            bucket["gt"] += g_n
            bucket["parts"] += 1

        for cls, m in r.discrete_match.items():
            if m.get("mode") == "attribute":
                match_rates.setdefault(cls, []).append(float(m["match_rate"]))

    print("\n  Per-class count accuracy (parts with gt yaml):")
    for cls in DISCRETE_TP_CLASSES:
        c_sum = g_sum = n_parts = 0
        for r in ok:
            for row in r.count_rows:
                if row["class"] == cls and row.get("mode") == "discrete":
                    c_sum += int(row["cascade"])
                    g_sum += int(row["gt"])
                    n_parts += 1
        if n_parts:
            acc = 1.0 - len([d for d in all_divergences if d[1] == cls]) / n_parts
            print(f"    {cls}: cascade={c_sum} gt={g_sum} "
                  f"({n_parts} part(s), count-agree rate={acc:.0%})")

    print("\n  Per-class discrete attribute match rate (where features: provided):")
    for cls, rates in match_rates.items():
        print(f"    {cls}: mean={float(np.mean(rates)):.0%} over {len(rates)} part(s)")

    if all_divergences:
        print("\n  WHERE IT BREAKS (part, class, cascade, gt):")
        for part, cls, c_n, g_n in all_divergences:
            print(f"    {part} / {cls}: cascade={c_n} gt={g_n}")
    else:
        print("\n  No discrete count divergences across evaluated parts.")

    ref_iou = [r for r in ok if r.iou_rows]
    if ref_iou:
        print("\n  REFERENCE-PLATE IoU BASELINE:")
        for r in ref_iou:
            overall = float(np.mean([row["iou"] for row in r.iou_rows]))
            print(f"    {r.part_id}: overall mean IoU = {overall:.4f}")
            if r.iou_below_threshold:
                print(f"      ALARM: {len(r.iou_below_threshold)} feature(s) below {IOU_WARN_THRESHOLD}")

    print("\n  VERDICT:")
    reliable: list[str] = []
    fragile: list[str] = []
    for cls in DISCRETE_TP_CLASSES:
        cls_divs = [d for d in all_divergences if d[1] == cls]
        rates = match_rates.get(cls, [])
        if not cls_divs and (not rates or float(np.mean(rates)) >= 0.95):
            reliable.append(cls)
        elif cls_divs or (rates and float(np.mean(rates)) < 0.8):
            fragile.append(cls)

    if reliable:
        print(f"    Reliable discrete classes: {', '.join(reliable)}")
    if fragile:
        print(f"    Fragile discrete classes: {', '.join(fragile)}")
    print("    Residual classes (contour/profile/wall): coarse by design — "
          "evaluate coverage, not instance counts.")
    wired = [c for c in WIRED_TP_CLASSES if c not in fragile]
    if wired:
        print(f"    Wired cascade classes: {', '.join(wired)}")

    bad_brep = [p for p, failed, flags in brep_correlates if failed and any(v > 0 for v in flags.values())]
    clean_brep = [p for p, failed, flags in brep_correlates if failed and not any(v > 0 for v in flags.values())]
    if bad_brep:
        print(f"    B-rep quality flags present on failing part(s): {', '.join(bad_brep)}")
    if clean_brep:
        print(f"    Failures without brep flags (logic/tuning): {', '.join(clean_brep)}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cascade vs Toolpath trust evaluation")
    ap.add_argument(
        "gt_files", nargs="*",
        help="GT yaml file(s); default: all eval/gt/*.yaml except _template.yaml",
    )
    ap.add_argument(
        "--gt-dir", type=Path, default=REPO_ROOT / "eval" / "gt",
        help="directory of GT yaml files",
    )
    args = ap.parse_args(argv)

    if args.gt_files:
        gt_paths = [Path(p) for p in args.gt_files]
    else:
        gt_paths = sorted(
            p for p in args.gt_dir.glob("*.yaml")
            if p.name != "_template.yaml"
        )

    if not gt_paths:
        print(f"No GT yaml files found under {args.gt_dir}")
        print("Copy eval/gt/_template.yaml and fill counts from Toolpath panels.")
        return 2

    reports: list[PartReport] = []
    for gt_path in gt_paths:
        reports.append(eval_part(gt_path.resolve()))

    for r in reports:
        _print_part_report(r)

    _aggregate_verdict(reports)

    any_error = any(r.error for r in reports)
    any_fail = any(
        r.count_divergences or r.zero_pass_flags or r.iou_below_threshold
        for r in reports if not r.error
    )
    return 2 if any_error else (1 if any_fail else 0)


if __name__ == "__main__":
    raise SystemExit(main())
