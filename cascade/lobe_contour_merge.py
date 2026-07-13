"""Post-extraction merge for over-segmented filleted-lobe cap contour fragments.

Runs after ``build_cascade_feature_graph`` and before reachability export. Groups
``contour_surface`` nodes that belong to the same lobe cap (angular lobe sector +
exterior axial band) into one auditable merged node. Cross-lobe boundaries are
never merged; area is never used as a merge gate.

Merge is enabled only when filleted lobe tiers are discovered (>= ``min_lobes``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from brep.feature_params import FaceGeom
from cascade.hole_detection import FaceGraph
from brep.instance_features import instance_features
from cascade.lobe_tier_detection import _project_uv, detect_filleted_lobe_tiers

# |cos(dihedral)| near 1.0 ? G1 (co-tangent) continuity across the shared edge.
G1_COS_TOL = 0.85


@dataclass
class LobeContourMergeConfig:
    min_lobes: int = 6
    g1_cos_tol: float = G1_COS_TOL


@dataclass
class LobeContourMergeReport:
    enabled: bool
    n_lobes: int = 0
    candidates_before: int = 0
    clusters: list[dict[str, Any]] = field(default_factory=list)
    nodes_before: int = 0
    nodes_after: int = 0
    nodes_removed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "n_lobes": self.n_lobes,
            "candidates_before": self.candidates_before,
            "clusters": self.clusters,
            "nodes_before": self.nodes_before,
            "nodes_after": self.nodes_after,
            "nodes_removed": self.nodes_removed,
        }


def _edge_is_g1_continuous(graph: FaceGraph, u: int, v: int, *, cos_tol: float) -> bool:
    kind = graph.edge_kind(u, v)
    if kind == "smooth":
        return True
    cos = graph.edge_cos(u, v)
    return cos is not None and abs(cos) >= cos_tol


def _make_lobe_assigner(
    lobe_result,
    by_index: dict[int, FaceGeom],
    opening_axis: np.ndarray,
):
    """Return (mouth_axial, face_id -> lobe_id callable)."""
    mouths = [lobe.mouth_step_face for lobe in lobe_result.lobes]
    mouth_axial = float(lobe_result.lobes[0].mouth_axial) if lobe_result.lobes else 0.0
    mouth_uv = {
        m: _project_uv(np.asarray(by_index[m].centroid, dtype=np.float64), opening_axis)
        for m in mouths
    }
    center = np.mean(np.array(list(mouth_uv.values())), axis=0)
    mouth_angle = {
        m: float(math.atan2(uv[1] - center[1], uv[0] - center[0]))
        for m, uv in mouth_uv.items()
    }
    mouth_to_lobe = {m: i for i, m in enumerate(mouths)}

    def _ang_dist(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 2.0 * math.pi - d)

    def assign(face_id: int) -> int:
        uv = _project_uv(np.asarray(by_index[face_id].centroid, dtype=np.float64), opening_axis)
        theta = float(math.atan2(uv[1] - center[1], uv[0] - center[0]))
        best_m = min(mouths, key=lambda m: _ang_dist(theta, mouth_angle[m]))
        return mouth_to_lobe[best_m]

    return mouth_axial, assign


def _axial_y(centroid: np.ndarray, opening_axis: np.ndarray) -> float:
    a = opening_axis / max(float(np.linalg.norm(opening_axis)), 1e-12)
    return float(np.dot(np.asarray(centroid, dtype=np.float64), a))


def _candidate_nodes(
    nodes: list[dict[str, Any]],
    by_index: dict[int, FaceGeom],
    assign_lobe,
    *,
    mouth_axial: float,
    opening_axis: np.ndarray,
) -> dict[int, list[int]]:
    """Map lobe_id -> list of contour_surface node indices eligible for merge."""
    per_lobe: dict[int, list[int]] = {}
    for idx, node in enumerate(nodes):
        if node.get("class_name") != "contour_surface":
            continue
        face_ids = [int(f) for f in node.get("face_ids", [])]
        if not face_ids:
            continue
        lobes: set[int] = set()
        in_cap = True
        for fid in face_ids:
            fg = by_index.get(fid)
            if fg is None:
                in_cap = False
                break
            if _axial_y(np.asarray(fg.centroid), opening_axis) <= mouth_axial:
                in_cap = False
                break
            lobes.add(assign_lobe(fid))
        if not in_cap or len(lobes) != 1:
            continue
        lid = next(iter(lobes))
        per_lobe.setdefault(lid, []).append(idx)
    return per_lobe


def _parent_score(
    node: dict[str, Any],
    *,
    prefer_rear: bool = False,
) -> tuple[float, float, int]:
    params = node.get("params") or {}
    hist = params.get("surface_type_histogram") or {}
    area = float(params.get("total_area") or 0.0)
    if "bspline" in hist:
        tier = 3.0
    elif "torus" in hist and area >= 50.0:
        tier = 2.0
    elif "torus" in hist:
        tier = 1.5
    else:
        tier = 1.0
    reach_bonus = 0.0
    reach = ((node.get("approach") or {}).get("reachability") or {})
    dirs = set(reach.get("reachable_dirs") or [])
    if prefer_rear and "-Z" in dirs:
        reach_bonus = 1.0
    elif not prefer_rear and "+Z" in dirs:
        reach_bonus = 1.0
    return (tier + reach_bonus, area, -int(node.get("feature_id", 0)))


def _dominant_parent_node(
    constituents: list[dict[str, Any]],
    *,
    prefer_rear: bool = False,
) -> dict[str, Any]:
    return max(constituents, key=lambda n: _parent_score(n, prefer_rear=prefer_rear))


def _dominant_parent_params(
    constituents: list[dict[str, Any]],
    *,
    prefer_rear: bool = False,
) -> dict[str, Any]:
    return (_dominant_parent_node(constituents, prefer_rear=prefer_rear).get("params") or {})


def _reachable_dirs(node: dict[str, Any]) -> set[str]:
    reach = ((node.get("approach") or {}).get("reachability") or {})
    return set(reach.get("reachable_dirs") or [])


def _is_machining_anchor(
    node: dict[str, Any],
    *,
    prefer_rear: bool,
) -> bool:
    """Constituent reachable from this panel's setup (machining parent, not cap debris)."""
    dirs = _reachable_dirs(node)
    if prefer_rear:
        return "-Z" in dirs
    return "+Z" in dirs


def _partition_constituents(
    constituents: list[dict[str, Any]],
    *,
    prefer_rear: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchors = [n for n in constituents if _is_machining_anchor(n, prefer_rear=prefer_rear)]
    debris = [n for n in constituents if n not in anchors]
    if not anchors:
        parent = _dominant_parent_node(constituents, prefer_rear=prefer_rear)
        anchors = [parent]
        debris = [n for n in constituents if n is not parent]
    return anchors, debris


def _build_merged_node(
    *,
    feature_id: int,
    face_ids: set[int],
    constituents: list[dict[str, Any]],
    parent_node: dict[str, Any],
    by_index: dict[int, FaceGeom],
    brep_graph: FaceGraph,
    lid: int,
    kind: str,
    prefer_rear: bool,
) -> dict[str, Any]:
    fragment_ids = sorted(int(n["feature_id"]) for n in constituents)
    merged_faces = [by_index[f] for f in sorted(face_ids) if f in by_index]
    parent_params = parent_node.get("params") or {}
    new_params = instance_features(
        merged_faces,
        graph=brep_graph,
        face_indices=face_ids,
    )
    for key, val in parent_params.items():
        if key in (
            "face_indices", "n_faces", "surface_type_histogram",
            "total_area", "3D_surface", "n_distinct_axes",
            "internal_edge_convexity", "bbox_size_x", "bbox_size_y",
            "bbox_size_z", "bbox_depth", "bbox_width", "bbox_length",
        ):
            continue
        if key not in new_params:
            new_params[key] = val
    new_params["merged_from_fragment_ids"] = fragment_ids
    new_params["lobe_contour_merge"] = True
    new_params["lobe_contour_merge_kind"] = kind
    new_params["lobe_id"] = lid

    merged_node: dict[str, Any] = {
        "feature_id": feature_id,
        "class_id": constituents[0].get("class_id"),
        "class_name": "contour_surface",
        "face_ids": sorted(face_ids),
        "n_faces": len(face_ids),
        "mean_confidence": 1.0,
        "params": new_params,
    }
    if kind == "anchor":
        approach_block = _inherit_approach_block(
            [parent_node], prefer_rear=prefer_rear,
        )
        if approach_block is not None:
            merged_node["approach"] = approach_block
    else:
        approach = dict(parent_node.get("approach") or {})
        approach["reachability"] = {
            "verified": True,
            "reachable_dirs": [],
            "occluded": True,
            "required_depth_mm": 0.0,
            "effective_tool_radius_mm": 0.0,
            "per_direction": {},
            "lobe_contour_debris_sink": True,
        }
        merged_node["approach"] = approach
    return merged_node


def _inherit_approach_block(
    constituents: list[dict[str, Any]],
    *,
    prefer_rear: bool = False,
) -> dict[str, Any] | None:
    parent = _dominant_parent_node(constituents, prefer_rear=prefer_rear)
    approach = parent.get("approach")
    return dict(approach) if approach else None


def merge_lobe_contour_fragments(
    graph: dict[str, Any],
    faces: Sequence[FaceGeom],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    opening_axis: Sequence[float] | None = None,
    occ_faces: Sequence[Any] | None = None,
    enabled: bool = True,
    config: LobeContourMergeConfig | None = None,
    prefer_rear: bool = False,
) -> tuple[dict[str, Any], LobeContourMergeReport]:
    """Merge same-lobe cap ``contour_surface`` fragments; return updated graph."""
    cfg = config or LobeContourMergeConfig()
    report = LobeContourMergeReport(
        enabled=enabled,
        nodes_before=len(graph.get("nodes", [])),
    )
    if not enabled:
        report.nodes_after = report.nodes_before
        return graph, report

    nodes: list[dict[str, Any]] = list(graph.get("nodes", []))
    if not nodes:
        report.nodes_after = 0
        return graph, report

    by_index = {int(f.index): f for f in faces}
    axis = (
        np.asarray(opening_axis, dtype=np.float64)
        if opening_axis is not None
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    brep_graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(faces))

    lobe_result = detect_filleted_lobe_tiers(
        faces, edge_index, edge_attr,
        occ_faces=occ_faces,
        opening_axis=axis,
        config=None,
    )
    report.n_lobes = len(lobe_result.lobes)
    if report.n_lobes < cfg.min_lobes:
        report.nodes_after = report.nodes_before
        return graph, report

    mouth_axial, assign_lobe = _make_lobe_assigner(lobe_result, by_index, axis)
    per_lobe = _candidate_nodes(
        nodes, by_index, assign_lobe,
        mouth_axial=mouth_axial,
        opening_axis=axis,
    )
    report.candidates_before = sum(len(v) for v in per_lobe.values())

    remove_indices: set[int] = set()
    merged_nodes: list[dict[str, Any]] = []

    for lid, cand_indices in sorted(per_lobe.items()):
        if len(cand_indices) < 2:
            continue

        merge_group = list(cand_indices)
        constituents = [nodes[i] for i in merge_group]
        anchors, debris = _partition_constituents(constituents, prefer_rear=prefer_rear)

        if len(debris) < 2:
            continue

        debris_faces: set[int] = set()
        for n in debris:
            debris_faces.update(int(f) for f in n.get("face_ids", []))

        if not debris_faces:
            continue

        debris_id = min(int(n["feature_id"]) for n in debris)
        all_fragment_ids = sorted(int(n["feature_id"]) for n in constituents)
        debris_node = _build_merged_node(
            feature_id=debris_id,
            face_ids=debris_faces,
            constituents=debris,
            parent_node=_dominant_parent_node(debris, prefer_rear=prefer_rear),
            by_index=by_index,
            brep_graph=brep_graph,
            lid=lid,
            kind="debris",
            prefer_rear=prefer_rear,
        )
        debris_node["params"]["merged_from_fragment_ids"] = all_fragment_ids
        merged_nodes.append(debris_node)
        remove_indices.update(i for i in merge_group if nodes[i] in debris)
        report.clusters.append({
            "lobe_id": lid,
            "merged_feature_id": debris_id,
            "kind": "debris",
            "fragment_ids": sorted(int(n["feature_id"]) for n in debris),
            "n_faces": len(debris_faces),
            "anchor_ids_preserved": sorted(int(n["feature_id"]) for n in anchors),
        })

    if not remove_indices:
        report.nodes_after = report.nodes_before
        return graph, report

    kept = [n for i, n in enumerate(nodes) if i not in remove_indices]
    kept.extend(merged_nodes)
    kept.sort(key=lambda n: int(n["feature_id"]))

    from brep.feature_graph import feature_adjacency

    class _Inst:
        def __init__(self, face_ids):
            self.face_ids = face_ids

    instances = [_Inst(n["face_ids"]) for n in kept]
    edges = feature_adjacency(instances, edge_index, graph["n_faces"])

    out = dict(graph)
    out["nodes"] = kept
    out["n_features"] = len(kept)
    out["n_edges"] = len(edges)
    out["edges"] = edges

    report.nodes_after = len(kept)
    report.nodes_removed = report.nodes_before - report.nodes_after
    out["lobe_contour_merge"] = report.to_dict()
    return out, report
