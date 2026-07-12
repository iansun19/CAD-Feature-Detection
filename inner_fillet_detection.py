"""
inner_fillet_detection.py - universal label for interior core tangent blends.

inner_fillet = a tangent curved blend on raised interior/core geometry
(convex-facing). Detection is a geometry-only template match (surface type,
radius, convex/concave edge counts, tangent-concave parent count, shell
membership, not envelope-coincident) whose thresholds are derived from a
reference face. The predicate is discriminative on its own: run purely on
geometry it selects exactly {159} on fish_mold and {} on both 96260B panels,
so it does NOT depend on STOCK gating (see require_stock_gated, default False).
Reference face: fish_mold face 159.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from feature_params import FaceGeom, analyze_step
from stock_cut_classification import (
    NEAR_TANGENT_ANGLE_DEG,
    _convex_summary,
    _envelope_coincident,
    _load_cad,
)

BLEND_SURFACE_TYPES = frozenset({"cylinder", "cone"})
REFERENCE_FACE_ID_FISH_MOLD = 159
REFERENCE_STEP_FISH_MOLD = "fish mold.stp"


@dataclass(frozen=True)
class InnerFilletThresholds:
    """Predicates derived from a reference inner_fillet face (do not guess)."""

    reference_face_id: int
    reference_surface_type: str
    reference_radius_mm: float
    max_radius_mm: float
    reference_shell_id: int
    core_shell_id: int
    reference_convex_edges: int
    reference_concave_edges: int
    max_concave_edges: int
    min_convex_edges: int
    max_strict_concave_edges: int
    tangent_concave_parent_count: int
    reference_area_mm2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_face_id": self.reference_face_id,
            "reference_surface_type": self.reference_surface_type,
            "reference_radius_mm": round(self.reference_radius_mm, 6),
            "max_radius_mm": round(self.max_radius_mm, 6),
            "reference_shell_id": self.reference_shell_id,
            "core_shell_id": self.core_shell_id,
            "reference_convex_edges": self.reference_convex_edges,
            "reference_concave_edges": self.reference_concave_edges,
            "max_concave_edges": self.max_concave_edges,
            "min_convex_edges": self.min_convex_edges,
            "max_strict_concave_edges": self.max_strict_concave_edges,
            "tangent_concave_parent_count": self.tangent_concave_parent_count,
            "reference_area_mm2": round(self.reference_area_mm2, 6),
        }


# Canonical template captured from the reference inner_fillet (fish_mold face
# 159). Lets the cascade pass build a config without re-deriving from the
# reference STEP at runtime. Regenerate via derive_thresholds_from_reference_face.
DEFAULT_INNER_FILLET_THRESHOLDS = InnerFilletThresholds(
    reference_face_id=159,
    reference_surface_type="cylinder",
    reference_radius_mm=3.175,
    max_radius_mm=4.7625,
    reference_shell_id=1,
    core_shell_id=1,
    reference_convex_edges=4,
    reference_concave_edges=0,
    max_concave_edges=0,
    min_convex_edges=4,
    max_strict_concave_edges=0,
    tangent_concave_parent_count=4,
    reference_area_mm2=198.855976,
)


@dataclass
class InnerFilletDetectionConfig:
    thresholds: InnerFilletThresholds
    radius_scale: float = 1.5
    # The geometric template is discriminative on its own, so gating on the
    # STOCK set is off by default. It was originally an extra guard, but it
    # breaks parts whose STOCK gate is disabled (e.g. fully-machined molds like
    # fish_mold, where the reference face 159 is no longer in any stock set).
    require_stock_gated: bool = False
    # Same story for the shell gate: requiring the candidate to live in the same
    # shell as the reference face is an extra guard, but it excludes genuine
    # inner_fillets that are geometrically identical yet sit in a different shell
    # (e.g. fish_mold face 90, an exact twin of reference face 159 in shell 0
    # rather than shell 1). Off by default; the template alone is discriminative
    # (matches exactly {90, 159} on fish_mold and nothing on the 96260B panels).
    require_core_shell: bool = False
    units: str = "mm"


def _face_shell_map(step_path: str | Path) -> dict[int, int]:
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from step_ingest import load_step_shape

    shape, _ = load_step_shape(str(step_path))
    topo = TopologyExplorer(shape)
    shape_faces = list(topo.faces())
    out: dict[int, int] = {}
    for si, shell in enumerate(topo.shells()):
        for sf in TopologyExplorer(shell).faces():
            for i, shf in enumerate(shape_faces):
                if sf.IsSame(shf):
                    out[i] = si
                    break
    return out


def _tangent_concave_parents_npz(
    face_id: int,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> set[int]:
    parents: set[int] = set()
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if face_id not in (u, v):
            continue
        other = v if u == face_id else u
        conv = int(edge_attr[k, 0])
        ang = float(edge_attr[k, 1]) if edge_attr.shape[1] > 1 else 180.0
        if conv != 0:
            continue
        if ang > NEAR_TANGENT_ANGLE_DEG and abs(ang - 180.0) > NEAR_TANGENT_ANGLE_DEG:
            continue
        parents.add(other)
    return parents


def derive_thresholds_from_reference_face(
    step_path: str | Path,
    face_id: int,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    radius_scale: float = 1.5,
) -> InnerFilletThresholds:
    faces = analyze_step(step_path)
    geom = faces[face_id]
    ctx = _load_cad(step_path)
    cs = _convex_summary(face_id, ctx)
    shell_map = _face_shell_map(step_path)
    shell_id = shell_map.get(face_id, 0)
    parents = _tangent_concave_parents_npz(face_id, edge_index, edge_attr)
    if geom.radius is None:
        raise ValueError(f"reference face {face_id} has no radius")
    return InnerFilletThresholds(
        reference_face_id=face_id,
        reference_surface_type=geom.surface_type,
        reference_radius_mm=float(geom.radius),
        max_radius_mm=float(geom.radius) * radius_scale,
        reference_shell_id=shell_id,
        core_shell_id=shell_id,
        reference_convex_edges=cs.convex_edges,
        reference_concave_edges=cs.concave_edges,
        max_concave_edges=cs.concave_edges,
        min_convex_edges=cs.convex_edges,
        max_strict_concave_edges=cs.strict_concave_edges,
        tangent_concave_parent_count=len(parents),
        reference_area_mm2=float(geom.area),
    )


def is_inner_fillet_candidate(
    face_id: int,
    faces: Sequence[FaceGeom],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    config: InnerFilletDetectionConfig,
    ctx: Any,
    shell_map: dict[int, int] | None = None,
    stock_face_ids: set[int] | frozenset[int] | None = None,
) -> bool:
    th = config.thresholds
    geom = faces[face_id]
    if geom.surface_type not in BLEND_SURFACE_TYPES:
        return False
    if geom.radius is None or geom.radius > th.max_radius_mm:
        return False
    cs = _convex_summary(face_id, ctx)
    if cs.concave_edges > th.max_concave_edges:
        return False
    if cs.convex_edges < th.min_convex_edges:
        return False
    if cs.strict_concave_edges > th.max_strict_concave_edges:
        return False
    parents = _tangent_concave_parents_npz(face_id, edge_index, edge_attr)
    if len(parents) != th.tangent_concave_parent_count:
        return False
    if (
        config.require_core_shell
        and shell_map is not None
        and shell_map.get(face_id) != th.core_shell_id
    ):
        return False
    if _envelope_coincident(face_id, ctx).coincident:
        return False
    if config.require_stock_gated:
        if not stock_face_ids or face_id not in stock_face_ids:
            return False
    return True


def matching_face_ids(
    step_path: str | Path,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    config: InnerFilletDetectionConfig,
    stock_face_ids: set[int] | frozenset[int] | None = None,
) -> list[int]:
    faces = analyze_step(step_path)
    ctx = _load_cad(step_path)
    shell_map = _face_shell_map(step_path)
    out: list[int] = []
    for i in range(len(faces)):
        if is_inner_fillet_candidate(
            i,
            faces,
            edge_index,
            edge_attr,
            config=config,
            ctx=ctx,
            shell_map=shell_map,
            stock_face_ids=stock_face_ids,
        ):
            out.append(i)
    return out


@dataclass
class InnerFilletFeature:
    feature_id: int
    face_indices: set[int]
    area: float
    reference_face_id: int
    kind: str = "inner_fillet"
    toolpath_class: str = "inner_fillet"
    # True concave fillet radius (mm) from the matched face's OCC geometry. The
    # planner uses 2*reference_radius as the FILLET_CAP tool-fit cap; None -> the
    # adapter falls back to the reachability probe radius (its prior behaviour).
    reference_radius_mm: float | None = None

    @property
    def n_faces(self) -> int:
        return len(self.face_indices)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "feature_id": self.feature_id,
            "kind": self.kind,
            "toolpath_class": self.toolpath_class,
            "face_indices": sorted(self.face_indices),
            "n_faces": self.n_faces,
            "area": round(float(self.area), 3),
            "reference_face_id": self.reference_face_id,
        }
        if self.reference_radius_mm is not None:
            d["reference_radius_mm"] = round(float(self.reference_radius_mm), 6)
        return d


@dataclass
class InnerFilletDetectionResult:
    features: list[InnerFilletFeature]
    claimed_faces: set[int]
    remaining_faces: set[int]
    n_faces: int

    def summary(self) -> str:
        return (
            f"{len(self.features)} inner_fillet(s); claimed "
            f"{len(self.claimed_faces)}/{self.n_faces} faces, "
            f"{len(self.remaining_faces)} remain"
        )


def default_inner_fillet_config(radius_scale: float = 1.5) -> InnerFilletDetectionConfig:
    """Config built from the canonical fish_mold-159 template (no STEP re-derive)."""
    return InnerFilletDetectionConfig(
        thresholds=DEFAULT_INNER_FILLET_THRESHOLDS,
        radius_scale=radius_scale,
    )


def detect_inner_fillets(
    step_path: str | Path,
    faces: Sequence[FaceGeom],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_faces: set[int] | frozenset[int] | None = None,
    config: InnerFilletDetectionConfig | None = None,
) -> InnerFilletDetectionResult:
    """Cascade pass: geometry-only inner_fillet template match over a face pool.

    Each matched face becomes its own single-face inner_fillet instance and is
    removed from the pool. Runs pure-geometry (no STOCK dependency), so it is
    stable whether or not a part's stock gate is enabled.
    """
    n_faces = len(faces)
    pool = (
        set(range(n_faces))
        if candidate_faces is None
        else {int(i) for i in candidate_faces}
    )
    cfg = config or default_inner_fillet_config()
    ctx = _load_cad(step_path)
    shell_map = _face_shell_map(step_path)

    features: list[InnerFilletFeature] = []
    claimed: set[int] = set()
    for i in sorted(pool):
        if is_inner_fillet_candidate(
            i, faces, edge_index, edge_attr,
            config=cfg, ctx=ctx, shell_map=shell_map,
        ):
            # True fillet radius from the matched face: cylinders carry it in
            # `radius`, toroidal blends in `torus_minor_r`. None when neither is
            # populated (adapter then falls back to the probe radius).
            geom = faces[i]
            ref_radius = getattr(geom, "radius", None)
            if ref_radius is None:
                ref_radius = getattr(geom, "torus_minor_r", None)
            features.append(InnerFilletFeature(
                feature_id=len(features),
                face_indices={i},
                area=float(geom.area),
                reference_face_id=cfg.thresholds.reference_face_id,
                reference_radius_mm=None if ref_radius is None else float(ref_radius),
            ))
            claimed.add(i)
    return InnerFilletDetectionResult(
        features=features,
        claimed_faces=claimed,
        remaining_faces=pool - claimed,
        n_faces=n_faces,
    )


def _detach_face_from_owner(node: dict[str, Any], face_id: int) -> None:
    """Remove ``face_id`` from an owning feature node's face lists in place."""
    node["face_ids"] = [f for f in node.get("face_ids", []) if f != face_id]
    node["n_faces"] = len(node["face_ids"])
    params = node.get("params")
    if isinstance(params, dict):
        if isinstance(params.get("face_indices"), list):
            params["face_indices"] = [
                f for f in params["face_indices"] if f != face_id
            ]
        if "n_faces" in params:
            params["n_faces"] = len(node["face_ids"])


def apply_inner_fillet_override(
    graph: dict[str, Any],
    face_id: int,
    *,
    step_path: str | Path | None = None,
    reference_face_id: int | None = None,
    labeled_by: str = "inner_fillet_detection",
) -> dict[str, Any]:
    """Direct graph edit: assign one face to a new inner_fillet instance.

    The face may be unclaimed or currently owned by another (non-inner_fillet)
    feature - e.g. absorbed into a pocket by the cascade. In the latter case the
    face is detached from its current owner before the inner_fillet instance is
    created. Re-applying when the face is already an inner_fillet is an error.
    """
    nodes = graph.get("nodes", [])
    before_feature_id = -1
    for node in nodes:
        if face_id in node.get("face_ids", []):
            if node.get("class_name") == "inner_fillet":
                raise ValueError(
                    f"face {face_id} already assigned to inner_fillet "
                    f"feature {node['feature_id']}"
                )
            before_feature_id = int(node["feature_id"])
            _detach_face_from_owner(node, face_id)
            break

    new_fid = max((int(n["feature_id"]) for n in nodes), default=-1) + 1
    area = 0.0
    if step_path is not None:
        area = float(analyze_step(step_path)[face_id].area)
    params: dict[str, Any] = {
        "feature_id": new_fid,
        "kind": "inner_fillet",
        "toolpath_class": "inner_fillet",
        "face_indices": [face_id],
        "n_faces": 1,
        "area": round(area, 3),
        "labeled_by": labeled_by,
    }
    if reference_face_id is not None:
        params["reference_face_id"] = reference_face_id
    nodes.append({
        "feature_id": new_fid,
        "class_id": 10,
        "class_name": "inner_fillet",
        "face_ids": [face_id],
        "n_faces": 1,
        "mean_confidence": 1.0,
        "params": params,
    })
    graph["nodes"] = nodes
    graph["n_features"] = len(nodes)

    stock_ids = [int(i) for i in graph.get("stock_face_ids", [])]
    if face_id in stock_ids:
        graph["stock_face_ids"] = sorted(i for i in stock_ids if i != face_id)

    overrides = dict(graph.get("face_label_overrides", {}))
    overrides[str(face_id)] = {
        "stock_cut_label": "CUT",
        "labeled_by": labeled_by,
        "gate_rule": None,
    }
    graph["face_label_overrides"] = overrides
    return {
        "new_feature_id": new_fid,
        "before_feature_id": before_feature_id,
    }


# Back-compat alias (was manual, single-face, fish_mold-specific).
def apply_inner_fillet_manual_override(
    graph: dict[str, Any],
    face_id: int,
    *,
    step_path: str | Path | None = None,
) -> dict[str, Any]:
    return apply_inner_fillet_override(
        graph, face_id, step_path=step_path,
        reference_face_id=REFERENCE_FACE_ID_FISH_MOLD,
        labeled_by="manual_override",
    )


def apply_inner_fillets(
    graph: dict[str, Any],
    step_path: str | Path,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    config: InnerFilletDetectionConfig,
    stock_face_ids: set[int] | frozenset[int] | None = None,
) -> dict[str, Any]:
    """Universal, detection-driven application of the inner_fillet label.

    Runs the geometric predicate over every face, then patches each matched
    face into its own inner_fillet instance (detaching it from any cascade
    feature that absorbed it). Faces already labeled inner_fillet are skipped,
    so this is idempotent. Returns {matches, applied, skipped}.
    """
    matches = matching_face_ids(
        step_path, edge_index, edge_attr,
        config=config, stock_face_ids=stock_face_ids,
    )
    applied: list[dict[str, Any]] = []
    skipped: list[int] = []
    for fid in matches:
        owner_class = next(
            (n.get("class_name") for n in graph.get("nodes", [])
             if fid in n.get("face_ids", [])),
            None,
        )
        if owner_class == "inner_fillet":
            skipped.append(fid)
            continue
        report = apply_inner_fillet_override(
            graph, fid, step_path=step_path,
            reference_face_id=config.thresholds.reference_face_id,
        )
        applied.append({"face_id": fid, **report})
    return {"matches": matches, "applied": applied, "skipped": skipped}


def print_derived_thresholds(th: InnerFilletThresholds) -> None:
    print("Derived inner_fillet thresholds (from reference face):")
    for key, val in th.to_dict().items():
        print(f"  {key}: {val}")
