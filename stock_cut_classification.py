"""
stock_cut_classification.py - geometry-only STOCK vs CUT face labels.

Old classifier: envelope / axis-aligned planar heuristics (pre-convexity-primary).
New classifier: convexity-primary (exterior = no concave adjacency -> STOCK).

Op-list reconciliation TODO
---------------------------
Faces that are boundary-coincident with the stock envelope *and* explicitly machined
by an operation in the shop program are out of scope for this geometry-only diff.
They should be surfaced via op-list reconciliation, not silently labeled here.
See classify_report() notes on ``boundary_coincident_machined`` (always None until wired).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.BRepTools import breptools
    from OCC.Extend.TopologyUtils import TopologyExplorer

    from brep_extents import collect_boundary_points
    from feature_params import FaceGeom, analyze_step, load_step_faces, require_occ
    from step_ingest import extract_brep_from_step, face_mid_normal, load_step_shape

    HAS_OCC = True
except ImportError:
    HAS_OCC = False

Label = Literal["STOCK", "CUT"]
ClassifierName = Literal["old", "new"]
LabeledBy = Literal[
    "pre_cascade_gate",
    "convexity_cascade",
    "convexity_cascade_pool_unclaimed",
]

ENVELOPE_TOL_MM = 0.05
AXIS_ALIGN_MIN = 0.9
EXTREME_NORMAL_ALIGN_MIN = 0.85
NEAR_TANGENT_ANGLE_DEG = 5.0

CLASSIFICATION_FIXTURES: dict[str, dict[str, Any]] = {
    "fish_mold": {
        "step": "fish mold.stp",
        "expectation": "change",
        "corner_chamfer_face_ids": [106, 107, 210, 213],
        "notes": (
            "Corner chamfers should flip CUT->STOCK under convexity-primary logic; "
            "bounding stock-face count (extreme/axis planes + corners) 6->10."
        ),
    },
    "96260B_front": {
        "step": "96260B_FRONT_XR004_PCD PLATE.stp copy",
        "expectation": "stable",
        "notes": (
            "Tripwire: any old-vs-new flip is a decision point, not an automatic bug. "
            "Inspect flip direction, driving signals, convex_summary, and "
            "normal_consistent to distinguish a correction from a regression."
        ),
    },
    "96260B_rear": {
        "step": "96260B_REAR_XR004_PCD PLATE.stp copy",
        "expectation": "stable",
        "notes": (
            "Same stability contract as 96260B_front - emergent agreement, not "
            "part-specific short-circuiting."
        ),
    },
}


@dataclass
class ConvexSummary:
    convex_edges: int = 0
    concave_edges: int = 0
    smooth_edges: int = 0
    boundary_edges: int = 0
    min_signed_convexity: float = 0.0
    near_tangent_edges: int = 0
    strict_concave_edges: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "convex_edges": self.convex_edges,
            "concave_edges": self.concave_edges,
            "smooth_edges": self.smooth_edges,
            "boundary_edges": self.boundary_edges,
            "min_signed_convexity": round(float(self.min_signed_convexity), 6),
            "near_tangent_edges": self.near_tangent_edges,
            "strict_concave_edges": self.strict_concave_edges,
        }


@dataclass
class EnvelopeCoincident:
    coincident: bool
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "coincident": self.coincident,
            "score": round(float(self.score), 6),
        }


@dataclass
class FaceClassificationRecord:
    face_id: int
    label: Label
    driving_signal: dict[str, Any]
    convex_summary: ConvexSummary
    envelope_coincident: EnvelopeCoincident
    normal_consistent: bool
    surface_type: str = ""
    boundary_coincident_machined: bool | None = None
    labeled_by: LabeledBy = "convexity_cascade_pool_unclaimed"
    gate_rule: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_id": self.face_id,
            "label": self.label,
            "driving_signal": self.driving_signal,
            "convex_summary": self.convex_summary.to_dict(),
            "envelope_coincident": self.envelope_coincident.to_dict(),
            "normal_consistent": self.normal_consistent,
            "surface_type": self.surface_type,
            "boundary_coincident_machined": self.boundary_coincident_machined,
            "labeled_by": self.labeled_by,
            "gate_rule": self.gate_rule,
        }


@dataclass
class ClassificationDiff:
    part_id: str
    n_faces: int
    n_flipped: int
    stock_to_cut: int
    cut_to_stock: int
    flips: list[dict[str, Any]] = field(default_factory=list)
    old_stock_count: int = 0
    new_stock_count: int = 0
    old_bounding_stock_count: int = 0
    new_bounding_stock_count: int = 0

    def summary_line(self) -> str:
        return (
            f"{self.n_flipped} faces flipped, "
            f"{self.stock_to_cut} STOCK->CUT, {self.cut_to_stock} CUT->STOCK "
            f"(stock {self.old_stock_count}->{self.new_stock_count}; "
            f"bounding stock {self.old_bounding_stock_count}->"
            f"{self.new_bounding_stock_count})"
        )


@dataclass
class _CadContext:
    step_path: Path
    faces: list[Any]
    geoms: list[FaceGeom]
    model: dict
    part_aabb: tuple[np.ndarray, np.ndarray]
    edge_info: dict[tuple[int, int], dict[str, float]]
    boundary_edges: dict[int, int]


def _require_occ() -> None:
    if not HAS_OCC:
        raise ImportError(
            "stock_cut_classification requires pythonocc-core "
            "(conda env create -f environment.yml && conda activate mlcad)"
        )
    require_occ()


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v * 0.0
    return v / n


def _canonical_pair(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _shape_aabb(shape) -> tuple[np.ndarray, np.ndarray]:
    box = Bnd_Box()
    brepbndlib.AddOptimal(shape, box, True, False)
    box.SetGap(0.0)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (
        np.array([xmin, ymin, zmin], dtype=np.float64),
        np.array([xmax, ymax, zmax], dtype=np.float64),
    )


def _load_cad(step_path: str | Path) -> _CadContext:
    _require_occ()
    step_path = Path(step_path)
    shape, _ = load_step_shape(str(step_path))
    faces = load_step_faces(step_path)
    geoms = analyze_step(step_path)
    model, _ = extract_brep_from_step(str(step_path), require_labels=False)
    part_aabb = _shape_aabb(shape)

    e1_pairs = {tuple(x) for x in model["E1"].tolist()}
    e2_pairs = {tuple(x) for x in model["E2"].tolist()}

    edge_info: dict[tuple[int, int], dict[str, float]] = {}
    if model["A"].size:
        seen: set[tuple[int, int]] = set()
        for (u, v), ang in zip(model["A"].tolist(), model["Aval"].tolist()):
            if u == v:
                continue
            key = _canonical_pair(int(u), int(v))
            if key in seen:
                continue
            seen.add(key)
            if key in e2_pairs:
                sgn = -1.0
            elif key in e1_pairs:
                sgn = 1.0
            else:
                sgn = 0.0
            edge_info[key] = {
                "signed_convexity": sgn,
                "angle_deg": float(np.degrees(ang)),
            }

    boundary_edges: dict[int, int] = {}
    topo = TopologyExplorer(shape)
    fidx = {face: i for i, face in enumerate(faces)}
    for edge in topo.edges():
        ef = [f for f in topo.faces_from_edge(edge) if f in fidx]
        if len(ef) == 1:
            boundary_edges[fidx[ef[0]]] = boundary_edges.get(fidx[ef[0]], 0) + 1

    return _CadContext(
        step_path=step_path,
        faces=faces,
        geoms=geoms,
        model=model,
        part_aabb=part_aabb,
        edge_info=edge_info,
        boundary_edges=boundary_edges,
    )


def _edge_is_near_tangent(angle_deg: float) -> bool:
    return (
        angle_deg <= NEAR_TANGENT_ANGLE_DEG
        or abs(angle_deg - 180.0) <= NEAR_TANGENT_ANGLE_DEG
    )


def _envelope_coincident(
    face_idx: int,
    ctx: _CadContext,
    *,
    tol_mm: float = ENVELOPE_TOL_MM,
) -> EnvelopeCoincident:
    occ_face = ctx.faces[face_idx]
    pts = collect_boundary_points(occ_face)
    mn, mx = ctx.part_aabb
    dists: list[float] = []
    for p in pts:
        dx = min(abs(p[0] - mn[0]), abs(p[0] - mx[0]))
        dy = min(abs(p[1] - mn[1]), abs(p[1] - mx[1]))
        dz = min(abs(p[2] - mn[2]), abs(p[2] - mx[2]))
        dists.append(float(min(dx, dy, dz)))
    if not dists:
        return EnvelopeCoincident(coincident=False, score=0.0)
    min_dist = min(dists)
    on_frac = sum(d <= tol_mm for d in dists) / len(dists)
    score = max(0.0, 1.0 - min_dist / max(tol_mm * 4.0, 1e-6))
    score = max(score, on_frac)
    return EnvelopeCoincident(
        coincident=min_dist <= tol_mm or on_frac >= 0.5,
        score=min(1.0, score),
    )


def _normal_consistent(face_idx: int, ctx: _CadContext) -> bool:
    occ_face = ctx.faces[face_idx]
    oriented = face_mid_normal(occ_face)
    umin, umax, vmin, vmax = breptools.UVBounds(occ_face)
    surf = BRepAdaptor_Surface(occ_face, True)
    p = BRepLProp_SLProps(
        surf, 0.5 * (umin + umax), 0.5 * (vmin + vmax), 1, 1e-6,
    )
    if not p.IsNormalDefined():
        return True
    d = p.Normal()
    parametric = np.array([d.X(), d.Y(), d.Z()], dtype=np.float64)
    if float(np.linalg.norm(oriented)) < 1e-9:
        return True
    if float(np.linalg.norm(parametric)) < 1e-9:
        return True
    return float(np.dot(_unit(oriented), _unit(parametric))) > 0.0


def _convex_summary(face_idx: int, ctx: _CadContext) -> ConvexSummary:
    summary = ConvexSummary(
        boundary_edges=ctx.boundary_edges.get(face_idx, 0),
        min_signed_convexity=1.0,
    )
    for (u, v), info in ctx.edge_info.items():
        if face_idx not in (u, v):
            continue
        sgn = info["signed_convexity"]
        ang = info["angle_deg"]
        if sgn > 0:
            summary.convex_edges += 1
        elif sgn < 0:
            summary.concave_edges += 1
            if not _edge_is_near_tangent(ang):
                summary.strict_concave_edges += 1
        else:
            summary.smooth_edges += 1
        summary.min_signed_convexity = min(summary.min_signed_convexity, sgn)
        if _edge_is_near_tangent(ang):
            summary.near_tangent_edges += 1
    if summary.convex_edges + summary.concave_edges + summary.smooth_edges == 0:
        summary.min_signed_convexity = 0.0
    return summary


def _is_planar(face_idx: int, ctx: _CadContext) -> bool:
    return ctx.geoms[face_idx].surface_type == "plane"


def _axis_aligned_normal(normal: np.ndarray) -> bool:
    n = _unit(normal)
    return max(abs(float(n[0])), abs(float(n[1])), abs(float(n[2]))) >= AXIS_ALIGN_MIN


def _extreme_plane_match(face_idx: int, ctx: _CadContext) -> bool:
    if not _is_planar(face_idx, ctx):
        return False
    env = _envelope_coincident(face_idx, ctx)
    if not env.coincident:
        return False
    n = _unit(ctx.geoms[face_idx].normal)
    mn, mx = ctx.part_aabb
    c = ctx.geoms[face_idx].centroid
    for axis_i, sign in (
        (0, mn[0]), (0, mx[0]),
        (1, mn[1]), (1, mx[1]),
        (2, mn[2]), (2, mx[2]),
    ):
        if abs(c[axis_i] - sign) > ENVELOPE_TOL_MM:
            continue
        axis_vec = np.zeros(3)
        axis_vec[axis_i] = 1.0 if c[axis_i] >= 0 else -1.0
        if abs(float(np.dot(n, axis_vec))) >= EXTREME_NORMAL_ALIGN_MIN:
            return True
    return False


def _sloped_envelope_planar(face_idx: int, ctx: _CadContext) -> bool:
    if not _is_planar(face_idx, ctx):
        return False
    env = _envelope_coincident(face_idx, ctx)
    return env.coincident and not _axis_aligned_normal(ctx.geoms[face_idx].normal)


def _classify_old(face_idx: int, ctx: _CadContext) -> tuple[Label, dict[str, Any]]:
    env = _envelope_coincident(face_idx, ctx)
    summary = _convex_summary(face_idx, ctx)

    if _sloped_envelope_planar(face_idx, ctx):
        return "CUT", {"rule": "planar_default", "envelope_score": env.score}
    if summary.concave_edges > 0:
        return "CUT", {"rule": "concave_adjacency", "envelope_score": env.score}
    if _is_planar(face_idx, ctx) and _extreme_plane_match(face_idx, ctx):
        return "STOCK", {"rule": "extreme_plane", "envelope_score": env.score}
    if _is_planar(face_idx, ctx) and env.coincident and _axis_aligned_normal(ctx.geoms[face_idx].normal):
        return "STOCK", {"rule": "axis_aligned", "envelope_score": env.score}
    return "STOCK", {"rule": "convex_exterior", "envelope_score": env.score}


def _classify_new(face_idx: int, ctx: _CadContext) -> tuple[Label, dict[str, Any]]:
    summary = _convex_summary(face_idx, ctx)
    env = _envelope_coincident(face_idx, ctx)

    if _sloped_envelope_planar(face_idx, ctx):
        label: Label = "CUT" if summary.strict_concave_edges > 0 else "STOCK"
        primary = "convexity"
    elif summary.concave_edges > 0:
        label = "CUT"
        primary = "convexity"
    else:
        label = "STOCK"
        primary = "convexity"

    if label == "STOCK" and env.coincident:
        agreement = "agreed"
    elif label == "CUT" and not env.coincident:
        agreement = "agreed"
    elif label == "STOCK" and not env.coincident:
        agreement = "disagreed"
    else:
        agreement = "disagreed"

    return label, {
        "primary": primary,
        "envelope_agreement": agreement,
        "envelope_score": env.score,
    }


def _is_axis_extreme_stock(face_idx: int, ctx: _CadContext) -> bool:
    if not _is_planar(face_idx, ctx):
        return False
    env = _envelope_coincident(face_idx, ctx)
    return _extreme_plane_match(face_idx, ctx) or (
        env.coincident and _axis_aligned_normal(ctx.geoms[face_idx].normal)
    )


def _count_bounding_stock(
    records: dict[int, FaceClassificationRecord],
    ctx: _CadContext,
    *,
    corner_chamfer_ids: set[int] | None = None,
) -> int:
    count = 0
    for face_id, record in records.items():
        if record.label != "STOCK":
            continue
        if _is_axis_extreme_stock(face_id, ctx):
            count += 1
        elif corner_chamfer_ids and face_id in corner_chamfer_ids:
            count += 1
    return count


def _derive_labeled_by(
    face_id: int,
    label: Label,
    stock_ids: set[int],
    cascade_face_ids: set[int] | frozenset[int] | None,
) -> LabeledBy:
    if face_id in stock_ids or label == "STOCK":
        return "pre_cascade_gate"
    if cascade_face_ids and face_id in cascade_face_ids:
        return "convexity_cascade"
    return "convexity_cascade_pool_unclaimed"


def _gate_rule(
    face_idx: int,
    ctx: _CadContext,
    record: FaceClassificationRecord,
) -> str | None:
    if record.labeled_by != "pre_cascade_gate":
        return None
    if _sloped_envelope_planar(face_idx, ctx):
        return "sloped-envelope"
    if _is_axis_extreme_stock(face_idx, ctx):
        return "envelope-extreme"
    rule = record.driving_signal.get("rule")
    if rule in ("extreme_plane", "axis_aligned"):
        return "envelope-extreme"
    return None


def _flip_reason(
    face_idx: int,
    ctx: _CadContext,
    old_label: Label,
    new_label: Label,
    old_signal: dict[str, Any],
    new_signal: dict[str, Any],
    summary: ConvexSummary,
) -> str:
    g = ctx.geoms[face_idx]
    old_rule = old_signal.get("rule", "?")
    if _sloped_envelope_planar(face_idx, ctx) and old_label == "CUT" and new_label == "STOCK":
        return (
            "chamfer: pre_cascade_gate (sloped-envelope); "
            "old planar_default said CUT"
        )
    if old_rule == "planar_default" and new_label == "STOCK":
        return "sloped planar on envelope: convex adjacency, old logic defaulted to CUT"
    if old_rule == "curved_default" and new_label == "STOCK":
        return "exterior curved face: convex, old logic defaulted to CUT"
    if old_rule in ("curved_envelope", "extreme_plane", "axis_aligned") and new_label == "CUT":
        return (
            "interior/exterior mismatch: old envelope rule said STOCK, concave adjacency says CUT"
        )
    if old_label == "CUT" and new_label == "STOCK":
        return "OD/exterior face: convex, old logic defaulted to CUT"
    if old_label == "STOCK" and new_label == "CUT":
        return "interior machined face: concave adjacency, old envelope rule said STOCK"
    return f"{g.surface_type}: {old_label}->{new_label}"


def classify_report(
    cad: str | Path | _CadContext,
    *,
    classifier: ClassifierName = "new",
    cascade_face_ids: set[int] | frozenset[int] | None = None,
) -> list[FaceClassificationRecord]:
    """Classify every B-rep face as STOCK or CUT with diagnostic fields."""
    ctx = cad if isinstance(cad, _CadContext) else _load_cad(cad)
    classify_fn = _classify_old if classifier == "old" else _classify_new
    records: list[FaceClassificationRecord] = []
    for face_idx, geom in enumerate(ctx.geoms):
        label, driving_signal = classify_fn(face_idx, ctx)
        records.append(
            FaceClassificationRecord(
                face_id=face_idx,
                label=label,
                driving_signal=driving_signal,
                convex_summary=_convex_summary(face_idx, ctx),
                envelope_coincident=_envelope_coincident(face_idx, ctx),
                normal_consistent=_normal_consistent(face_idx, ctx),
                surface_type=geom.surface_type,
                boundary_coincident_machined=None,
            )
        )

    stock_ids = {r.face_id for r in records if r.label == "STOCK"}
    cascade_ids = cascade_face_ids or set()
    for record in records:
        record.labeled_by = _derive_labeled_by(
            record.face_id, record.label, stock_ids, cascade_ids,
        )
        record.gate_rule = _gate_rule(record.face_id, ctx, record)

    return records


def diff_classification(
    cad: str | Path | _CadContext,
    *,
    part_id: str | None = None,
    corner_chamfer_ids: set[int] | None = None,
) -> ClassificationDiff:
    """Run old and new classifiers; return faces whose label flipped."""
    ctx = cad if isinstance(cad, _CadContext) else _load_cad(cad)
    old_records = {r.face_id: r for r in classify_report(ctx, classifier="old")}
    new_records = {r.face_id: r for r in classify_report(ctx, classifier="new")}

    flips: list[dict[str, Any]] = []
    stock_to_cut = 0
    cut_to_stock = 0
    for face_id in sorted(old_records):
        old_r = old_records[face_id]
        new_r = new_records[face_id]
        if old_r.label == new_r.label:
            continue
        if old_r.label == "STOCK" and new_r.label == "CUT":
            stock_to_cut += 1
        else:
            cut_to_stock += 1
        flips.append({
            "face_id": face_id,
            "surface_type": old_r.surface_type,
            "old_label": old_r.label,
            "new_label": new_r.label,
            "driving_signal_old": old_r.driving_signal,
            "driving_signal_new": new_r.driving_signal,
            "convex_summary": new_r.convex_summary.to_dict(),
            "envelope_coincident": new_r.envelope_coincident.to_dict(),
            "normal_consistent": new_r.normal_consistent,
            "reason": _flip_reason(
                face_id,
                ctx,
                old_r.label,
                new_r.label,
                old_r.driving_signal,
                new_r.driving_signal,
                new_r.convex_summary,
            ),
        })

    old_stock = sum(1 for r in old_records.values() if r.label == "STOCK")
    new_stock = sum(1 for r in new_records.values() if r.label == "STOCK")
    old_bounding = _count_bounding_stock(
        old_records, ctx, corner_chamfer_ids=corner_chamfer_ids,
    )
    new_bounding = _count_bounding_stock(
        new_records, ctx, corner_chamfer_ids=corner_chamfer_ids,
    )
    return ClassificationDiff(
        part_id=part_id or ctx.step_path.stem,
        n_faces=len(old_records),
        n_flipped=len(flips),
        stock_to_cut=stock_to_cut,
        cut_to_stock=cut_to_stock,
        flips=flips,
        old_stock_count=old_stock,
        new_stock_count=new_stock,
        old_bounding_stock_count=old_bounding,
        new_bounding_stock_count=new_bounding,
    )


def format_flip_report(diff: ClassificationDiff) -> str:
    lines = [
        f"=== {diff.part_id}: {diff.summary_line()} ===",
        (
            "Note: boundary-coincident faces machined per shop op-list are out of scope "
            "for this geometry-only diff (op-list reconciliation TODO)."
        ),
    ]
    if not diff.flips:
        lines.append("(no flips)")
        return "\n".join(lines)
    for flip in diff.flips:
        lines.append(
            f"  face {flip['face_id']:3d}  {flip['old_label']}->{flip['new_label']}  "
            f"{flip['surface_type']:10s}  normal_ok={flip['normal_consistent']}  "
            f"env={flip['envelope_coincident']['coincident']}  "
            f"reason: {flip['reason']}"
        )
        lines.append(
            f"           old={json.dumps(flip['driving_signal_old'], sort_keys=True)}  "
            f"new={json.dumps(flip['driving_signal_new'], sort_keys=True)}"
        )
        lines.append(f"           convex={json.dumps(flip['convex_summary'], sort_keys=True)}")
    return "\n".join(lines)


def envelope_stock_face_ids(
    cad: str | Path | _CadContext,
    *,
    classifier: ClassifierName = "new",
) -> set[int]:
    """B-rep faces labeled STOCK and coincident with the part envelope."""
    records = classify_report(cad, classifier=classifier)
    return {
        r.face_id
        for r in records
        if r.label == "STOCK" and r.envelope_coincident.coincident
    }


def stock_face_ids(
    cad: str | Path | _CadContext,
    *,
    classifier: ClassifierName = "new",
) -> set[int]:
    """Face indices labeled STOCK by the requested classifier."""
    records = classify_report(cad, classifier=classifier)
    return {r.face_id for r in records if r.label == "STOCK"}


def run_fixture_diff(
    fixture_name: str,
    *,
    repo_root: str | Path | None = None,
) -> ClassificationDiff:
    root = Path(repo_root or Path(__file__).resolve().parent)
    spec = CLASSIFICATION_FIXTURES[fixture_name]
    step = root / spec["step"]
    if not step.is_file():
        raise FileNotFoundError(f"missing fixture STEP: {step}")
    corner_ids = spec.get("corner_chamfer_face_ids")
    corner_set = set(corner_ids) if corner_ids else None
    return diff_classification(
        step,
        part_id=fixture_name,
        corner_chamfer_ids=corner_set,
    )
