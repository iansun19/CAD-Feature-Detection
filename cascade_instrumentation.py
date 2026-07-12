"""
Cascade diagnostic instrumentation for run_cascade.py.

Artifacts (under pipeline_out/<part>/instrument/):
  pass_claims.json - per-pass claim log from the real cascade; ``face_owner`` maps
      each claimed face to the single pass and CAM class that owns it.
  contested_faces.json - dry-run overlap report with two variants:
      ``context_influenced`` (full candidate pool, prior-pass context from the
      dry-run chain) and ``no_context`` (full pool, empty cross-pass claim context).
      Use these to spot ordering conflicts (e.g. pockets vs holes on one cylinder).
  gnn_disagreement.json - per-face cascade vs GNN labels where MFCAD_TO_CAM maps
      the GNN class to a comparable CAM label. The mapping is an editable hypothesis,
      not ground truth; ``class_pair_counts`` buckets disagreements by (cascade, gnn).
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from coaxial_stack_detection import CoaxialStackDetectionConfig, detect_coaxial_stack
from flats_detection import FlatDetectionConfig, detect_flats
from hole_detection import HoleDetectionConfig, detect_holes
from outer_fillet_detection import OuterFilletDetectionConfig, detect_outer_fillets
from pocket_detection import (
    PocketDetectionConfig,
    apply_filleted_lobe_tiers_to_result,
    detect_pockets,
    _part_axis_top,
)
from profile_detection import ProfileDetectionConfig, detect_profiles
from residual_detection import ResidualDetectionConfig, detect_residual_candidates
from wall_detection import WallDetectionConfig, detect_walls

logger = logging.getLogger("cascade_instrumentation")

# Editable hypothesis: MFCAD++ 12-class names -> cascade CAM / Toolpath labels.
# None = skip comparison for that GNN class.
MFCAD_TO_CAM: dict[str, str | None] = {
    "through_hole": "through_hole",
    "blind_hole": "filleted_blind_hole",
    "blind_pocket": "filleted_pocket",
    "round_fillet": "outer_fillet",
    "through_step": "profile",
    "blind_step": "profile",
    "poly_through_passage": None,
    "through_slot": None,
    "o_ring": None,
    "blind_slot": None,
    "chamfer": None,
    "stock": None,
}

CASCADE_KIND_TO_TP: dict[str, str] = {
    "pocket": "filleted_pocket",
    "through_hole": "through_hole",
    "blind_hole": "filleted_blind_hole",
    "flat": "flat",
    "outer_fillet": "outer_fillet",
}

PASS_ORDER: list[tuple[str, str]] = [
    ("pockets", "pocket_detection"),
    ("holes", "hole_detection"),
    ("coaxial_stack", "coaxial_stack_detection"),
    ("flats", "flats_detection"),
    ("outer_fillets", "outer_fillet_detection"),
    ("wall", "wall_detection"),
    ("profile", "profile_detection"),
    ("residual_candidates", "residual_detection"),
]

TIER_FIELD_SEMANTICS: dict[str, str] = {
    "tier_label": "CAM emit class (filleted_open_pocket / filleted_pocket).",
    "tier_source": (
        "setup = label from run config (--pocket-access / --machining-side / default); "
        "lobe_tier = label from axial depth-band geometry after lobe tier split."
    ),
    "setup_access": (
        "Config-resolved machining access (open/closed). Only when tier_source=setup. "
        "Distinct from band."
    ),
    "setup_access_origin": (
        "How setup_access was resolved: pocket_access, machining_side, or default."
    ),
    "band": (
        "lobe_tier only: mouth (shallow/mouth-step band) or deep (deep-step band). "
        "Reflects axial position on the loaded STEP slice, NOT CAM machining access."
    ),
    "band_axial_mm": "Face centroid projection on opening_axis (mm). lobe_tier only.",
    "mouth_axial_mm": "Reference mouth-step axial position for this lobe (mm).",
    "deep_axial_mm": "Reference deep-step axial position for this lobe (mm).",
    "tier_assignment": (
        "lobe_tier only: how the face entered its band "
        "(region_grow, annexed_fillet, overlap_resolved, closed_tier_open_ext, orphan_reassigned). "
        "Null when trace partition disagrees with real lobe split or trace failed."
    ),
}


@dataclass
class PassLabelContext:
    faces: Sequence[Any] | None = None
    edge_index: np.ndarray | None = None
    edge_attr: np.ndarray | None = None
    occ_faces: Sequence[Any] | None = None
    pocket_config: PocketDetectionConfig | None = None


def resolve_part_id(step_path: Path, part_id: str | None = None) -> str:
    pid = part_id or step_path.stem.replace(" ", "_").split(".")[0]
    if "96260B" in pid.upper() and "FRONT" in step_path.name.upper():
        return "96260B_front"
    return pid


def resolve_instrument_dir(
    step_path: Path,
    *,
    export_dir: Path | None = None,
    part_id: str | None = None,
) -> Path:
    pid = resolve_part_id(step_path, part_id)
    base = export_dir if export_dir is not None else Path("pipeline_out") / pid
    return base / "instrument"


def _hole_tp_class(kind: str) -> str:
    return CASCADE_KIND_TO_TP.get(kind, kind)


def _class_from_label_entry(entry: dict[str, Any] | str) -> str:
    if isinstance(entry, str):
        return entry
    return str(entry["class"])


def _resolve_setup_access(
    setup: Any,
) -> tuple[str | None, str | None]:
    """Return (setup_access, setup_access_origin) without detector edits."""
    pocket_access = getattr(setup, "pocket_access", None)
    machining_side = getattr(setup, "machining_side", None)
    if pocket_access is not None:
        return str(pocket_access), "pocket_access"
    if machining_side == "front":
        return "open", "machining_side"
    if machining_side == "back":
        return "closed", "machining_side"
    access, explicit = setup.resolved_access()
    if explicit:
        return str(access), "machining_side"
    return str(access), "default"


def _face_axial_mm(
    by_index: dict[int, Any],
    face_id: int,
    opening_axis: np.ndarray,
) -> float:
    from lobe_tier_detection import _axial_y

    return float(_axial_y(by_index, face_id, opening_axis))


def _band_from_tier_label(tier_label: str) -> str | None:
    if tier_label == "filleted_open_pocket":
        return "mouth"
    if tier_label == "filleted_pocket":
        return "deep"
    return None


def _trace_split_lobe_pool_assignments(
    pool: set[int],
    mouth_step: int,
    deep_step: int,
    graph: Any,
    by_index: dict[int, Any],
    occ_map: dict[int, Any] | None,
    opening_axis: np.ndarray,
    cfg: Any,
) -> tuple[set[int], set[int], dict[int, str]]:
    """Replay split_lobe_pool steps; return final sets and per-face assignment reasons."""
    from lobe_tier_detection import (
        _annex_fillets,
        _annex_sculpt_cap_bsplines,
        _axial_side_of_mouth,
        _closed_tier_opening_extension_wall,
        _migrate_closed_tier_opening_extension_walls,
        _migrate_sculpt_cap_bsplines_follow_open_bridge,
        _mouth_tier_boundary_tol_mm,
        _open_fillet_should_drop,
        _prune_closed_tier_fillets,
        _prune_open_tier_fillets,
        _prune_paired_floor_spheres,
        _region_grow_tier,
        _reassign_orphan_lobe_faces,
        _resolve_overlap,
    )

    assignments: dict[int, str] = {}
    open_seeds, closed_seeds = {mouth_step}, {deep_step}
    deep_steps = {deep_step}
    mouth_boundary_tol = _mouth_tier_boundary_tol_mm(
        pool, mouth_step, deep_step, by_index, occ_map, graph, opening_axis, cfg,
    )

    open_faces = _region_grow_tier(
        open_seeds, pool, graph, by_index, occ_map, opening_axis, cfg,
        tier_hint="open", mouth_step=mouth_step,
        mouth_boundary_tol_mm=mouth_boundary_tol, deep_step_faces=deep_steps,
    )
    closed_faces = _region_grow_tier(
        closed_seeds, pool, graph, by_index, occ_map, opening_axis, cfg,
        tier_hint="closed", mouth_step=mouth_step,
        mouth_boundary_tol_mm=mouth_boundary_tol, deep_step_faces=deep_steps,
    )
    for face_id in open_faces | closed_faces:
        assignments[face_id] = "region_grow"

    prev_open = set(open_faces)
    open_faces, sphere_cap_logs = _annex_fillets(
        open_faces, pool, graph, by_index,
        occ_map=occ_map,
        cfg=cfg,
        skip=lambda f: _open_fillet_should_drop(
            f, open_faces, graph, by_index, occ_map, cfg, deep_steps,
        ),
    )
    concave_caps = {
        int(entry["sphere_face"])
        for entry in sphere_cap_logs
        if entry.get("edge_kind") == "concave"
    }
    prev_open_before_bspline = set(open_faces)
    open_faces = _annex_sculpt_cap_bsplines(
        open_faces, pool, graph, by_index, opening_axis, cfg,
        mouth_step=mouth_step,
        mouth_boundary_tol_mm=mouth_boundary_tol,
        concave_annexed_cap_spheres=concave_caps,
    )
    for face_id in open_faces - prev_open:
        if any(entry["sphere_face"] == face_id for entry in sphere_cap_logs):
            assignments[face_id] = "annexed_fillet_sphere_cap"
        elif face_id in prev_open_before_bspline:
            assignments[face_id] = "annexed_fillet"
        else:
            assignments[face_id] = "annexed_sculpt_cap_bspline"

    prev_closed = set(closed_faces)
    closed_faces, _ = _annex_fillets(
        closed_faces, pool, graph, by_index,
        occ_map=occ_map,
        cfg=cfg,
        skip=lambda f: _axial_side_of_mouth(
            f, mouth_step, mouth_boundary_tol, by_index, opening_axis,
        ) == "open",
    )
    if sphere_cap_logs:
        closed_faces -= {entry["sphere_face"] for entry in sphere_cap_logs}
    for face_id in closed_faces - prev_closed:
        assignments[face_id] = "annexed_fillet"

    open_faces = _prune_open_tier_fillets(
        open_faces, graph, by_index, occ_map, cfg, deep_steps,
    )
    closed_faces = _prune_paired_floor_spheres(
        closed_faces, by_index, graph, opening_axis,
    )
    closed_faces = _prune_closed_tier_fillets(
        closed_faces, mouth_step, deep_step, graph, by_index, occ_map, cfg, opening_axis,
    )

    overlap = open_faces & closed_faces
    open_faces, closed_faces = _resolve_overlap(
        open_faces, closed_faces, open_seeds, closed_seeds,
        by_index, occ_map, graph, opening_axis, cfg,
        mouth_step=mouth_step,
        mouth_boundary_tol_mm=mouth_boundary_tol,
        deep_step_faces=deep_steps,
    )
    for face_id in overlap:
        if face_id in open_faces or face_id in closed_faces:
            assignments[face_id] = "overlap_resolved"

    open_faces, closed_faces = _migrate_closed_tier_opening_extension_walls(
        pool, open_faces, closed_faces, mouth_step,
        by_index, occ_map, graph, opening_axis, cfg,
        mouth_boundary_tol_mm=mouth_boundary_tol,
    )
    for face_id in sorted(pool):
        if face_id not in closed_faces:
            continue
        if _closed_tier_opening_extension_wall(
            face_id, mouth_step, mouth_boundary_tol,
            by_index, occ_map, graph, opening_axis, cfg,
        ):
            assignments[face_id] = "closed_tier_open_ext"

    before_orphan = open_faces | closed_faces
    open_faces, closed_faces = _reassign_orphan_lobe_faces(
        pool, open_faces, closed_faces, mouth_step, deep_step,
        by_index, occ_map, graph, opening_axis, cfg,
        mouth_boundary_tol_mm=mouth_boundary_tol,
        concave_annexed_cap_spheres=concave_caps,
    )
    for face_id in (open_faces | closed_faces) - before_orphan:
        assignments[face_id] = "orphan_reassigned"

    closed_before_bridge = set(closed_faces)
    open_faces, closed_faces = _migrate_sculpt_cap_bsplines_follow_open_bridge(
        pool, open_faces, closed_faces, graph, by_index,
    )
    for face_id in closed_before_bridge - closed_faces:
        if face_id in open_faces:
            assignments[face_id] = "sculpt_cap_follow_open_bridge"

    return open_faces, closed_faces, assignments


def _lobe_tier_provenance(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    occ_faces: Sequence[Any] | None,
    opening_axis: Sequence[float],
    *,
    min_lobes: int = 6,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    """Build lobe-tier partition (band/axial) and optional assignment trace."""
    from lobe_tier_detection import LobeTierConfig, detect_filleted_lobe_tiers
    from pocket_detection import FaceGraph

    partition: dict[int, dict[str, Any]] = {}
    assignments: dict[int, str] = {}

    try:
        lobe = detect_filleted_lobe_tiers(
            faces, edge_index, edge_attr,
            occ_faces=occ_faces,
            opening_axis=opening_axis,
            config=LobeTierConfig(),
        )
    except Exception as exc:
        logger.warning("lobe tier partition failed: %s", exc)
        return partition, assignments

    if len(lobe.lobes) < min_lobes:
        return partition, assignments

    axis = np.asarray(opening_axis, dtype=np.float64)
    by_index = {int(f.index): f for f in faces}
    n_faces = len(faces)
    occ_map = {i: occ_faces[i] for i in range(n_faces)} if occ_faces else None
    graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)
    cfg = LobeTierConfig()

    real_open: dict[int, str] = {}
    real_closed: dict[int, str] = {}
    trace_open: dict[int, str] = {}
    trace_closed: dict[int, str] = {}
    trace_assignments: dict[int, str] = {}
    trace_ok = True

    for lob in lobe.lobes:
        lobe_meta = {
            "lobe_id": lob.lobe_id,
            "mouth_axial_mm": round(float(lob.mouth_axial), 4),
            "deep_axial_mm": round(float(lob.deep_axial), 4),
        }
        for face_id in lob.open_faces:
            real_open[int(face_id)] = "mouth"
            partition[int(face_id)] = {
                **lobe_meta,
                "band": "mouth",
                "band_axial_mm": round(_face_axial_mm(by_index, int(face_id), axis), 4),
            }
        for face_id in lob.closed_faces:
            real_closed[int(face_id)] = "deep"
            partition[int(face_id)] = {
                **lobe_meta,
                "band": "deep",
                "band_axial_mm": round(_face_axial_mm(by_index, int(face_id), axis), 4),
            }

        try:
            tr_open, tr_closed, tr_assign = _trace_split_lobe_pool_assignments(
                lob.pool_faces, lob.mouth_step_face, lob.deep_step_face,
                graph, by_index, occ_map, axis, cfg,
            )
        except Exception as exc:
            logger.warning(
                "lobe tier assignment trace failed for lobe %d: %s",
                lob.lobe_id, exc,
            )
            trace_ok = False
            continue

        for face_id in tr_open:
            trace_open[int(face_id)] = "mouth"
        for face_id in tr_closed:
            trace_closed[int(face_id)] = "deep"
        trace_assignments.update(tr_assign)

    if trace_ok:
        divergent: set[int] = set()
        all_faces = set(real_open) | set(real_closed) | set(trace_open) | set(trace_closed)
        for face_id in all_faces:
            if real_open.get(face_id) != trace_open.get(face_id):
                divergent.add(face_id)
            if real_closed.get(face_id) != trace_closed.get(face_id):
                divergent.add(face_id)
        if divergent:
            logger.warning(
                "lobe tier trace partition diverges from real on %d face(s); "
                "nulling tier_assignment for those faces",
                len(divergent),
            )
        for face_id, reason in trace_assignments.items():
            if face_id in divergent:
                continue
            assignments[face_id] = reason

    return partition, assignments


def _pocket_tier_fields(
    face_id: int,
    tier_label: str,
    *,
    tier_source: str,
    setup: Any | None,
    lobe_partition: dict[int, dict[str, Any]],
    lobe_assignments: dict[int, str],
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tier_label": tier_label,
        "tier_source": tier_source,
        "setup_access": None,
        "setup_access_origin": None,
        "lobe_id": None,
        "band": None,
        "band_axial_mm": None,
        "mouth_axial_mm": None,
        "deep_axial_mm": None,
        "tier_assignment": None,
    }

    if tier_source == "setup":
        if setup is not None:
            access, origin = _resolve_setup_access(setup)
            fields["setup_access"] = access
            fields["setup_access_origin"] = origin
        return fields

    part = lobe_partition.get(face_id)
    if part is not None:
        fields["lobe_id"] = part.get("lobe_id")
        fields["band"] = part.get("band")
        fields["band_axial_mm"] = part.get("band_axial_mm")
        fields["mouth_axial_mm"] = part.get("mouth_axial_mm")
        fields["deep_axial_mm"] = part.get("deep_axial_mm")

    expected_band = _band_from_tier_label(tier_label)
    actual_band = fields.get("band")
    assignment = lobe_assignments.get(face_id)
    if (
        expected_band is not None
        and actual_band is not None
        and expected_band != actual_band
    ):
        logger.warning(
            "pocket face %d: tier_label implies band=%s but lobe partition has band=%s; "
            "nulling tier_assignment",
            face_id, expected_band, actual_band,
        )
        assignment = None

    fields["tier_assignment"] = assignment
    return fields


def face_labels_from_pass_result(
    pass_key: str,
    result: Any,
    *,
    label_ctx: PassLabelContext | None = None,
) -> dict[int, dict[str, Any]]:
    """Map claimed face indices to label records for one cascade pass result."""
    labels: dict[int, dict[str, Any]] = {}
    features = getattr(result, "features", None) or []

    if pass_key == "pockets":
        ctx = label_ctx or PassLabelContext()
        cfg = ctx.pocket_config or PocketDetectionConfig()
        setup = cfg.setup
        lobe_partition: dict[int, dict[str, Any]] = {}
        lobe_assignments: dict[int, str] = {}
        if (
            ctx.faces is not None
            and ctx.edge_index is not None
            and ctx.edge_attr is not None
        ):
            lobe_partition, lobe_assignments = _lobe_tier_provenance(
                ctx.faces,
                ctx.edge_index,
                ctx.edge_attr,
                ctx.occ_faces,
                result.opening_axis,
            )

        for feat in features:
            tier_label = getattr(feat, "toolpath_class", "filleted_pocket")
            is_lobe_tier = bool(
                getattr(feat, "template_deviation", {}).get("lobe_tier")
            )
            tier_source = "lobe_tier" if is_lobe_tier else "setup"
            for face_id in feat.face_indices:
                fid = int(face_id)
                entry: dict[str, Any] = {
                    "class": tier_label,
                    **_pocket_tier_fields(
                        fid,
                        tier_label,
                        tier_source=tier_source,
                        setup=setup if tier_source == "setup" else None,
                        lobe_partition=lobe_partition,
                        lobe_assignments=lobe_assignments,
                    ),
                }
                labels[fid] = entry
        return labels

    if pass_key == "holes":
        for feat in features:
            cls = _hole_tp_class(feat.kind)
            for i in feat.face_indices:
                labels[int(i)] = {"class": cls}
    elif pass_key == "coaxial_stack":
        for feat in features:
            cls = feat.toolpath_class
            for i in feat.face_indices:
                labels[int(i)] = {"class": cls}
    elif pass_key == "flats":
        for feat in features:
            for i in feat.face_indices:
                labels[int(i)] = {"class": "flat"}
    elif pass_key == "outer_fillets":
        for feat in features:
            cls = feat.toolpath_class
            for i in feat.face_indices:
                labels[int(i)] = {"class": cls}
    elif pass_key == "wall":
        for feat in features:
            cls = feat.toolpath_class
            for i in feat.face_indices:
                labels[int(i)] = {"class": cls}
    elif pass_key == "profile":
        for feat in features:
            cls = feat.toolpath_class
            for i in feat.face_indices:
                labels[int(i)] = {"class": cls}
    elif pass_key == "residual_candidates":
        for feat in features:
            cls = getattr(feat, "toolpath_class", "contour_surface")
            for i in feat.face_indices:
                labels[int(i)] = {"class": cls}
    return labels


def cascade_face_labels(
    pocket_result: Any,
    hole_result: Any,
    coaxial_result: Any,
    flat_result: Any,
    outer_fillet_result: Any,
    wall_result: Any,
    profile_result: Any,
    residual_result: Any,
    *,
    label_ctx: PassLabelContext | None = None,
) -> dict[int, str]:
    """Merge per-pass labels into a single face -> CAM class map (real cascade)."""
    results = (
        ("pockets", pocket_result),
        ("holes", hole_result),
        ("coaxial_stack", coaxial_result),
        ("flats", flat_result),
        ("outer_fillets", outer_fillet_result),
        ("wall", wall_result),
        ("profile", profile_result),
        ("residual_candidates", residual_result),
    )
    merged: dict[int, str] = {}
    for pass_key, result in results:
        ctx = label_ctx if pass_key == "pockets" else None
        for face_id, entry in face_labels_from_pass_result(
            pass_key, result, label_ctx=ctx,
        ).items():
            merged[face_id] = _class_from_label_entry(entry)
    return merged


@dataclass
class PassClaimRecord:
    pass_name: str
    index: int
    module: str
    input_pool_size: int
    output_pool_size: int
    n_claimed: int
    claimed: dict[str, dict[str, Any]] = field(default_factory=dict)


class PassClaimRecorder:
    """Collect per-pass claim records during run_cascade()."""

    def __init__(self) -> None:
        self.passes: list[PassClaimRecord] = []

    def record_pass(
        self,
        pass_name: str,
        index: int,
        module: str,
        input_pool_size: int,
        result: Any,
        *,
        label_ctx: PassLabelContext | None = None,
    ) -> None:
        labels = face_labels_from_pass_result(
            pass_name, result, label_ctx=label_ctx,
        )
        claimed = result.claimed_faces
        self.passes.append(PassClaimRecord(
            pass_name=pass_name,
            index=index,
            module=module,
            input_pool_size=input_pool_size,
            output_pool_size=len(result.remaining_faces),
            n_claimed=len(claimed),
            claimed={str(i): labels[i] for i in sorted(claimed) if i in labels},
        ))

    def to_dict(self) -> dict[str, Any]:
        face_owner: dict[str, dict[str, Any]] = {}
        for rec in self.passes:
            for face_str, entry in rec.claimed.items():
                face_owner[face_str] = {
                    "pass": rec.pass_name,
                    "index": rec.index,
                    **entry,
                }
        return {
            "tier_field_semantics": TIER_FIELD_SEMANTICS,
            "passes": [
                {
                    "pass": rec.pass_name,
                    "index": rec.index,
                    "module": rec.module,
                    "input_pool_size": rec.input_pool_size,
                    "output_pool_size": rec.output_pool_size,
                    "n_claimed": rec.n_claimed,
                    "claimed": rec.claimed,
                }
                for rec in self.passes
            ],
            "face_owner": face_owner,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")


def _hub_open_pocket_faces(coaxial_result: Any) -> set[int]:
    out: set[int] = set()
    for feat in coaxial_result.features:
        if feat.kind == "open_pocket":
            out |= feat.face_indices
    return out


def _dry_run_cascade(
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    occ_faces: Sequence[Any] | None,
    n_faces: int,
    *,
    full_pool: set[int],
    context_mode: str,
    pocket_config: PocketDetectionConfig | None,
    hole_config: HoleDetectionConfig | None,
    coaxial_config: CoaxialStackDetectionConfig | None,
    flat_config: FlatDetectionConfig | None,
    outer_fillet_config: OuterFilletDetectionConfig | None,
    wall_config: WallDetectionConfig | None,
    profile_config: ProfileDetectionConfig | None,
    residual_config: ResidualDetectionConfig | None,
) -> tuple[dict[str, set[int]], Any]:
    """Run all passes with candidate_faces=full_pool; return (wanted, pocket_result)."""
    use_context = context_mode == "context_influenced"
    cfg_pocket = pocket_config or PocketDetectionConfig()

    pocket_result = detect_pockets(
        faces, edge_index, edge_attr,
        occ_faces=occ_faces, candidate_faces=full_pool,
        config=pocket_config,
    )
    pocket_result = apply_filleted_lobe_tiers_to_result(
        pocket_result, faces, edge_index, edge_attr, occ_faces, cfg_pocket,
    )

    hole_result = detect_holes(
        faces, edge_index, edge_attr,
        occ_faces=occ_faces, candidate_faces=full_pool,
        config=hole_config,
    )

    occ_map = {i: occ_faces[i] for i in range(n_faces)} if occ_faces else None
    opening_axis = np.asarray(pocket_result.opening_axis, dtype=float)
    part_axis_top = _part_axis_top(faces, opening_axis, occ_map)

    if use_context:
        cx_hole_claimed = hole_result.claimed_faces
        cx_hole_features = hole_result.features
    else:
        cx_hole_claimed = set()
        cx_hole_features = []

    coaxial_result = detect_coaxial_stack(
        faces, edge_index, edge_attr,
        candidate_faces=full_pool,
        hole_claimed_faces=cx_hole_claimed,
        hole_features=cx_hole_features,
        opening_axis=pocket_result.opening_axis,
        part_axis_top=part_axis_top,
        config=coaxial_config,
    )

    hub_open = _hub_open_pocket_faces(coaxial_result)

    if use_context:
        fl_hole = hole_result.claimed_faces
        fl_pocket = pocket_result.claimed_faces
        fl_floor = pocket_result.floor_absorbed_faces
        fl_hub_flat = coaxial_result.hub_flat_faces
    else:
        fl_hole = fl_pocket = fl_floor = fl_hub_flat = set()

    flat_result = detect_flats(
        faces, edge_index, edge_attr,
        candidate_faces=full_pool,
        hole_claimed_faces=fl_hole,
        pocket_claimed_faces=fl_pocket,
        pocket_floor_absorbed_faces=fl_floor,
        hub_flat_faces=fl_hub_flat,
        opening_axis=pocket_result.opening_axis,
        occ_faces=occ_faces,
        config=flat_config,
    )

    if use_context:
        of_pocket = pocket_result.claimed_faces
        of_stack = hole_result.deferred_feature_fillet_groups
        of_flat = flat_result.claimed_faces
        of_hub_ctx = coaxial_result.hub_perimeter_context
    else:
        of_pocket = of_flat = set()
        of_stack = []
        of_hub_ctx = None

    outer_fillet_result = detect_outer_fillets(
        faces, edge_index, edge_attr,
        candidate_faces=full_pool,
        pocket_claimed_faces=of_pocket,
        stack_boundary_fillet_groups=of_stack,
        flat_claimed_faces=of_flat,
        hub_open_pocket_faces=hub_open,
        hub_perimeter_context=of_hub_ctx,
        opening_axis=pocket_result.opening_axis,
        part_axis_top=part_axis_top,
        config=outer_fillet_config,
    )

    if use_context:
        wl_pocket = pocket_result.claimed_faces
        wl_hole = hole_result.claimed_faces
    else:
        wl_pocket = wl_hole = set()

    wall_result = detect_walls(
        faces, edge_index, edge_attr,
        candidate_faces=full_pool,
        pocket_claimed_faces=wl_pocket,
        hole_claimed_faces=wl_hole,
        hub_open_pocket_faces=hub_open,
        opening_axis=pocket_result.opening_axis,
        config=wall_config,
    )

    if use_context:
        pr_pocket = pocket_result.claimed_faces
        pr_hole = hole_result.claimed_faces
        pr_wall = wall_result.claimed_faces
        pr_seeds = wall_result.seed_faces
    else:
        pr_pocket = pr_hole = pr_wall = pr_seeds = set()

    profile_result = detect_profiles(
        faces, edge_index, edge_attr,
        candidate_faces=full_pool,
        pocket_claimed_faces=pr_pocket,
        hole_claimed_faces=pr_hole,
        hub_open_pocket_faces=hub_open,
        wall_claimed_faces=pr_wall,
        wall_seed_faces=pr_seeds,
        opening_axis=pocket_result.opening_axis,
        config=profile_config,
    )

    residual_result = detect_residual_candidates(
        faces, edge_index, edge_attr,
        candidate_faces=full_pool,
        config=residual_config,
    )

    results = (
        pocket_result, hole_result, coaxial_result, flat_result,
        outer_fillet_result, wall_result, profile_result, residual_result,
    )
    wanted: dict[str, set[int]] = {}
    for (pass_name, _), result in zip(PASS_ORDER, results):
        wanted[pass_name] = set(result.claimed_faces)
    return wanted, pocket_result


def _contested_from_wanted(
    wanted: dict[str, set[int]],
    *,
    pocket_result: Any,
    label_ctx: PassLabelContext,
) -> dict[str, Any]:
    face_to_passes: dict[int, list[str]] = {}
    for pass_name, faces in wanted.items():
        for face in faces:
            face_to_passes.setdefault(face, []).append(pass_name)

    contested: dict[str, list[str]] = {}
    pair_counts: Counter[tuple[str, str]] = Counter()
    for face, passes in face_to_passes.items():
        if len(passes) < 2:
            continue
        passes_sorted = sorted(passes)
        contested[str(face)] = passes_sorted
        for a, b in combinations(passes_sorted, 2):
            pair_counts[(a, b)] += 1

    top_pairs = [
        {"pass_a": a, "pass_b": b, "count": count}
        for (a, b), count in pair_counts.most_common(10)
    ]
    pocket_labels = face_labels_from_pass_result(
        "pockets", pocket_result, label_ctx=label_ctx,
    )
    pocket_face_labels = {
        str(face_id): entry
        for face_id, entry in sorted(pocket_labels.items())
        if face_id in wanted.get("pockets", set())
    }
    return {
        "total_contested_faces": len(contested),
        "top_pass_pairs": top_pairs,
        "contested_faces": contested,
        "pass_wanted": {k: sorted(v) for k, v in wanted.items()},
        "pocket_face_labels": pocket_face_labels,
    }


def write_contested_faces(
    path: Path,
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    occ_faces: Sequence[Any] | None,
    n_faces: int,
    *,
    pocket_config: PocketDetectionConfig | None = None,
    hole_config: HoleDetectionConfig | None = None,
    coaxial_config: CoaxialStackDetectionConfig | None = None,
    flat_config: FlatDetectionConfig | None = None,
    outer_fillet_config: OuterFilletDetectionConfig | None = None,
    wall_config: WallDetectionConfig | None = None,
    profile_config: ProfileDetectionConfig | None = None,
    residual_config: ResidualDetectionConfig | None = None,
) -> None:
    full_pool = set(range(n_faces))
    common = dict(
        faces=faces,
        edge_index=edge_index,
        edge_attr=edge_attr,
        occ_faces=occ_faces,
        n_faces=n_faces,
        full_pool=full_pool,
        pocket_config=pocket_config,
        hole_config=hole_config,
        coaxial_config=coaxial_config,
        flat_config=flat_config,
        outer_fillet_config=outer_fillet_config,
        wall_config=wall_config,
        profile_config=profile_config,
        residual_config=residual_config,
    )

    variants: dict[str, Any] = {}
    summaries: list[str] = []

    for mode, description in (
        (
            "context_influenced",
            "Full candidate pool; cross-pass context from the dry-run chain "
            "(same kwargs wiring as run_cascade, but candidate_faces=all faces).",
        ),
        (
            "no_context",
            "Full candidate pool; empty cross-pass claim context "
            "(opening_axis/part_axis_top still derived from geometry).",
        ),
    ):
        wanted, pocket_result = _dry_run_cascade(**common, context_mode=mode)
        label_ctx = PassLabelContext(
            faces=faces,
            edge_index=edge_index,
            edge_attr=edge_attr,
            occ_faces=occ_faces,
            pocket_config=pocket_config,
        )
        report = _contested_from_wanted(
            wanted, pocket_result=pocket_result, label_ctx=label_ctx,
        )
        report["description"] = description
        variants[mode] = report
        summaries.append(
            f"  {mode}: {report['total_contested_faces']} contested faces"
        )
        if report["top_pass_pairs"]:
            top = report["top_pass_pairs"][0]
            summaries.append(
                f"    top pair: {top['pass_a']} vs {top['pass_b']} "
                f"({top['count']} faces)"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tier_field_semantics": TIER_FIELD_SEMANTICS,
        "variants": variants,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print("\nINSTRUMENT - contested faces (dry-run, no claiming)")
    for line in summaries:
        print(line)


def _resolve_gnn_artifact(
    instrument_dir: Path,
    graph_npz: Path | None,
    part_id: str,
) -> Path | None:
    candidates: list[Path] = [
        instrument_dir.parent / "face_predictions.jsonl",
    ]
    if graph_npz is not None:
        candidates.append(graph_npz.parent / "face_predictions.jsonl")
    candidates.append(Path("pipeline_out") / part_id / "face_predictions.jsonl")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _load_gnn_predictions(path: Path, n_faces: int) -> dict[int, str] | None:
    by_face: dict[int, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("GNN artifact %s line %d: invalid JSON (%s)", path, line_no, exc)
                return None
            face_id = int(rec["face_id"])
            by_face[face_id] = str(rec.get("class_name", ""))

    if len(by_face) != n_faces:
        logger.warning(
            "GNN artifact %s: face count mismatch (artifact=%d, cascade=%d); skipping",
            path, len(by_face), n_faces,
        )
        return None
    for i in range(n_faces):
        if i not in by_face:
            logger.warning(
                "GNN artifact %s: missing face_id %d; skipping comparison", path, i,
            )
            return None
    return by_face


def write_gnn_disagreement(
    path: Path,
    cascade_labels: dict[int, str],
    n_faces: int,
    *,
    instrument_dir: Path,
    graph_npz: Path | None,
    part_id: str,
) -> None:
    gnn_path = _resolve_gnn_artifact(instrument_dir, graph_npz, part_id)
    if gnn_path is None:
        logger.warning("No GNN face_predictions.jsonl found; skipping gnn_disagreement.json")
        return

    print(f"INSTRUMENT - GNN artifact: {gnn_path}")

    gnn_by_face = _load_gnn_predictions(gnn_path, n_faces)
    if gnn_by_face is None:
        return

    face_rows: list[dict[str, Any]] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    n_comparable = 0
    n_disagreements = 0

    for face_id in range(n_faces):
        gnn_class = gnn_by_face[face_id]
        gnn_mapped = MFCAD_TO_CAM.get(gnn_class)
        cascade_class = cascade_labels.get(face_id)
        row: dict[str, Any] = {
            "face_id": face_id,
            "cascade_class": cascade_class,
            "gnn_class": gnn_class,
            "gnn_mapped": gnn_mapped,
            "comparable": False,
            "disagreement": False,
        }
        if gnn_mapped is None or cascade_class is None:
            face_rows.append(row)
            continue
        n_comparable += 1
        row["comparable"] = True
        if cascade_class != gnn_mapped:
            row["disagreement"] = True
            n_disagreements += 1
            pair_counts[(cascade_class, gnn_mapped)] += 1
        face_rows.append(row)

    class_pair_counts = [
        {
            "cascade_class": cascade,
            "gnn_mapped_class": gnn,
            "count": count,
        }
        for (cascade, gnn), count in pair_counts.most_common()
    ]

    payload = {
        "gnn_artifact": str(gnn_path),
        "mapping_hypothesis": "MFCAD_TO_CAM in cascade_instrumentation.py (untrusted)",
        "n_faces": n_faces,
        "n_comparable": n_comparable,
        "n_disagreements": n_disagreements,
        "class_pair_counts": class_pair_counts,
        "faces": face_rows,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(
        f"INSTRUMENT - GNN disagreements: {n_disagreements} / {n_comparable} comparable faces"
    )
    if class_pair_counts:
        top = class_pair_counts[0]
        print(
            f"  top conflict: cascade={top['cascade_class']} vs "
            f"gnn={top['gnn_mapped_class']} ({top['count']} faces)"
        )


def run_instrumentation(
    instrument_dir: Path,
    *,
    n_faces: int,
    faces: Sequence[Any],
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    occ_faces: Sequence[Any] | None,
    pocket_result: Any,
    hole_result: Any,
    coaxial_result: Any,
    flat_result: Any,
    outer_fillet_result: Any,
    wall_result: Any,
    profile_result: Any,
    residual_result: Any,
    graph_npz: Path | None = None,
    part_id: str = "",
    pocket_config: PocketDetectionConfig | None = None,
    hole_config: HoleDetectionConfig | None = None,
    coaxial_config: CoaxialStackDetectionConfig | None = None,
    flat_config: FlatDetectionConfig | None = None,
    outer_fillet_config: OuterFilletDetectionConfig | None = None,
    wall_config: WallDetectionConfig | None = None,
    profile_config: ProfileDetectionConfig | None = None,
    residual_config: ResidualDetectionConfig | None = None,
) -> None:
    """Emit contested_faces.json and gnn_disagreement.json (pass_claims written in run_cascade)."""
    instrument_dir.mkdir(parents=True, exist_ok=True)

    label_ctx = PassLabelContext(
        faces=faces,
        edge_index=edge_index,
        edge_attr=edge_attr,
        occ_faces=occ_faces,
        pocket_config=pocket_config,
    )
    cascade_labels = cascade_face_labels(
        pocket_result, hole_result, coaxial_result, flat_result,
        outer_fillet_result, wall_result, profile_result, residual_result,
        label_ctx=label_ctx,
    )

    write_contested_faces(
        instrument_dir / "contested_faces.json",
        faces, edge_index, edge_attr, occ_faces, n_faces,
        pocket_config=pocket_config,
        hole_config=hole_config,
        coaxial_config=coaxial_config,
        flat_config=flat_config,
        outer_fillet_config=outer_fillet_config,
        wall_config=wall_config,
        profile_config=profile_config,
        residual_config=residual_config,
    )

    write_gnn_disagreement(
        instrument_dir / "gnn_disagreement.json",
        cascade_labels,
        n_faces,
        instrument_dir=instrument_dir,
        graph_npz=graph_npz,
        part_id=part_id,
    )


def cascade_result_fingerprint(
    pocket_result: Any,
    hole_result: Any,
    coaxial_result: Any,
    flat_result: Any,
    outer_fillet_result: Any,
    wall_result: Any,
    profile_result: Any,
    residual_result: Any,
) -> str:
    """Deterministic JSON fingerprint of cascade claim state (for regression checks)."""

    def _pass_fp(name: str, result: Any) -> dict[str, Any]:
        return {
            "pass": name,
            "claimed": sorted(int(i) for i in result.claimed_faces),
            "remaining": sorted(int(i) for i in result.remaining_faces),
            "n_features": len(result.features),
        }

    results = (
        ("pockets", pocket_result),
        ("holes", hole_result),
        ("coaxial_stack", coaxial_result),
        ("flats", flat_result),
        ("outer_fillets", outer_fillet_result),
        ("wall", wall_result),
        ("profile", profile_result),
        ("residual_candidates", residual_result),
    )
    payload = [_pass_fp(name, res) for name, res in results]
    return json.dumps(payload, sort_keys=True)
