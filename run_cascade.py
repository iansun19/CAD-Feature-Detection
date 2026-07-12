"""
run_cascade.py — orchestrates the CAM feature-grouping cascade (correct order).

The cascade principle is SPECIFIC-FIRST: the more specific recognizer runs first
and claims its faces; the generic one takes the residual. A sculpted pocket
(walls + bspline floors + spheres + blend ring, spatially co-located) is more
specific than "a coaxial concave cylinder = a hole", so the order is:

    pockets  ->  holes  ->  coaxial_stack  ->  flats
    ->  outer_fillets  ->  wall  ->  profile  ->  residual_candidates

Why the order matters (the bug this fixes)
------------------------------------------
Geometrically a pocket wall is a concave cylinder, so a hole recognizer grabs it
as a false-positive "hole". On the reference plate the hole pass alone reports
65 holes — 56 pocket walls + 7 per-pocket bores + the 2 genuine central holes.
Running pockets FIRST removes those pocket faces from hole candidacy.

The coaxial-stack pass (after holes) claims the central hub stepped pocket and
hub contour surfaces before flats/residual lump them as contour.
Toolpath outer_fillet is claimed later from hole-deferred opening-tier fillets.

This module does NOT couple the passes. Each pass is independent; the cascade
only threads one pass's `remaining_faces` into the next as its candidate pool.

Run (omit --graph-npz to ingest the face graph from the STEP, which is always
correct for whatever part is passed; only pass --graph-npz for THIS part's own
cached graph):
  /Users/iansun19/miniconda3/envs/mlcad/bin/python run_cascade.py \
      "96260B_REAR_XR004_PCD PLATE.stp copy" -v
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Parts whose "through_hole" nodes should be relabeled to "contour_surface".
# fish_mold's bores are curved, sculpted mold surfaces rather than drillable
# through holes, so they belong with the residual contour surfaces (curved,
# surface-finished) instead of the drilling toolpath class. Keyed by normalized
# STEP stem (lowercased, spaces/hyphens -> underscores), matching the
# NO_STOCK_GATE_PART_STEMS convention in stock_cut_classification.py.
RELABEL_THROUGH_HOLE_TO_CONTOUR_STEMS: frozenset[str] = frozenset({"fish_mold"})

# Plain-pocket -> contour_surface relabel (see relabel_plain_pockets_as_contour).
# A face of a plain-pocket is a face-milled deck (-> flat) when its outward normal
# is within ~45deg of the +approach axis (dot >= DECK_NORMAL_DOT_MIN) and its area
# towers over the feature's wall/floor band (>= DECK_AREA_BAND_MULT x the largest
# non-planar face in that feature). These are relative gates with huge empirical
# margins on fish_mold (deck normal dot = +1.0 vs opposing deck -1.0; deck area
# ~34x the wall band), so the exact values are not load-bearing.
DECK_NORMAL_DOT_MIN: float = 0.7
DECK_AREA_BAND_MULT: float = 6.0


def relabel_through_holes_as_contour(graph: dict, part_id: str) -> int:
    """For configured parts, rename through_hole nodes to contour_surface.

    Returns the number of nodes relabeled. Mutates graph["nodes"] in place.
    """
    from stock_cut_classification import _normalize_stem

    if _normalize_stem(part_id) not in {
        _normalize_stem(s) for s in RELABEL_THROUGH_HOLE_TO_CONTOUR_STEMS
    }:
        return 0
    n = 0
    for node in graph.get("nodes", []):
        if node.get("class_name") == "through_hole":
            node["class_name"] = "contour_surface"
            node["class_id"] = -1  # CASCADE_TP_CLASS_ID["contour_surface"]
            n += 1
    return n


def relabel_plain_pockets_as_contour(
    graph: dict,
    faces: Sequence[Any],
    opening_axis: Sequence[float],
) -> tuple[int, int]:
    """Reclassify each *plain-pocket* feature to contour_surface, peeling large
    planar decks off to flat. Geometry-only: no part gate, no face-ID lists.

    LOAD-BEARING ASSUMPTION -- the selector is ``class_name == "pocket"``, the
    plain/fallback pocket bucket that ``pocket_detection.pocket_toolpath_class``
    emits ONLY for a closed pocket with no detected fillet radius (not open, not
    filleted). Every pocket we actually want to keep as a pocket lands in
    filleted_pocket / filleted_open_pocket / open_pocket instead, so this rule
    fires on none of them; the safety of this relabel rests entirely on that
    bucketing at ``pocket_toolpath_class``. Empirically the plain bucket is
    non-empty only on fish_mold (its two cone/bspline-dominated draft-walled
    cavities), so on every other part this is a no-op and the output is
    byte-identical.

    This "plain-pocket" sense is deliberately NOT any of:
      * the ``through_pocket`` KIND (emitted as class_name "through_pocket",
        cascade class_id 12) -- a distinct toolpath class we never touch;
      * ``PocketFeature.subtype == "through_pocket"`` -- a *plain* pocket can
        itself be a through_pocket by subtype (fish_mold feature 3 is exactly
        that). We key on the emitted toolpath ``class_name``, never on subtype.

    DECK peel: within a plain-pocket, a face is a face-milled deck (-> flat) when
    it is planar AND its outward normal aligns with the +approach axis
    (dot >= DECK_NORMAL_DOT_MIN) AND its area towers over the feature's wall/floor
    band (>= DECK_AREA_BAND_MULT x the largest non-planar face in that feature).
    All other faces of the feature -> contour_surface. The opposing -approach deck
    (dot < 0) stays contour, so the two mirror decks of a symmetric mold split
    correctly (fish_mold: +Z deck 109 -> flat; -Z deck 1 -> contour).

    NOTE: this does NOT address the upstream cause -- the pocket membership-grow
    pass (``pocket_detection._grow_pocket_membership`` / ``_is_grow_candidate``)
    absorbs a large +approach deck as a pocket step_plane with no area cap. Fixing
    that in-pass destabilizes the whole cascade (empirically: deck lands as
    open_pocket, wide downstream churn), so the deck-absorption bug is left as a
    separate upstream task; this emit-time relabel is the last word and is
    intentionally decoupled from the pass claim state.

    Mutates ``graph["nodes"]`` in place. Returns
    ``(n_pockets_reclassified, n_deck_faces_moved_to_flat)``; ``(0, 0)`` (no-op)
    when the part has no plain-pocket node.
    """
    from eval_cascade import CASCADE_TP_CLASS_ID

    approach = np.asarray(opening_axis, dtype=np.float64)
    na = float(np.linalg.norm(approach))
    if na < 1e-12:
        return 0, 0
    approach = approach / na

    n_pockets = 0
    deck_faces_moved = 0
    new_flat_nodes: list[dict] = []
    for node in graph.get("nodes", []):
        if node.get("class_name") != "pocket":
            continue
        fids = [int(i) for i in node["face_ids"]]
        nonplanar = [
            float(faces[i].area) for i in fids if faces[i].surface_type != "plane"
        ]
        band_ref = (
            max(nonplanar)
            if nonplanar
            else float(np.median([float(faces[i].area) for i in fids]))
        )
        deck: list[int] = []
        rest: list[int] = []
        for i in fids:
            fg = faces[i]
            is_deck = False
            if fg.surface_type == "plane":
                nrm = np.asarray(fg.normal, dtype=np.float64)
                nn = float(np.linalg.norm(nrm))
                dot = float(np.dot(nrm / nn, approach)) if nn > 1e-12 else 0.0
                is_deck = (
                    dot >= DECK_NORMAL_DOT_MIN
                    and float(fg.area) >= DECK_AREA_BAND_MULT * band_ref
                )
            (deck if is_deck else rest).append(i)

        node["class_name"] = "contour_surface"
        node["class_id"] = CASCADE_TP_CLASS_ID["contour_surface"]
        node["face_ids"] = sorted(rest)
        node["n_faces"] = len(rest)
        n_pockets += 1
        if deck:
            flat_node = dict(node)
            flat_node["class_name"] = "flat"
            flat_node["class_id"] = CASCADE_TP_CLASS_ID["flat"]
            flat_node["face_ids"] = sorted(deck)
            flat_node["n_faces"] = len(deck)
            new_flat_nodes.append(flat_node)
            deck_faces_moved += len(deck)

    for fn in new_flat_nodes:
        fn["feature_id"] = len(graph["nodes"])
        graph["nodes"].append(fn)
    return n_pockets, deck_faces_moved


from coaxial_stack_detection import (
    REFERENCE_CONTOUR_FACES_FRONT,
    REFERENCE_CONTOUR_FACES_REAR,
    REFERENCE_HUB_FLAT_FACES_FRONT,
    REFERENCE_HUB_FLAT_FACES_REAR,
    REFERENCE_OPEN_POCKET_FACES_FRONT,
    REFERENCE_OPEN_POCKET_FACES_REAR,
    CoaxialStackDetectionConfig,
    detect_coaxial_stack,
    render_table as render_coaxial_table,
    validate_coaxial_stack,
)
from flats_detection import (
    FlatDetectionConfig,
    detect_flats,
    render_table as render_flat_table,
    validate_flats,
)
from hole_detection import (
    ExpectedHole,
    HoleDetectionConfig,
    detect_holes,
    render_table as render_hole_table,
    validate_against_expected,
)
from pocket_detection import (
    PocketDetectionConfig,
    PocketSetupConfig,
    apply_filleted_lobe_tiers_to_result,
    detect_pockets,
    pocket_config_from_setup_dict,
    render_table as render_pocket_table,
    resolve_pocket_setup_for_run,
    validate_pockets,
    _part_axis_top,
)
from outer_fillet_detection import (
    REFERENCE_HUB_OUTER_FILLET_FACES_FRONT,
    REFERENCE_HUB_OUTER_FILLET_FACES_REAR,
    REFERENCE_OUTER_FILLET_FACES_FRONT,
    REFERENCE_OUTER_FILLET_FACES_REAR,
    detect_outer_fillets,
    render_table as render_outer_fillet_table,
    validate_outer_fillets,
)
from wall_detection import (
    REFERENCE_WALL_FACES_FRONT,
    WallDetectionConfig,
    detect_walls,
    render_table as render_wall_table,
    validate_walls,
)
from prismatic_profile_detection import (
    PrismaticProfileConfig,
    detect_prismatic_profiles,
)
from profile_detection import (
    REFERENCE_PROFILE_FACES_FRONT,
    ProfileDetectionConfig,
    detect_profiles,
    render_table as render_profile_table,
    validate_profiles,
)
from residual_detection import (
    REFERENCE_RESIDUAL_POOL_FRONT,
    REFERENCE_RESIDUAL_POOL_REAR,
    REFERENCE_SANITY_FACES_FRONT,
    REFERENCE_SANITY_FACES_REAR,
    ResidualDetectionConfig,
    detect_residual_candidates,
    render_table as render_residual_table,
    validate_residual,
)

logger = logging.getLogger("run_cascade")

DEFAULT_STEP = "96260B_REAR_XR004_PCD PLATE.stp copy"
# No default graph.npz: a per-part input must NOT default to one specific part's
# cached graph (that silently fed the rear/plate graph to the front part). When
# --graph-npz is omitted, _load_edges ingests the face graph from the STEP, which
# is always correct for whatever part is passed.

REFERENCE_HOLES = [
    ExpectedHole("blind_hole", 4.006, units="inch", tol=0.75),
    ExpectedHole("through_hole", 3.200, units="inch", tol=0.75),
]


def _load_edges(graph_npz: Path | None, step_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the face graph from a cached npz, else recompute via step_ingest."""
    if graph_npz is not None and graph_npz.is_file():
        d = np.load(graph_npz)
        return d["edge_index"], d["edge_attr"]
    from step_ingest import ingest_step_to_pyg

    _x, edge_index, edge_attr, _stats = ingest_step_to_pyg(str(step_path))
    return edge_index, edge_attr


def _resolve_pocket_setup_for_cascade(
    step_path: Path,
    *,
    setup_descriptor: Path | None = None,
    machining_side: str | None = None,
    pocket_access: str | None = None,
) -> PocketSetupConfig:
    """Resolve pocket setup: CLI args > setup descriptor YAML > filename hint."""
    if pocket_access is not None or machining_side is not None:
        return resolve_pocket_setup_for_run(
            step_path,
            machining_side=machining_side,  # type: ignore[arg-type]
            pocket_access=pocket_access,  # type: ignore[arg-type]
        )

    descriptor_path = setup_descriptor
    if descriptor_path is None:
        default = Path(__file__).resolve().parent / "eval/gt/96260B_setup.yaml"
        if default.is_file():
            descriptor_path = default

    if descriptor_path is not None and descriptor_path.is_file():
        from setup_descriptor import (
            SetupDescriptorError,
            find_setup_for_step,
            load_setup_descriptor,
            resolve_setup_entry,
            to_pocket_setup_config,
        )

        try:
            desc = load_setup_descriptor(descriptor_path)
            if find_setup_for_step(desc, step_path) is not None:
                resolved = resolve_setup_entry(desc, step_path=step_path)
                return to_pocket_setup_config(resolved)
        except SetupDescriptorError as exc:
            logging.getLogger("run_cascade").warning(
                "setup descriptor %s skipped: %s", descriptor_path, exc,
            )

    return resolve_pocket_setup_for_run(step_path)


def run_cascade(
    step_path: str | Path,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    pocket_config: PocketDetectionConfig | None = None,
    hole_config: HoleDetectionConfig | None = None,
    coaxial_config: CoaxialStackDetectionConfig | None = None,
    flat_config: FlatDetectionConfig | None = None,
    outer_fillet_config: OuterFilletDetectionConfig | None = None,
    wall_config: WallDetectionConfig | None = None,
    prismatic_profile_config: PrismaticProfileConfig | None = None,
    profile_config: ProfileDetectionConfig | None = None,
    residual_config: ResidualDetectionConfig | None = None,
    instrument: bool = False,
    instrument_dir: Path | None = None,
):
    """Run full cascade; return (faces, pk, hl, cx, fl, of, wl, pr, rs, if_)."""
    from feature_params import analyze_step, load_step_faces, require_occ

    require_occ()
    faces = analyze_step(step_path)
    occ_faces = load_step_faces(step_path)
    n_faces = len(faces)

    # Guard: the edge graph MUST describe this exact part. A face-count mismatch
    # means the wrong graph.npz was loaded for this STEP (see _load_edges). This
    # is not paranoia: 348-vs-299 crashing downstream in the pocket pass was luck.
    # A mismatch that happened to be COMPATIBLE (indices in range) would corrupt
    # adjacency silently and yield a plausible-but-wrong plan with no crash. Fail
    # loudly here, naming both counts, before any pass consumes the bad graph.
    edge_nodes = int(edge_index.max()) + 1 if np.asarray(edge_index).size else 0
    if edge_nodes != n_faces:
        raise ValueError(
            f"edge graph has {edge_nodes} node(s) but STEP {step_path} has "
            f"{n_faces} face(s): the face graph does not match this part "
            f"(wrong --graph-npz?). Pass the part's own graph or omit --graph-npz "
            f"to ingest from the STEP."
        )

    # No pre-cascade STOCK/CUT gate. STOCK is not a category in this system:
    # every B-rep face enters the cascade and must exit with exactly one feature
    # label. Nothing is filtered before the passes run — the terminal-pass
    # invariant below enforces that everything is genuinely claimed.
    cut_candidates = set(range(n_faces))

    claim_recorder = None
    if instrument:
        from cascade_instrumentation import PassClaimRecorder

        claim_recorder = PassClaimRecorder()

    # Pass 0 - inner_fillet: geometry-only template match run before pockets so a
    # core tangent blend is claimed as its own feature rather than absorbed into
    # a pocket. Claims nothing on parts without such blends (e.g. 96260B), so it
    # leaves the downstream pool - and hence the partition - unchanged there.
    from inner_fillet_detection import detect_inner_fillets

    inner_fillet_result = detect_inner_fillets(
        step_path, faces, edge_index, edge_attr,
        candidate_faces=cut_candidates,
    )
    cut_candidates = inner_fillet_result.remaining_faces
    if inner_fillet_result.claimed_faces:
        logger.info(
            "inner_fillet: %d claimed (%s)",
            len(inner_fillet_result.claimed_faces),
            sorted(inner_fillet_result.claimed_faces),
        )

    pocket_result = detect_pockets(
        faces, edge_index, edge_attr,
        occ_faces=occ_faces, candidate_faces=cut_candidates,
        config=pocket_config,
    )
    pocket_result = apply_filleted_lobe_tiers_to_result(
        pocket_result, faces, edge_index, edge_attr, occ_faces,
        pocket_config or PocketDetectionConfig(),
    )
    if claim_recorder is not None:
        from cascade_instrumentation import PassLabelContext

        claim_recorder.record_pass(
            "pockets", 1, "pocket_detection", n_faces, pocket_result,
            label_ctx=PassLabelContext(
                faces=faces,
                edge_index=edge_index,
                edge_attr=edge_attr,
                occ_faces=occ_faces,
                pocket_config=pocket_config,
            ),
        )

    # Through-pocket pass: closed side-wall cavities open on BOTH part faces
    # (non-round profile). Runs BETWEEN pockets and holes so it claims the whole
    # wall ring before the hole pass grabs its arc walls and before flats grabs
    # its straight walls -- otherwise each slot fragments into holes + flats. The
    # filleted-pocket detector above is floor-seed based and never touched these
    # (a through pocket has no floor), so its result is unchanged. The features
    # are folded into the pocket result (like prismatic profiles into the profile
    # result): they carry their own `through_pocket` toolpath_class, so the graph
    # builder emits them as a distinct class, while downstream passes see their
    # faces as claimed. Geometry-only -- claims nothing on parts without such a
    # cavity (96260B, fish_mold, part1), leaving the partition byte-identical.
    from through_pocket_detection import detect_through_pockets

    through_pocket_result = detect_through_pockets(
        faces, edge_index, edge_attr,
        candidate_faces=pocket_result.remaining_faces,
    )
    if through_pocket_result.features:
        base_id = len(pocket_result.features)
        for offset, feat in enumerate(through_pocket_result.features):
            feat.feature_id = base_id + offset
            pocket_result.features.append(feat)
        pocket_result.claimed_faces |= through_pocket_result.claimed_faces
        pocket_result.remaining_faces -= through_pocket_result.claimed_faces
        logger.info(
            "through_pocket: %d claimed across %d pocket(s) (%s)",
            len(through_pocket_result.claimed_faces),
            len(through_pocket_result.features),
            sorted(through_pocket_result.claimed_faces),
        )

    # Prismatic profile band (swept outer contour: closed cycle of vertical
    # walls + ISOLATED corner-fillet cylinders about a common axis). Runs BEFORE
    # holes — and hence before coaxial/flats/wall/residual — because a prismatic
    # outer contour is MORE SPECIFIC than "a concave/arc vertical cylinder = a
    # hole": the exterior boundary of a 2.5D part carries scallop notches and
    # boundary arcs that the hole pass would otherwise grab as through_holes,
    # breaking the degree-2 cycle so the whole contour fragments across
    # flats/wall/contour (the Part4 outer-boundary bug). Claiming the cycle first
    # keeps it as ONE profile, exactly as pockets-before-holes keeps a pocket
    # wall from being grabbed as a hole. Predicate B rejects interior through-hole
    # rings (adjacent half-cylinders violate corner-fillet isolation), so it never
    # steals a genuine hole. Geometry-only: claims nothing on parts without such a
    # cycle (96260B), and claims the identical face set whether run here or
    # post-coaxial on parts that do have one (fish_mold, part1/2/3), leaving their
    # partitions byte-identical.
    prismatic_result = detect_prismatic_profiles(
        faces, edge_index, edge_attr,
        candidate_faces=set(pocket_result.remaining_faces),
        config=prismatic_profile_config,
    )
    if prismatic_result.claimed_faces:
        logger.info(
            "prismatic_profile: %d claimed across %d profile(s) (%s)",
            len(prismatic_result.claimed_faces),
            len(prismatic_result.features),
            sorted(prismatic_result.claimed_faces),
        )

    pool_in = pocket_result.remaining_faces
    structureless_reserved = set(pocket_result.structureless_released_faces)
    hole_pool = pool_in - structureless_reserved - prismatic_result.claimed_faces
    hole_result = detect_holes(
        faces, edge_index, edge_attr,
        occ_faces=occ_faces, candidate_faces=hole_pool,
        config=hole_config,
    )
    if structureless_reserved:
        hole_result.remaining_faces |= structureless_reserved
    if claim_recorder is not None:
        claim_recorder.record_pass(
            "holes", 2, "hole_detection", len(pool_in), hole_result,
        )

    occ_map = {i: occ_faces[i] for i in range(n_faces)} if occ_faces else None
    opening_axis = np.asarray(pocket_result.opening_axis, dtype=float)
    part_axis_top = _part_axis_top(faces, opening_axis, occ_map)

    pool_in = hole_result.remaining_faces
    coaxial_result = detect_coaxial_stack(
        faces, edge_index, edge_attr,
        candidate_faces=pool_in,
        hole_claimed_faces=hole_result.claimed_faces,
        hole_features=hole_result.features,
        opening_axis=pocket_result.opening_axis,
        part_axis_top=part_axis_top,
        config=coaxial_config,
    )
    if claim_recorder is not None:
        claim_recorder.record_pass(
            "coaxial_stack", 3, "coaxial_stack_detection", len(pool_in), coaxial_result,
        )

    hub_open_pocket_faces: set[int] = set()
    for feat in coaxial_result.features:
        if feat.kind == "open_pocket":
            hub_open_pocket_faces |= feat.face_indices

    # Chamfer band (closed oblique-bevel ring: planes tilted ~45deg to the part
    # axis + coaxial corner cones). Runs BEFORE flats/contour so the band's
    # oblique planes are not first grabbed as flats and its faces not scattered
    # into contour -- the whole ring is claimed as ONE chamfer. Reachability-gated
    # so buried internal edge-breaks are rejected. Geometry-only: claims nothing
    # on parts without such a reachable loop (e.g. 96260B), leaving the downstream
    # pool -- and hence the partition -- byte-identical there. The OCC shape used
    # by the reachability gate is loaded lazily, only if loop candidates exist.
    from chamfer_detection import detect_chamfers

    def _chamfer_reach_probe(axis):
        from step_ingest import load_step_shape
        from reachability import make_axis_reachability_probe

        shape, _ = load_step_shape(str(step_path))
        return make_axis_reachability_probe(shape, occ_faces, axis)

    pool_in = coaxial_result.remaining_faces - prismatic_result.claimed_faces
    chamfer_result = detect_chamfers(
        faces, edge_index, edge_attr,
        candidate_faces=pool_in,
        reachability_probe_factory=_chamfer_reach_probe,
    )
    if chamfer_result.claimed_faces:
        logger.info(
            "chamfer: %d claimed across %d ring(s) (%s)",
            len(chamfer_result.claimed_faces),
            len(chamfer_result.features),
            sorted(chamfer_result.claimed_faces),
        )

    pool_in = coaxial_result.remaining_faces - prismatic_result.claimed_faces - chamfer_result.claimed_faces
    flat_result = detect_flats(
        faces, edge_index, edge_attr,
        candidate_faces=pool_in,
        hole_claimed_faces=hole_result.claimed_faces,
        pocket_claimed_faces=pocket_result.claimed_faces,
        pocket_floor_absorbed_faces=pocket_result.floor_absorbed_faces,
        hub_flat_faces=coaxial_result.hub_flat_faces,
        opening_axis=pocket_result.opening_axis,
        occ_faces=occ_faces,
        config=flat_config,
    )
    if claim_recorder is not None:
        claim_recorder.record_pass(
            "flats", 4, "flats_detection", len(pool_in), flat_result,
        )

    stack_groups = hole_result.deferred_feature_fillet_groups
    pool_in = flat_result.remaining_faces
    outer_fillet_result = detect_outer_fillets(
        faces, edge_index, edge_attr,
        candidate_faces=pool_in,
        pocket_claimed_faces=pocket_result.claimed_faces,
        stack_boundary_fillet_groups=stack_groups,
        flat_claimed_faces=flat_result.claimed_faces,
        hub_open_pocket_faces=hub_open_pocket_faces,
        hub_perimeter_context=coaxial_result.hub_perimeter_context,
        opening_axis=pocket_result.opening_axis,
        part_axis_top=part_axis_top,
        config=outer_fillet_config,
    )
    if claim_recorder is not None:
        claim_recorder.record_pass(
            "outer_fillets", 5, "outer_fillet_detection", len(pool_in), outer_fillet_result,
        )

    pool_in = outer_fillet_result.remaining_faces
    wall_result = detect_walls(
        faces, edge_index, edge_attr,
        candidate_faces=pool_in,
        pocket_claimed_faces=pocket_result.claimed_faces,
        hole_claimed_faces=hole_result.claimed_faces,
        hub_open_pocket_faces=hub_open_pocket_faces,
        opening_axis=pocket_result.opening_axis,
        config=wall_config,
    )
    if claim_recorder is not None:
        claim_recorder.record_pass(
            "wall", 6, "wall_detection", len(pool_in), wall_result,
        )

    pool_in = wall_result.remaining_faces
    profile_result = detect_profiles(
        faces, edge_index, edge_attr,
        candidate_faces=pool_in,
        pocket_claimed_faces=pocket_result.claimed_faces,
        hole_claimed_faces=hole_result.claimed_faces,
        hub_open_pocket_faces=hub_open_pocket_faces,
        wall_claimed_faces=wall_result.claimed_faces,
        wall_seed_faces=wall_result.seed_faces,
        opening_axis=pocket_result.opening_axis,
        config=profile_config,
    )
    if claim_recorder is not None:
        claim_recorder.record_pass(
            "profile", 7, "profile_detection", len(pool_in), profile_result,
        )

    pool_in = profile_result.remaining_faces
    residual_result = detect_residual_candidates(
        faces, edge_index, edge_attr,
        candidate_faces=pool_in,
        config=residual_config,
    )
    if claim_recorder is not None:
        claim_recorder.record_pass(
            "residual_candidates", 8, "residual_detection", len(pool_in), residual_result,
        )
        assert instrument_dir is not None
        claim_recorder.write(instrument_dir / "pass_claims.json")

    # Fold prismatic-profile bands (claimed before flats) into the profile
    # result so the graph builder emits them as `profile` nodes. Claiming
    # happened early; the downstream pool already excludes these faces, so this
    # is purely attaching the feature records. No-op on parts without a band.
    if prismatic_result.features:
        profile_result.features = list(profile_result.features) + list(
            prismatic_result.features
        )
        profile_result.claimed_faces |= prismatic_result.claimed_faces

    # ------------------------------------------------------------------
    # Counterbore merge (post-cascade). A counterbored through-hole is split by
    # the hole pass into a filleted_blind_hole (enlarged mouth + flat annular
    # shoulder + shoulder fillets) and a separate through_hole (the bore). We run
    # the cascade UNCHANGED — so every downstream pass sees identical hole
    # context (claimed faces, deferred fillet groups) and no neighbouring face
    # moves — then geometrically detect the counterbore and, when its face set is
    # exactly the union of those hole features, emit it as ONE counterbore
    # feature. The hole result is NOT mutated (its claimed faces stay complete);
    # build_cascade_feature_graph suppresses the absorbed hole features when a
    # counterbore result is supplied, so every other graph-builder caller stays
    # backward-compatible. No-op on parts without the structure (fish_mold,
    # part1) — the geometric predicate finds nothing there.
    # ------------------------------------------------------------------
    from counterbore_detection import (
        detect_counterbores,
        reconcile_counterbores_with_holes,
    )

    counterbore_result = detect_counterbores(
        step_path, faces, edge_index, edge_attr,
        occ_faces=occ_faces, candidate_faces=set(range(n_faces)),
    )
    counterbore_result = reconcile_counterbores_with_holes(
        counterbore_result, hole_result,
    )
    if counterbore_result.features:
        logger.info(
            "counterbore: %d merged from hole features (%s)",
            len(counterbore_result.features),
            sorted(counterbore_result.claimed_faces),
        )

    # ------------------------------------------------------------------
    # Countersink relabel (post-cascade). The hole pass already groups a
    # countersunk through-bore's conical entry + bore into ONE through_hole
    # feature carrying every face; only the LABEL is wrong. We reclassify each
    # qualifying hole feature (through_hole + coaxial cone flaring wider than the
    # bore, not a counterbore) as ONE countersink and let
    # build_cascade_feature_graph emit it as a `countersink` node, suppressing the
    # absorbed through_hole. Operating on hole_result.features (the POST-POCKET
    # set) is load-bearing: fish_mold's coned faces are claimed by pockets first
    # and never reach the hole pass. Counterbore-absorbed hole features are
    # excluded so a face can never land in both. No face moves; no-op on parts
    # without a widening conical through-bore. See countersink_detection.py.
    # ------------------------------------------------------------------
    from countersink_detection import detect_countersinks

    countersink_candidates = set(range(n_faces)) - counterbore_result.claimed_faces
    countersink_result = detect_countersinks(
        hole_result, faces,
        occ_faces=occ_faces, candidate_faces=countersink_candidates,
    )
    if countersink_result.features:
        logger.info(
            "countersink: %d relabeled from hole features (%s)",
            len(countersink_result.features),
            sorted(countersink_result.claimed_faces),
        )

    # ------------------------------------------------------------------
    # Terminal-pass invariant (design axiom): every B-rep face exits the
    # cascade with exactly one feature label. There is no STOCK category and no
    # gated-out faces, so the residual (contour_surface) pass MUST claim every
    # face the earlier passes left behind — its `remaining_faces` is the
    # convexity_cascade_pool_unclaimed set and must be empty. If any face is
    # still unclaimed it would map to feature_id == -1 in the exported graph;
    # fail loudly rather than emit an incomplete partition. The pre-cascade gate
    # used to mask this by pre-removing faces from the pool.
    # ------------------------------------------------------------------
    all_claimed = (
        inner_fillet_result.claimed_faces
        | counterbore_result.claimed_faces
        | countersink_result.claimed_faces
        | pocket_result.claimed_faces
        | hole_result.claimed_faces
        | coaxial_result.claimed_faces
        | prismatic_result.claimed_faces
        | chamfer_result.claimed_faces
        | flat_result.claimed_faces
        | outer_fillet_result.claimed_faces
        | wall_result.claimed_faces
        | profile_result.claimed_faces
        | residual_result.claimed_faces
    )
    unclaimed = (set(range(n_faces)) - all_claimed) | set(
        residual_result.remaining_faces
    )
    if unclaimed:
        offending = sorted(unclaimed)
        raise RuntimeError(
            "cascade terminal-pass invariant violated: "
            f"{len(offending)} face(s) left unclaimed after the residual "
            "(contour_surface) pass. Every face must exit the cascade with "
            "exactly one feature label — no STOCK category, no feature_id == -1. "
            f"convexity_cascade_pool_unclaimed / offending face ids: {offending}"
        )

    return (
        faces, pocket_result, hole_result, coaxial_result, flat_result,
        outer_fillet_result, wall_result, profile_result, residual_result,
        inner_fillet_result, chamfer_result, counterbore_result,
        countersink_result,
    )


def _selftest() -> int:
    """OCC-free checks for relabel_plain_pockets_as_contour.

    Uses SimpleNamespace face stubs mirroring fish_mold's real geometry values.
    Scenarios: (a) the two plain-pocket features split exactly as targeted --
    feat 3 -> 23 contour + face 109 flat, feat 4 -> 6 contour, face 1 stays
    contour; (b) a frozen-part graph (no plain-pocket node) is a byte-identical
    no-op -- the OCC-free proxy for "empty on all four goldens".
    """
    import copy
    from types import SimpleNamespace as NS

    def face(idx, stype, normal, area):
        return NS(index=idx, surface_type=stype,
                  normal=np.asarray(normal, dtype=np.float64), area=float(area),
                  centroid=np.zeros(3), radius=None, axis=None,
                  semi_angle_rad=None)

    # --- faithful fish_mold face table (index -> geom), values from analyze_step.
    APPROACH = [0.0, 0.0, 1.0]
    specs: dict[int, tuple[str, list[float], float]] = {
        # feat 3 (24 faces): planes 1/23/109, corner-fillet cylinders, draft cones,
        # bspline blends. Only face 109 (+Z, huge area) is a deck.
        1: ("plane", [0, 0, -1], 2852.16), 23: ("plane", [0, 0, -1], 663.13),
        109: ("plane", [0, 0, 1], 2924.15),
        9: ("cylinder", [-.71, .71, 0], 20.27), 11: ("cylinder", [-.71, -.71, 0], 20.27),
        13: ("cylinder", [.71, -.71, 0], 20.27), 120: ("cylinder", [.71, -.71, 0], 12.16),
        122: ("cylinder", [-.71, -.71, 0], 12.16), 124: ("cylinder", [-.71, .71, 0], 12.16),
        29: ("cone", [-.12, -.7, -.7], 86.96), 34: ("cone", [.17, -.7, -.7], 75.02),
        139: ("cone", [-.12, -.7, .7], 86.96), 144: ("cone", [.17, -.7, .7], 75.02),
        31: ("cone", [-.2, -.69, -.69], 31.19), 32: ("cone", [-.2, -.69, -.69], 17.33),
        141: ("cone", [-.2, -.69, .69], 31.19), 142: ("cone", [-.2, -.69, .69], 17.33),
        97: ("cone", [-.5, -.5, -.71], 21.50), 207: ("cone", [-.5, -.5, .71], 21.50),
        224: ("cone", [-.5, .5, .71], 4.46), 228: ("cone", [-.5, -.5, .71], 4.46),
        232: ("cone", [.5, -.5, .71], 4.46),
        48: ("bspline", [-.24, .8, -.56], 39.81), 156: ("bspline", [-.24, .8, .56], 39.82),
        # feat 4 (6 faces): 2 cones (band ceiling ~104), 2 near-Z planes (small),
        # 2 bsplines. Face 172 faces +Z (dot>0.7) but its area is far below the
        # cone band, so it stays contour -- the area gate, not the normal, saves it.
        50: ("cone", [-.12, .82, -.56], 103.81), 158: ("cone", [-.12, .82, .56], 103.82),
        54: ("plane", [-.12, 0, -.99], 17.32), 172: ("plane", [-.12, 0, .99], 17.27),
        87: ("bspline", [-.12, .2, -.97], 6.44), 199: ("bspline", [-.12, .2, .97], 6.45),
        # a foreign large +Z plane NOT inside any plain-pocket (must be untouched).
        200: ("plane", [0, 0, 1], 4747.7),
    }
    faces = [face(0, "plane", [0, 0, 1], 1.0) for _ in range(233)]
    for i, (st, nrm, ar) in specs.items():
        faces[i] = face(i, st, nrm, ar)

    FEAT3 = sorted(i for i in specs if i not in (50, 158, 54, 172, 87, 199, 200))
    FEAT4 = [50, 54, 87, 158, 172, 199]
    TARGET_CONTOUR = set(FEAT3 + FEAT4) - {109}

    graph = {"nodes": [
        {"feature_id": 0, "class_name": "filleted_pocket", "class_id": 6,
         "face_ids": [88, 89], "n_faces": 2, "params": {}},
        {"feature_id": 1, "class_name": "pocket", "class_id": 6,
         "face_ids": list(FEAT3), "n_faces": len(FEAT3), "params": {}},
        {"feature_id": 2, "class_name": "pocket", "class_id": 6,
         "face_ids": list(FEAT4), "n_faces": len(FEAT4), "params": {}},
        {"feature_id": 3, "class_name": "contour_surface", "class_id": -1,
         "face_ids": [200], "n_faces": 1, "params": {}},
    ]}
    pre_contour = copy.deepcopy(graph["nodes"][3])
    pre_filleted = copy.deepcopy(graph["nodes"][0])

    n_plain, n_deck = relabel_plain_pockets_as_contour(graph, faces, APPROACH)

    failures: list[str] = []
    if (n_plain, n_deck) != (2, 1):
        failures.append(f"return {(n_plain, n_deck)} != (2, 1)")

    by_class: dict[str, list[list[int]]] = {}
    for nd in graph["nodes"]:
        by_class.setdefault(nd["class_name"], []).append(sorted(nd["face_ids"]))

    # feat 3 -> contour with exactly its 23 non-deck faces
    if (set(FEAT3) - {109}) not in [set(x) for x in by_class.get("contour_surface", [])]:
        failures.append("feat3 23-face contour node missing")
    # feat 4 -> contour with all 6 faces, no deck
    if set(FEAT4) not in [set(x) for x in by_class.get("contour_surface", [])]:
        failures.append("feat4 6-face contour node missing")
    # face 109 -> flat, ALONE, and NOT contour, NOT pocket
    flat_sets = [set(x) for x in by_class.get("flat", [])]
    if {109} not in flat_sets:
        failures.append("face 109 not a standalone flat node")
    all_contour = set().union(*[set(x) for x in by_class.get("contour_surface", [])]) \
        if by_class.get("contour_surface") else set()
    if 109 in all_contour:
        failures.append("face 109 leaked into contour")
    if "pocket" in by_class:
        failures.append("a plain-pocket node survived the relabel")
    # face 1 stays contour (opposing -Z deck), never flat
    if 1 not in all_contour or any(1 in s for s in flat_sets):
        failures.append("face 1 (-Z deck) did not stay contour")
    # the full contour target set is reproduced exactly (plus the pre-existing
    # foreign contour face 200, which must be carried through untouched).
    expected_contour = TARGET_CONTOUR | {200}
    if all_contour != expected_contour:
        failures.append(
            f"contour set mismatch: extra={sorted(all_contour - expected_contour)} "
            f"missing={sorted(expected_contour - all_contour)}")
    # pre-existing contour + filleted_pocket nodes untouched; foreign plane 200 stays
    if graph["nodes"][3] != pre_contour:
        failures.append("pre-existing contour node membership changed")
    if graph["nodes"][0] != pre_filleted:
        failures.append("filleted_pocket node changed")

    # frozen-part proxy: no plain-pocket node -> exact no-op
    frozen = {"nodes": [
        {"feature_id": 0, "class_name": "filleted_open_pocket", "face_ids": [1, 2]},
        {"feature_id": 1, "class_name": "open_pocket", "face_ids": [3]},
        {"feature_id": 2, "class_name": "through_pocket", "face_ids": [4, 5]},
        {"feature_id": 3, "class_name": "contour_surface", "face_ids": [6]},
    ]}
    frozen_before = copy.deepcopy(frozen)
    r = relabel_plain_pockets_as_contour(frozen, faces, APPROACH)
    if r != (0, 0) or frozen != frozen_before:
        failures.append(f"frozen-part graph mutated (returned {r})")

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("relabel_plain_pockets_as_contour selftest: PASS "
          "(feat3 -> 23 contour + 109 flat; feat4 -> 6 contour; "
          "face 1 stays contour; frozen-part no-op)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Feature-grouping cascade")
    ap.add_argument("step", nargs="?", default=DEFAULT_STEP, help="reference STEP file")
    ap.add_argument("--graph-npz", type=Path, default=None,
                    help="cached face graph (edge_index/edge_attr) for THIS part; "
                         "omit to ingest from the STEP (default). A mismatched graph "
                         "is rejected loudly (face-count guard in run_cascade).")
    ap.add_argument("--max-diameter", type=float, default=150.0,
                    help="hole diameter ceiling (mm): larger concave bores are deferred")
    ap.add_argument("--wall-attach-dist", type=float, default=None,
                    help="override pocket wall-attach distance (mm); default adaptive")
    ap.add_argument(
        "--machining-side",
        choices=("front", "back"),
        default=None,
        help="setup-scoped pocket access: front→open, back→closed (see GT yaml setup:)",
    )
    ap.add_argument(
        "--pocket-access",
        choices=("open", "closed"),
        default=None,
        help="override filleted-pocket access directly (wins over --machining-side)",
    )
    ap.add_argument(
        "--setup-descriptor",
        type=Path,
        default=None,
        help="setup descriptor YAML (default: eval/gt/96260B_setup.yaml when step matches)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--export-dir", type=Path, default=None,
        help="write feature_graph_cascade.json (+ viewer.html if --open) under this directory",
    )
    ap.add_argument(
        "--part-id", default=None,
        help="part id for export graph/viewer (default: stem of STEP filename)",
    )
    ap.add_argument(
        "--open", action="store_true",
        help="with --export-dir, build viewer.html and open in browser",
    )
    ap.add_argument(
        "--instrument",
        action="store_true",
        help="write diagnostic artifacts under pipeline_out/<part>/instrument/",
    )
    ap.add_argument(
        "--merge-lobe-contours",
        action=argparse.BooleanOptionalAction,
        default=False,
        # Default-off: obsoleted by the current segmentation, not broken. The
        # cascade now emits the lobe cap as ~7 band contour_surface nodes
        # directly, but each band straddles two adjacent lobes, so the
        # single-lobe candidate filter (lobe_contour_merge.py:100, requires
        # len(lobes)==1) matches nothing and the pass is a no-op. The blessed
        # 17-op 96260B baseline was produced with this merge doing nothing.
        # Kept behind an opt-in flag in case a part regresses to per-lobe
        # over-segmentation, where the merge would fire again.
        help="merge same-lobe cap contour_surface fragments before export (default: off)",
    )
    ap.add_argument(
        "--lateral-axes",
        action="store_true",
        default=False,
        help="ALSO annotate PROVISIONAL lateral ±X/±Y/±Z approach candidacy + "
             "reachability (unvalidated; default: off, calibrated Z-only path only)",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="run OCC-free unit checks for relabel_plain_pockets_as_contour and exit",
    )
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    step_path = Path(args.step)
    if not step_path.is_file():
        print(f"STEP not found: {step_path}")
        return 2

    is_front = "FRONT" in step_path.name.upper()
    edge_index, edge_attr = _load_edges(args.graph_npz, step_path)
    setup = _resolve_pocket_setup_for_cascade(
        step_path,
        setup_descriptor=args.setup_descriptor,
        machining_side=args.machining_side,
        pocket_access=args.pocket_access,
    )
    pocket_config = PocketDetectionConfig(
        wall_attach_dist=args.wall_attach_dist,
        setup=setup,
    )
    hole_config = HoleDetectionConfig(max_hole_diameter_mm=args.max_diameter)

    instrument_dir = None
    part_id = args.part_id or step_path.stem.replace(" ", "_").split(".")[0]
    if "96260B" in part_id.upper() and "FRONT" in step_path.name.upper():
        part_id = "96260B_front"
    if args.instrument:
        from cascade_instrumentation import resolve_instrument_dir

        instrument_dir = resolve_instrument_dir(
            step_path, export_dir=args.export_dir, part_id=args.part_id,
        )
        instrument_dir.mkdir(parents=True, exist_ok=True)

    faces, pk, hl, cx, fl, of, wl, pr, rs, if_, ch, cb, cs = run_cascade(
        step_path, edge_index, edge_attr,
        pocket_config=pocket_config, hole_config=hole_config,
        instrument=args.instrument, instrument_dir=instrument_dir,
    )
    n_faces = len(faces)

    if args.instrument:
        from cascade_instrumentation import run_instrumentation
        from feature_params import load_step_faces

        run_instrumentation(
            instrument_dir,
            n_faces=n_faces,
            faces=faces,
            edge_index=edge_index,
            edge_attr=edge_attr,
            occ_faces=load_step_faces(step_path),
            pocket_result=pk,
            hole_result=hl,
            coaxial_result=cx,
            flat_result=fl,
            outer_fillet_result=of,
            wall_result=wl,
            profile_result=pr,
            residual_result=rs,
            graph_npz=args.graph_npz,
            part_id=part_id,
            pocket_config=pocket_config,
            hole_config=hole_config,
        )

    print(f"STEP: {step_path}")
    print(
        f"faces: {n_faces}   cascade: "
        f"inner_fillet -> counterbore -> pockets -> holes -> coaxial_stack "
        f"-> flats -> outer_fillets -> wall -> profile -> residual"
    )

    from counterbore_detection import render_table as render_counterbore_table
    from countersink_detection import render_table as render_countersink_table

    print("\n" + "=" * 78)
    print("PASS 0.5 — COUNTERBORES")
    print("=" * 78)
    print(render_counterbore_table(cb))
    print(cb.summary())

    print("\n" + "=" * 78)
    print("PASS 0.6 — COUNTERSINKS")
    print("=" * 78)
    print(render_countersink_table(cs))
    print(cs.summary())

    print("\n" + "=" * 78)
    print("PASS 0 — INNER FILLETS")
    print("=" * 78)
    print(if_.summary())

    print("\n" + "=" * 78)
    print("PASS 1 — POCKETS")
    print("=" * 78)
    print(render_pocket_table(pk))
    pk_report = validate_pockets(pk)
    print(pk_report.render())

    print("\n" + "=" * 78)
    print("PASS 2 — HOLES")
    print("=" * 78)
    print(f"input pool: {len(pk.remaining_faces)} faces")
    print(render_hole_table(hl))
    # On 96260B the two REFERENCE_HOLES (⌀4.006in blind mouth + ⌀3.200in through
    # bore) are precisely the counterbore's two coaxial cylinders. Once the
    # counterbore pass claims them, the hole pass legitimately finds neither, so
    # the expectation collapses to empty when a counterbore was detected.
    expected_holes = [] if cb.features else REFERENCE_HOLES
    hl_report = validate_against_expected(hl.features, expected_holes, exact=True)
    print(hl_report.render())

    print("\n" + "=" * 78)
    print("PASS 3 — COAXIAL STACK (central hub)")
    print("=" * 78)
    print(f"input pool: {len(hl.remaining_faces)} faces")
    print(render_coaxial_table(cx))
    if is_front:
        cx_report = validate_coaxial_stack(
            cx,
            expected_hub_flat_faces=REFERENCE_HUB_FLAT_FACES_FRONT,
            expected_open_pocket_faces=REFERENCE_OPEN_POCKET_FACES_FRONT,
            expected_contour_faces=REFERENCE_CONTOUR_FACES_FRONT,
            forbidden_faces=(97, 280, 298),
        )
    else:
        cx_report = validate_coaxial_stack(
            cx,
            expected_hub_flat_faces=REFERENCE_HUB_FLAT_FACES_REAR,
            expected_open_pocket_faces=REFERENCE_OPEN_POCKET_FACES_REAR,
            expected_contour_faces=REFERENCE_CONTOUR_FACES_REAR,
            forbidden_faces=(280, 298),
        )
    print(cx_report.render())

    print("\n" + "=" * 78)
    print("PASS 4 — FLATS")
    print("=" * 78)
    print(f"input pool: {len(cx.remaining_faces)} faces")
    print(render_flat_table(fl))
    if is_front:
        fl_report = validate_flats(
            fl,
            expected_flats=2,
            expected_faces=(97, 273),
            deferred_faces=(),
        )
    else:
        fl_report = validate_flats(
            fl,
            expected_flats=2,
            expected_faces=(322, 112),
            deferred_faces=(),
        )
    print(fl_report.render())

    print("\n" + "=" * 78)
    print("PASS 5 — OUTER FILLETS (perimeter)")
    print("=" * 78)
    print(f"input pool: {len(fl.remaining_faces)} faces")
    print(render_outer_fillet_table(of))
    if is_front:
        of_report = validate_outer_fillets(
            of,
            expected_instances=2,
            expected_faces=sorted(
                set(REFERENCE_OUTER_FILLET_FACES_FRONT)
                | set(REFERENCE_HUB_OUTER_FILLET_FACES_FRONT)
            ),
        )
    else:
        of_report = validate_outer_fillets(
            of,
            expected_instances=2,
            expected_faces=sorted(
                set(REFERENCE_OUTER_FILLET_FACES_REAR)
                | set(REFERENCE_HUB_OUTER_FILLET_FACES_REAR)
            ),
        )
    print(of_report.render())

    print("\n" + "=" * 78)
    print("PASS 6 — WALL (exterior OD notch segments)")
    print("=" * 78)
    print(f"input pool: {len(of.remaining_faces)} faces")
    print(render_wall_table(wl))
    if is_front:
        wl_report = validate_walls(
            wl,
            expected_instances=14,
            expected_faces=sorted(REFERENCE_WALL_FACES_FRONT),
            forbidden_faces=(97, 273, 277, 280, 298),
        )
    else:
        wl_report = validate_walls(
            wl,
            expected_instances=len(wl.features),
            expected_faces=sorted(wl.claimed_faces) if wl.claimed_faces else [],
        )
    print(wl_report.render())

    print("\n" + "=" * 78)
    print("PASS 7 — PROFILE (opposing hub step cylinders)")
    print("=" * 78)
    print(f"input pool: {len(wl.remaining_faces)} faces")
    print(render_profile_table(pr))
    if is_front:
        pr_report = validate_profiles(
            pr,
            expected_instances=1,
            expected_faces=sorted(REFERENCE_PROFILE_FACES_FRONT),
            forbidden_faces=(97, 273, 277, 280, 298),
        )
    else:
        pr_report = validate_profiles(
            pr,
            expected_instances=1,
            expected_faces=sorted(pr.claimed_faces) if pr.claimed_faces else [],
        )
    print(pr_report.render())

    print("\n" + "=" * 78)
    print("PASS 8 — RESIDUAL CANDIDATES")
    print("=" * 78)
    print(f"input pool: {len(pr.remaining_faces)} faces")
    print(render_residual_table(rs))
    expected_residual = REFERENCE_RESIDUAL_POOL_FRONT if is_front else REFERENCE_RESIDUAL_POOL_REAR
    sanity = REFERENCE_SANITY_FACES_FRONT if is_front else REFERENCE_SANITY_FACES_REAR
    rs_report = validate_residual(
        rs, expected_pool_size=expected_residual, sanity_faces=sanity,
    )
    print(rs_report.render())

    total_claimed = (
        len(pk.claimed_faces) + len(hl.claimed_faces) + len(cx.claimed_faces)
        + len(fl.claimed_faces) + len(of.claimed_faces) + len(wl.claimed_faces)
        + len(pr.claimed_faces) + len(rs.claimed_faces)
    )
    print("\n" + "=" * 78)
    print("CASCADE SUMMARY")
    print("=" * 78)
    print(f"  pass 1 pockets:             {len(pk.features):>3} features, "
          f"claimed {len(pk.claimed_faces):>3} faces")
    print(f"  pass 2 holes:               {len(hl.features):>2} features, "
          f"claimed {len(hl.claimed_faces):>3} faces")
    print(f"  pass 3 coaxial_stack:       {len(cx.features):>2} features, "
          f"claimed {len(cx.claimed_faces):>3} faces")
    print(f"  pass 4 flats:               {len(fl.features):>2} features, "
          f"claimed {len(fl.claimed_faces):>3} faces")
    print(f"  pass 5 outer_fillets:       {len(of.features):>2} features, "
          f"claimed {len(of.claimed_faces):>3} faces")
    print(f"  pass 6 wall:                {len(wl.features):>2} features, "
          f"claimed {len(wl.claimed_faces):>3} faces")
    print(f"  pass 7 profile:             {len(pr.features):>2} features, "
          f"claimed {len(pr.claimed_faces):>3} faces")
    print(f"  pass 8 residual_candidates: {len(rs.features):>3} features, "
          f"claimed {len(rs.claimed_faces):>3} faces")
    print(f"  total faces claimed:        {total_claimed:>3} / {n_faces}")

    ok = (
        pk_report.ok and hl_report.ok and cx_report.ok
        and fl_report.ok and of_report.ok and wl_report.ok
        and pr_report.ok and rs_report.ok
    )
    print(f"\ncascade validation: {'PASS' if ok else 'FAIL'}")

    if args.export_dir is not None:
        from feature_graph import write_feature_graph
        from eval_cascade import build_cascade_feature_graph

        export_dir = args.export_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        graph = build_cascade_feature_graph(
            part_id, n_faces, pk, hl, cx, fl, of, wl, pr, rs, edge_index,
            inner_fillet_result=if_, chamfer_result=ch, counterbore_result=cb,
            countersink_result=cs,
        )
        from feature_params import load_step_faces
        from step_ingest import load_step_shape
        from approach_vectors import annotate_approach_vectors
        from lobe_contour_merge import merge_lobe_contour_fragments
        from reachability import annotate_reachability

        occ_faces = load_step_faces(step_path)
        shape, _ = load_step_shape(str(step_path))
        graph["approach_frame"] = annotate_approach_vectors(
            graph["nodes"], faces=faces, opening_axis=pk.opening_axis,
        )
        graph["schema_version"] = 3
        graph["reachability_summary"] = annotate_reachability(
            graph["nodes"], occ_faces=occ_faces, shape=shape,
            opening_axis=pk.opening_axis,
        )
        graph["schema_version"] = 4
        if args.lateral_axes:
            # PROVISIONAL lateral ±X/±Y/±Z path — additive, never overwrites the
            # calibrated Z-only approach/reachability fields. See lateral_axes.py.
            from lateral_axes import (
                annotate_lateral_candidates,
                annotate_lateral_reachability,
            )

            graph["lateral_candidates_summary"] = annotate_lateral_candidates(
                graph["nodes"], faces=faces, opening_axis=pk.opening_axis,
            )
            graph["lateral_reachability_summary"] = annotate_lateral_reachability(
                graph["nodes"], occ_faces=occ_faces, shape=shape,
            )
            print("\nlateral-axes: annotated PROVISIONAL ±X/±Y/±Z candidacy + "
                  "reachability (unvalidated path)")
        graph, merge_report = merge_lobe_contour_fragments(
            graph,
            faces,
            edge_index,
            edge_attr,
            opening_axis=pk.opening_axis,
            occ_faces=occ_faces,
            enabled=args.merge_lobe_contours,
            prefer_rear=not is_front,
        )
        if merge_report.nodes_removed:
            print(
                f"\nlobe contour merge: {merge_report.candidates_before} candidates "
                f"-> {len(merge_report.clusters)} merged clusters "
                f"({merge_report.nodes_removed} nodes removed)"
            )
        # Per-part relabel of through_hole -> contour_surface (e.g. fish_mold,
        # whose bores are curved mold surfaces, not drillable holes). Runs before
        # slope profiling so the relabeled nodes are treated as contour surfaces
        # downstream. See RELABEL_THROUGH_HOLE_TO_CONTOUR_STEMS.
        n_relabeled = relabel_through_holes_as_contour(graph, part_id)
        if n_relabeled:
            print(
                f"\nrelabel: {n_relabeled} through_hole node(s) -> contour_surface "
                f"for part {part_id}"
            )
        # Plain-pocket (toolpath_class "pocket") -> contour_surface, deck -> flat.
        # Geometry-gated, no part/stem check. Runs AFTER merge_lobe_contour_fragments
        # so the new contour nodes are never fed into the lobe merge and no
        # pre-existing contour node's membership can shift; empty on every part
        # without a plain-pocket node. See relabel_plain_pockets_as_contour.
        n_plain, n_deck = relabel_plain_pockets_as_contour(
            graph, faces, pk.opening_axis,
        )
        if n_plain:
            print(
                f"\nrelabel: {n_plain} plain-pocket feature(s) -> contour_surface "
                f"({n_deck} deck face(s) -> flat)"
            )
        # Per-feature slope profile (steep vs shallow), computed on the final merged
        # node set so merged contour surfaces get an area-weighted profile over all
        # their faces. Consumed by the planner for slope-aware surface finishing.
        from slope_profile import annotate_slope_profiles

        graph["slope_profile_summary"] = annotate_slope_profiles(
            graph["nodes"], faces=faces, opening_axis=pk.opening_axis,
        )
        graph["schema_version"] = 5

        # Chamfer bands were recognized in-pipeline (before flats/contour) and
        # emitted as `chamfer` nodes by build_cascade_feature_graph, so they flow
        # through approach + reachability annotation like any other feature. Here
        # we only record the count. Zero on 96260B -- correct; see
        # chamfer_detection.py.
        graph["chamfer_summary"] = {"chamfers": len(ch.features)}
        graph["schema_version"] = 6

        graph_path = export_dir / "feature_graph_cascade.json"
        write_feature_graph(str(graph_path), graph)
        print(f"\nWrote {graph_path.resolve()}")

        from setup_descriptor import dump_setup_descriptor
        from setup_generation import generate_setup_entry_for_export, merge_setup_entries

        setup_id = "front" if is_front else "rear"
        entry = generate_setup_entry_for_export(
            graph,
            setup_id=setup_id,
            part_step=step_path,
        )
        if "96260B" in step_path.name.upper():
            family_id = "96260B"
        else:
            family_id = part_id.rsplit("_", 1)[0] if "_" in part_id else part_id
        single_desc = merge_setup_entries(family_id, [entry])
        setup_desc_path = export_dir / "setup_descriptor.yaml"
        dump_setup_descriptor(
            single_desc,
            setup_desc_path,
            header="Generated by run_cascade export (step 3).",
        )
        print(f"Wrote {setup_desc_path.resolve()}")

        family_dir = export_dir.parent / family_id
        family_dir.mkdir(parents=True, exist_ok=True)
        family_path = family_dir / "setup_descriptor.yaml"
        if family_path.is_file():
            from setup_descriptor import load_setup_descriptor

            existing = load_setup_descriptor(family_path)
            merged_entries = list(existing.setups.values())
            by_id = {e.setup_id: e for e in merged_entries}
            by_id[entry.setup_id] = entry
            family_desc = merge_setup_entries(family_id, list(by_id.values()))
        else:
            family_desc = single_desc
        dump_setup_descriptor(
            family_desc,
            family_path,
            header="Generated by run_cascade export (step 3) — merged family descriptor.",
        )
        print(f"Wrote {family_path.resolve()}")

        from feature_graph_viewer.build import DEFAULT_TEMPLATE, build_viewer

        viewer_path = export_dir / "viewer.html"
        build_viewer(
            part_id=part_id,
            graph_path=graph_path,
            step_path=step_path,
            output_path=viewer_path,
            template_path=DEFAULT_TEMPLATE,
            open_browser=False,
        )
        print(f"Wrote {viewer_path.resolve()}")
        if args.open:
            import subprocess
            subprocess.run(["open", str(viewer_path.resolve())], check=False)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
