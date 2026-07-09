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

Run:
  /Users/iansun19/miniconda3/envs/mlcad/bin/python run_cascade.py \
      "96260B_REAR_XR004_PCD PLATE.stp copy" \
      --graph-npz pipeline_out/96260B_plate/graph.npz -v
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

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
DEFAULT_GRAPH_NPZ = "pipeline_out/96260B_plate/graph.npz"

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
    stock_classifier: str | None = "new",
    pocket_config: PocketDetectionConfig | None = None,
    hole_config: HoleDetectionConfig | None = None,
    coaxial_config: CoaxialStackDetectionConfig | None = None,
    flat_config: FlatDetectionConfig | None = None,
    outer_fillet_config: OuterFilletDetectionConfig | None = None,
    wall_config: WallDetectionConfig | None = None,
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

    stock_faces: set[int] = set()
    if stock_classifier and stock_classifier != "off":
        from stock_cut_classification import stock_face_ids

        stock_faces = stock_face_ids(step_path, classifier=stock_classifier)  # type: ignore[arg-type]
    cut_candidates = set(range(n_faces)) - stock_faces
    if stock_faces:
        logger.info(
            "stock_cut_classification (%s): %d STOCK, %d CUT candidates",
            stock_classifier, len(stock_faces), len(cut_candidates),
        )

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

    pool_in = pocket_result.remaining_faces
    hole_result = detect_holes(
        faces, edge_index, edge_attr,
        occ_faces=occ_faces, candidate_faces=pool_in,
        config=hole_config,
    )
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

    pool_in = coaxial_result.remaining_faces
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

    return (
        faces, pocket_result, hole_result, coaxial_result, flat_result,
        outer_fillet_result, wall_result, profile_result, residual_result,
        inner_fillet_result,
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Feature-grouping cascade")
    ap.add_argument("step", nargs="?", default=DEFAULT_STEP, help="reference STEP file")
    ap.add_argument("--graph-npz", type=Path, default=Path(DEFAULT_GRAPH_NPZ),
                    help="cached face graph (edge_index/edge_attr); recomputed if absent")
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
        "--stock-classifier",
        choices=("new", "old", "off"),
        default="new",
        help="STOCK/CUT face gate before cascade (default: new convexity-primary; off=all faces)",
    )
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
        default=True,
        help="merge same-lobe cap contour_surface fragments before export (default: on)",
    )
    ap.add_argument(
        "--lateral-axes",
        action="store_true",
        default=False,
        help="ALSO annotate PROVISIONAL lateral ±X/±Y/±Z approach candidacy + "
             "reachability (unvalidated; default: off, calibrated Z-only path only)",
    )
    args = ap.parse_args(argv)

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

    faces, pk, hl, cx, fl, of, wl, pr, rs, if_ = run_cascade(
        step_path, edge_index, edge_attr,
        stock_classifier=args.stock_classifier,
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
        f"inner_fillet -> pockets -> holes -> coaxial_stack -> flats "
        f"-> outer_fillets -> wall -> profile -> residual"
    )

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
    hl_report = validate_against_expected(hl.features, REFERENCE_HOLES, exact=True)
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
            inner_fillet_result=if_,
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
        if args.stock_classifier and args.stock_classifier != "off":
            from stock_cut_classification import stock_face_ids

            graph["stock_face_ids"] = sorted(
                stock_face_ids(step_path, classifier=args.stock_classifier)  # type: ignore[arg-type]
            )
            graph["stock_classifier"] = args.stock_classifier
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
