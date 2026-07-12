"""Regression tests for coaxial hub stack detection on 96260B_front."""
from __future__ import annotations

from pathlib import Path

from coaxial_stack_detection import (
    REFERENCE_CONTOUR_FACES_FRONT,
    REFERENCE_HUB_FLAT_FACES_FRONT,
    REFERENCE_OPEN_POCKET_FACES_FRONT,
    validate_coaxial_stack,
)
from outer_fillet_detection import (
    REFERENCE_HUB_OUTER_FILLET_FACES_FRONT,
    REFERENCE_OUTER_FILLET_FACES_FRONT,
    validate_outer_fillets,
)
from pocket_detection import PocketDetectionConfig, pocket_config_from_setup_dict
from run_cascade import _load_edges, run_cascade

STEP = "96260B_FRONT_XR004_PCD PLATE.stp copy"
NPZ = "pipeline_out/96260B_front/graph.npz"


def _front_cascade():
    if not Path(STEP).is_file():
        raise SystemExit(f"skip: reference STEP missing: {STEP}")
    edge_index, edge_attr = _load_edges(Path(NPZ), Path(STEP))
    setup = pocket_config_from_setup_dict({"setup": {"machining_side": "front"}})
    return run_cascade(
        STEP, edge_index, edge_attr,
        pocket_config=PocketDetectionConfig(setup=setup),
    )


def test_coaxial_hub_face_sets():
    _faces, _pk, _hl, cx, _fl, _of, _wl, _pr, _rs, *_ = _front_cascade()
    report = validate_coaxial_stack(
        cx,
        expected_hub_flat_faces=REFERENCE_HUB_FLAT_FACES_FRONT,
        expected_open_pocket_faces=REFERENCE_OPEN_POCKET_FACES_FRONT,
        expected_contour_faces=REFERENCE_CONTOUR_FACES_FRONT,
        forbidden_faces=(97, 280, 298),
    )
    assert report.ok, report.render()


def test_flat_instances_include_hub():
    _faces, _pk, _hl, cx, fl, _of, _wl, _pr, _rs, *_ = _front_cascade()
    assert 97 in fl.claimed_faces
    assert set(REFERENCE_HUB_FLAT_FACES_FRONT) <= fl.claimed_faces
    assert cx.hub_flat_faces == set(REFERENCE_HUB_FLAT_FACES_FRONT)


def test_hub_not_in_residual():
    hub = (
        set(REFERENCE_HUB_FLAT_FACES_FRONT)
        | set(REFERENCE_OPEN_POCKET_FACES_FRONT)
        | set(REFERENCE_CONTOUR_FACES_FRONT)
        | set(REFERENCE_HUB_OUTER_FILLET_FACES_FRONT)
    )
    _faces, _pk, _hl, _cx, _fl, _of, _wl, _pr, rs, *_ = _front_cascade()
    assert not (hub & rs.claimed_faces)


def test_outer_fillet_opening_and_hub_tiers():
    _faces, _pk, _hl, _cx, _fl, of_r, _wl, _pr, rs, *_ = _front_cascade()
    exp = set(REFERENCE_OUTER_FILLET_FACES_FRONT) | set(REFERENCE_HUB_OUTER_FILLET_FACES_FRONT)
    report = validate_outer_fillets(
        of_r,
        expected_instances=2,
        expected_faces=sorted(exp),
    )
    assert report.ok, report.render()
    assert not (exp & rs.claimed_faces)


if __name__ == "__main__":
    test_coaxial_hub_face_sets()
    test_flat_instances_include_hub()
    test_hub_not_in_residual()
    test_outer_fillet_opening_and_hub_tiers()
    print("test_coaxial_stack: PASS")
