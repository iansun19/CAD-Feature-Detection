"""Regression tests for hub step profile detection on 96260B_front."""
from __future__ import annotations

from pathlib import Path

from outer_fillet_detection import (
    REFERENCE_HUB_OUTER_FILLET_FACES_FRONT,
    REFERENCE_OUTER_FILLET_FACES_FRONT,
)
from pocket_detection import PocketDetectionConfig, pocket_config_from_setup_dict
from profile_detection import (
    REFERENCE_PROFILE_FACES_FRONT,
    REFERENCE_PROFILE_INSTANCES_FRONT,
    validate_profiles,
)
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


def test_profile_reference_face_set():
    _faces, _pk, _hl, _cx, _fl, _of, _wl, pr, _rs = _front_cascade()
    report = validate_profiles(
        pr,
        expected_instances=REFERENCE_PROFILE_INSTANCES_FRONT,
        expected_faces=sorted(REFERENCE_PROFILE_FACES_FRONT),
        forbidden_faces=(97, 273, 277, 280, 298),
    )
    assert report.ok, report.render()


def test_profile_not_in_residual_or_hub():
    hub = {273, 277, 281, 282, 283, 274, 275, 276}
    hub |= set(REFERENCE_OUTER_FILLET_FACES_FRONT)
    hub |= set(REFERENCE_HUB_OUTER_FILLET_FACES_FRONT)
    _faces, _pk, _hl, _cx, _fl, _of, _wl, pr, rs = _front_cascade()
    profile_faces = set().union(*(f.face_indices for f in pr.features))
    assert not (profile_faces & hub)
    assert not (profile_faces & rs.claimed_faces)


def test_profile_hub_step_diameter():
    _faces, _pk, _hl, _cx, _fl, _of, _wl, pr, _rs = _front_cascade()
    assert len(pr.features) == 1
    feat = pr.features[0]
    assert feat.face_indices == REFERENCE_PROFILE_FACES_FRONT
    assert 160.0 < feat.nominal_diameter_mm < 163.0
    assert feat.n_faces == 2


if __name__ == "__main__":
    test_profile_reference_face_set()
    test_profile_not_in_residual_or_hub()
    test_profile_hub_step_diameter()
    print("test_profile_detection: PASS")
