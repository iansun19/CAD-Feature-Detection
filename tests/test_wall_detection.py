"""Regression tests for exterior wall detection on 96260B_front."""
from __future__ import annotations

from pathlib import Path

from cascade.pocket_detection import PocketDetectionConfig, pocket_config_from_setup_dict
from run_cascade import _load_edges, run_cascade
from cascade.wall_detection import (
    REFERENCE_WALL_FACES_FRONT,
    REFERENCE_WALL_INSTANCES_FRONT,
    validate_walls,
)

STEP = "fixtures/step/96260B_front.stp"
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


def test_wall_reference_face_set():
    _faces, _pk, _hl, _cx, _fl, _of, wl, _pr, _rs, *_ = _front_cascade()
    report = validate_walls(
        wl,
        expected_instances=REFERENCE_WALL_INSTANCES_FRONT,
        expected_faces=sorted(REFERENCE_WALL_FACES_FRONT),
        forbidden_faces=(97, 273, 277, 280, 298),
    )
    assert report.ok, report.render()


def test_wall_one_face_per_instance():
    _faces, _pk, _hl, _cx, _fl, _of, wl, _pr, _rs, *_ = _front_cascade()
    assert len(wl.features) == REFERENCE_WALL_INSTANCES_FRONT
    assert all(len(f.face_indices) == 1 for f in wl.features)


def test_wall_seeds_match_claimed():
    _faces, _pk, _hl, _cx, _fl, _of, wl, _pr, _rs, *_ = _front_cascade()
    assert wl.seed_faces == wl.claimed_faces


if __name__ == "__main__":
    test_wall_reference_face_set()
    test_wall_one_face_per_instance()
    test_wall_seeds_match_claimed()
    print("test_wall_detection: PASS")
