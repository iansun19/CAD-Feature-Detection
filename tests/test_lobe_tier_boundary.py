"""Regression tests for lobe tier mouth-boundary classification."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from cascade_instrumentation import _trace_split_lobe_pool_assignments
from eval_cascade import build_cascade_feature_graph, eval_part
from feature_params import analyze_step, load_step_faces
from hole_detection import FaceGraph
from lobe_tier_detection import (
    LobeTierConfig,
    TOLERANCE_EPSILON_MM,
    TOLERANCE_FLOOR_MM,
    DEFAULT_MOUTH_TIER_SPAN_FRACTION,
    _mouth_tier_boundary_tol_mm,
    _opening_side_lobe_wall,
    detect_filleted_lobe_tiers,
)
from pocket_detection import (
    PocketDetectionConfig,
    detect_pockets,
    resolve_pocket_setup_for_run,
)
from run_cascade import _load_edges, run_cascade

OPENING_BORE_WALLS = frozenset({
    258, 268, 270, 237, 247, 212, 222, 185, 187, 195,
    162, 170, 172, 135, 137, 145, 147, 108, 122,
    110, 197, 260,
})

FRONT_SAFETY_NET_ORPHAN = frozenset({134, 52, 138, 183})

FRONT_STEP = Path("96260B_FRONT_XR004_PCD PLATE.stp copy")
FRONT_NPZ = Path("pipeline_out/96260B_front/graph.npz")
FRONT_GT = Path("eval/gt/96260B_front.yaml")
REAR_STEP = Path("96260B_REAR_XR004_PCD PLATE.stp copy")
REAR_NPZ = Path("pipeline_out/96260B_plate/graph.npz")

FRONT_PARTITION_BASELINE_MD5 = "6763ebe651426c409ae5738fab691a95"
REAR_PARTITION_CONFIG1_MD5 = "894e56e2d5371c9cbd21f0bef883c66b"

# cap_area / torus_area on 96260B: real split-export caps ? 0.149; sculpt spheres ? 0.206.
REAL_CAP_AREA_RATIO = 0.149
SCULPT_SPHERE_AREA_RATIO = 0.206
DEFAULT_AREA_RATIO_GUARD = 0.18
REAR_OPEN_MOUTH_FACES = frozenset({
    54, 60, 226, 39, 44, 24, 30, 6, 15, 300, 97, 103, 84, 90, 69, 75,
    # Sculpt-cap bsplines convex-bridged to open-tier cap spheres (96260B rear
    # lobes 0/2); previously stranded in closed tier by crown-sphere gate.
    7, 45,
})
# Previously asserted closed-tier deep band; reassigned to open for connectivity.
REAR_CLOSED_SCULPT_CAP_BSPLINES = frozenset()
REAR_CAP_SPHERES = frozenset({6, 44, 68, 74})
REAR_CAP_ANNEX_SET = frozenset({6, 14, 23, 29, 38, 44, 53, 59, 68, 74, 83, 89, 96, 102})
FRONT_SCULPT_SPHERE_FP = 123
REAR_SCULPT_SPHERE_FP = 128


def _load_front():
    config = PocketDetectionConfig(
        setup=resolve_pocket_setup_for_run(FRONT_STEP, machining_side="front"),
    )
    edge_index, edge_attr = _load_edges(FRONT_NPZ, FRONT_STEP)
    faces = analyze_step(FRONT_STEP)
    occ = load_step_faces(FRONT_STEP)
    return faces, occ, edge_index, edge_attr, config


def _load_rear():
    config = PocketDetectionConfig(
        setup=resolve_pocket_setup_for_run(REAR_STEP, machining_side="rear"),
    )
    edge_index, edge_attr = _load_edges(REAR_NPZ, REAR_STEP)
    faces = analyze_step(REAR_STEP)
    occ = load_step_faces(REAR_STEP)
    return faces, occ, edge_index, edge_attr, config


def _partition_md5(lobe) -> str:
    payload = {
        "lobes": [
            {"open": sorted(l.open_faces), "closed": sorted(l.closed_faces)}
            for l in lobe.lobes
        ]
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _annexed_spheres(lobe) -> set[int]:
    out: set[int] = set()
    for lob in lobe.lobes:
        for entry in lob.fillet_sphere_cap_annex:
            out.add(int(entry["sphere_face"]))
    return out


def _tier_label(lobe, face_id: int) -> tuple[str | None, str | None]:
    for lob in lobe.lobes:
        if face_id in lob.open_faces:
            return "filleted_open_pocket", "mouth"
        if face_id in lob.closed_faces:
            return "filleted_pocket", "deep"
    return None, None


def _detect(step, side, npz, cfg: LobeTierConfig | None = None):
    cfg = cfg or LobeTierConfig()
    edge_index, edge_attr = _load_edges(npz, step)
    faces = analyze_step(step)
    occ = load_step_faces(step)
    pk = detect_pockets(
        faces, edge_index, edge_attr, occ_faces=occ,
        config=PocketDetectionConfig(
            setup=resolve_pocket_setup_for_run(step, machining_side=side),
        ),
    )
    lobe = detect_filleted_lobe_tiers(
        faces, edge_index, edge_attr, occ_faces=occ,
        opening_axis=pk.opening_axis, config=cfg, n_lobes_hint=7,
    )
    return lobe


class LobeTierBoundaryTests(unittest.TestCase):
    def test_tol_precision_floor_arithmetic(self):
        clearance = 0.12
        span = 5.08
        span_tol = DEFAULT_MOUTH_TIER_SPAN_FRACTION * span
        tol_candidate = min(span_tol, 0.5 * clearance - TOLERANCE_EPSILON_MM)
        self.assertLess(tol_candidate, TOLERANCE_FLOOR_MM)

    def test_measured_front_tol(self):
        faces, occ, edge_index, edge_attr, config = _load_front()
        by_index = {f.index: f for f in faces}
        occ_map = {i: occ[i] for i in range(len(faces))}
        graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(faces))
        cfg = LobeTierConfig()
        pk = detect_pockets(faces, edge_index, edge_attr, occ_faces=occ, config=config)
        lobe = detect_filleted_lobe_tiers(
            faces, edge_index, edge_attr, occ_faces=occ,
            opening_axis=pk.opening_axis, config=cfg, n_lobes_hint=7,
        )
        axis = pk.opening_axis
        tol = _mouth_tier_boundary_tol_mm(
            lobe.lobes[0].pool_faces,
            lobe.lobes[0].mouth_step_face,
            lobe.lobes[0].deep_step_face,
            by_index, occ_map, graph, axis, cfg,
        )
        self.assertGreaterEqual(tol, cfg.mouth_tolerance_floor_mm)
        self.assertAlmostEqual(tol, 0.254, places=2)

    def test_opening_bore_walls_closed_tier(self):
        faces, occ, edge_index, edge_attr, config = _load_front()
        by_index = {f.index: f for f in faces}
        occ_map = {i: occ[i] for i in range(len(faces))}
        graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(faces))
        cfg = LobeTierConfig()
        pk = detect_pockets(faces, edge_index, edge_attr, occ_faces=occ, config=config)
        lobe = detect_filleted_lobe_tiers(
            faces, edge_index, edge_attr, occ_faces=occ,
            opening_axis=pk.opening_axis, config=cfg, n_lobes_hint=7,
        )
        reasons: dict[int, str] = {}
        closed_by_face: dict[int, bool] = {}
        for lob in lobe.lobes:
            _, closed_f, assign = _trace_split_lobe_pool_assignments(
                lob.pool_faces, lob.mouth_step_face, lob.deep_step_face,
                graph, by_index, occ_map, pk.opening_axis, cfg,
            )
            for fid in OPENING_BORE_WALLS:
                if fid not in lob.pool_faces:
                    continue
                closed_by_face[fid] = fid in closed_f
                reasons[fid] = assign[fid]
        for fid in OPENING_BORE_WALLS:
            self.assertTrue(
                closed_by_face.get(fid),
                f"face {fid} expected closed tier",
            )
            self.assertEqual(
                reasons.get(fid), "closed_tier_open_ext",
                f"face {fid} expected closed_tier_open_ext, got {reasons.get(fid)}",
            )

    def test_safety_net_faces_stay_orphan(self):
        faces, occ, edge_index, edge_attr, config = _load_front()
        by_index = {f.index: f for f in faces}
        occ_map = {i: occ[i] for i in range(len(faces))}
        graph = FaceGraph.from_edge_tensors(edge_index, edge_attr, len(faces))
        cfg = LobeTierConfig()
        pk = detect_pockets(faces, edge_index, edge_attr, occ_faces=occ, config=config)
        lobe = detect_filleted_lobe_tiers(
            faces, edge_index, edge_attr, occ_faces=occ,
            opening_axis=pk.opening_axis, config=cfg, n_lobes_hint=7,
        )
        for lob in lobe.lobes:
            _, _, reasons = _trace_split_lobe_pool_assignments(
                lob.pool_faces, lob.mouth_step_face, lob.deep_step_face,
                graph, by_index, occ_map, pk.opening_axis, cfg,
            )
            tol = _mouth_tier_boundary_tol_mm(
                lob.pool_faces, lob.mouth_step_face, lob.deep_step_face,
                by_index, occ_map, graph, pk.opening_axis, cfg,
            )
            for fid in FRONT_SAFETY_NET_ORPHAN & lob.pool_faces:
                self.assertEqual(reasons.get(fid), "orphan_reassigned")
                self.assertFalse(_opening_side_lobe_wall(
                    fid, lob.mouth_step_face, tol,
                    by_index, occ_map, graph, pk.opening_axis, cfg,
                ))

    def test_front_cascade_classes(self):
        faces, occ, edge_index, edge_attr, config = _load_front()
        _, pk, hl, cx, fl, of, wl, pr, rs, *_ = run_cascade(
            FRONT_STEP, edge_index, edge_attr, pocket_config=config,
        )
        graph = build_cascade_feature_graph(
            "96260B_front", len(faces), pk, hl, cx, fl, of, wl, pr, rs, edge_index,
        )
        labels = {fi: n["class_name"] for n in graph["nodes"] for fi in n["face_ids"]}
        for fid in OPENING_BORE_WALLS:
            self.assertEqual(labels.get(fid), "filleted_pocket")

    def test_front_eval_counts(self):
        report = eval_part(FRONT_GT)
        self.assertEqual(report.count_divergences, [])

    def test_fillet_sphere_cap_area_ratio_guard(self):
        self.assertAlmostEqual(
            LobeTierConfig().fillet_sphere_cap_max_torus_area_ratio,
            DEFAULT_AREA_RATIO_GUARD,
        )
        self.assertLess(REAL_CAP_AREA_RATIO, DEFAULT_AREA_RATIO_GUARD)
        self.assertGreater(SCULPT_SPHERE_AREA_RATIO, DEFAULT_AREA_RATIO_GUARD)

        front = _detect(FRONT_STEP, "front", FRONT_NPZ)
        rear = _detect(REAR_STEP, "rear", REAR_NPZ)

        self.assertEqual(_partition_md5(front), FRONT_PARTITION_BASELINE_MD5)
        self.assertEqual(_partition_md5(rear), REAR_PARTITION_CONFIG1_MD5)
        self.assertEqual(_annexed_spheres(rear), REAR_CAP_ANNEX_SET)
        self.assertEqual(_annexed_spheres(front), set())

        for fid in REAR_CAP_SPHERES:
            tier, band = _tier_label(rear, fid)
            self.assertEqual(tier, "filleted_open_pocket", f"rear {fid}")
            self.assertEqual(band, "mouth", f"rear {fid}")

        for fid in (FRONT_SCULPT_SPHERE_FP, REAR_SCULPT_SPHERE_FP):
            self.assertNotIn(fid, _annexed_spheres(front) | _annexed_spheres(rear))

        for entry in (
            e for lob in rear.lobes for e in lob.fillet_sphere_cap_annex
        ):
            ratio = entry["cap_torus_area_ratio"]
            self.assertIsNotNone(ratio)
            self.assertLessEqual(float(ratio), DEFAULT_AREA_RATIO_GUARD)
            self.assertLess(float(ratio), SCULPT_SPHERE_AREA_RATIO)

        leaky = _detect(
            FRONT_STEP, "front", FRONT_NPZ,
            LobeTierConfig(fillet_sphere_cap_max_torus_area_ratio=1.0),
        )
        self.assertIn(FRONT_SCULPT_SPHERE_FP, _annexed_spheres(leaky))
        self.assertNotEqual(_partition_md5(leaky), FRONT_PARTITION_BASELINE_MD5)

        disabled = _detect(
            REAR_STEP, "rear", REAR_NPZ,
            LobeTierConfig(fillet_sphere_cap_max_torus_area_ratio=None),
        )
        self.assertIn(REAR_SCULPT_SPHERE_FP, _annexed_spheres(disabled))
        self.assertGreater(len(_annexed_spheres(disabled)), len(REAR_CAP_ANNEX_SET))

    def test_rear_mouth_band_face_tiers(self):
        rear = _detect(REAR_STEP, "rear", REAR_NPZ)
        for fid in REAR_OPEN_MOUTH_FACES:
            tier, band = _tier_label(rear, fid)
            self.assertEqual(tier, "filleted_open_pocket", f"rear {fid}")
            self.assertEqual(band, "mouth", f"rear {fid}")
        for fid in REAR_CLOSED_SCULPT_CAP_BSPLINES:
            tier, band = _tier_label(rear, fid)
            self.assertEqual(tier, "filleted_pocket", f"rear {fid}")
            self.assertEqual(band, "deep", f"rear {fid}")

    def test_rear_sculpt_cap_bsplines_emit_open_class(self):
        """Open-tier sculpt caps (7, 45) stay filleted_open_pocket under rear setup."""
        from pocket_detection import (
            PocketDetectionConfig,
            PocketSetupConfig,
            apply_filleted_lobe_tiers_to_result,
            detect_pockets,
        )

        setup = PocketSetupConfig(machining_side="back", pocket_access="closed")
        edge_index, edge_attr = _load_edges(REAR_NPZ, REAR_STEP)
        faces = analyze_step(REAR_STEP)
        occ = load_step_faces(REAR_STEP)
        config = PocketDetectionConfig(setup=setup)
        pk = detect_pockets(faces, edge_index, edge_attr, occ_faces=occ, config=config)
        result = apply_filleted_lobe_tiers_to_result(
            pk, faces, edge_index, edge_attr, occ, config,
        )
        for fid in (7, 45):
            feat = next(f for f in result.features if fid in f.face_indices)
            self.assertEqual(feat.access, "open", f"rear face {fid}")
            self.assertEqual(
                feat.toolpath_class, "filleted_open_pocket", f"rear face {fid}",
            )


if __name__ == "__main__":
    unittest.main()
