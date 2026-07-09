"""Tests for setup-descriptor generation (Stage 3, step 3 — interpretation A).

The round-trip and parity tests need no OCC and run in the repo .venv. The
cascade integration test lazily imports the OCC stack and skips when it is
unavailable (run it under the conda ``mlcad`` env; see [[cascade-test-runner]]).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from setup_descriptor import (  # noqa: E402
    load_setup_descriptor,
    parse_setup_descriptor,
    resolve_setup_entry,
    setup_descriptor_to_dict,
    to_pocket_setup_config,
)
from setup_generation import (  # noqa: E402
    features_by_approach_dir,
    generate_part_setup_descriptor,
    generate_setup_entry,
)

GOLDEN = Path(ROOT) / "eval" / "gt" / "96260B_setup.yaml"
FRONT_STEP = "96260B_FRONT_XR004_PCD PLATE.stp copy"
REAR_STEP = "96260B_REAR_XR004_PCD PLATE.stp copy"


def _build_generated():
    """Generate the 96260B family descriptor from the process inputs a human
    would supply, feeding the documented detected opening axis (+Y)."""
    axis = [0.0, 1.0, 0.0]
    front = generate_setup_entry(
        setup_id="front", part_step=FRONT_STEP, opening_axis=axis,
        machining_side="front", scope=["facing"],
    )
    rear = generate_setup_entry(
        setup_id="rear", part_step=REAR_STEP, opening_axis=axis,
        machining_side="back", scope="full",
    )
    return generate_part_setup_descriptor("96260B", [front, rear])


class SetupSerializerRoundTripTests(unittest.TestCase):
    def test_golden_round_trips(self):
        descriptor = load_setup_descriptor(GOLDEN)
        reparsed = parse_setup_descriptor(setup_descriptor_to_dict(descriptor))
        self.assertEqual(
            setup_descriptor_to_dict(descriptor), setup_descriptor_to_dict(reparsed),
        )


class SetupGenerationParityTests(unittest.TestCase):
    """Generated descriptor must reproduce the hand-authored golden's facts."""

    def setUp(self):
        self.gen = _build_generated()
        self.golden = load_setup_descriptor(GOLDEN)

    def test_part_and_setup_identity(self):
        self.assertEqual(self.gen.part_id, self.golden.part_id)
        self.assertEqual(set(self.gen.setups), set(self.golden.setups))

    def test_carried_inputs_match_golden(self):
        for sid in ("front", "rear"):
            g = self.gen.setups[sid]
            h = self.golden.setups[sid]
            self.assertEqual(g.machining_side, h.machining_side, sid)
            self.assertEqual(g.part_step, h.part_step, sid)
            self.assertEqual(
                g.effective_scope(self.gen.defaults),
                h.effective_scope(self.golden.defaults),
                sid,
            )

    def test_pocket_config_is_drop_in_equivalent(self):
        # to_pocket_setup_config is what the cascade actually consumes; it must
        # be identical between generated and golden for each setup.
        for sid in ("front", "rear"):
            gen_cfg = to_pocket_setup_config(
                resolve_setup_entry(self.gen, setup_id=sid)
            )
            gold_cfg = to_pocket_setup_config(
                resolve_setup_entry(self.golden, setup_id=sid)
            )
            self.assertEqual(gen_cfg, gold_cfg, sid)

    def test_opening_axis_captured_explicitly(self):
        front = self.gen.setups["front"]
        self.assertEqual(front.opening_axis.mode, "explicit")
        self.assertEqual(list(front.opening_axis.vector), [0.0, 1.0, 0.0])


class FeatureGroupingTests(unittest.TestCase):
    def test_groups_by_setup_dir(self):
        graph = {
            "nodes": [
                {"feature_id": 0, "approach": {"setup_dir": "+Z"}},
                {"feature_id": 1, "approach": {"setup_dir": "+Z"}},
                {"feature_id": 2, "approach": {"setup_dir": "-Z"}},
                {"feature_id": 3, "approach": {"setup_dir": None}},
            ]
        }
        groups = features_by_approach_dir(graph)
        self.assertEqual(sorted(groups["+Z"]), [0, 1])
        self.assertEqual(groups["-Z"], [2])
        self.assertEqual(groups[None], [3])


class SetupGenerationFromCascadeTests(unittest.TestCase):
    """End-to-end: cascade run -> step-2 graph -> generated setup entry."""

    def _run(self, step: str, side: str, npz: str):
        from eval_cascade import build_cascade_feature_graph
        from feature_params import analyze_step
        from pocket_detection import PocketDetectionConfig, resolve_pocket_setup_for_run
        from run_cascade import _load_edges, run_cascade

        step_p = Path(step)
        cfg = PocketDetectionConfig(
            setup=resolve_pocket_setup_for_run(step_p, machining_side=side),
        )
        ei, ea = _load_edges(Path(npz), step_p)
        faces = analyze_step(step_p)
        _, pk, hl, cx, fl, of, wl, pr, rs = run_cascade(
            step_p, ei, ea, pocket_config=cfg,
        )
        graph = build_cascade_feature_graph(
            f"96260B_{side}", len(faces), pk, hl, cx, fl, of, wl, pr, rs, ei,
            faces=faces, opening_axis=pk.opening_axis,
        )
        return graph

    def test_generate_from_front_cascade(self):
        try:
            import feature_params  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"OCC stack unavailable: {exc}")

        from setup_generation import generate_setup_entry_from_graph

        graph = self._run(FRONT_STEP, "front", "pipeline_out/96260B_front/graph.npz")

        # Every pocket must be reachable from the single +Z front setup.
        groups = features_by_approach_dir(graph)
        pocket_ids = {
            int(n["feature_id"])
            for n in graph["nodes"]
            if (n.get("params") or {}).get("opening_axis") is not None
        }
        self.assertTrue(pocket_ids, "expected pockets in front partition")
        self.assertTrue(pocket_ids.issubset(set(groups.get("+Z", []))))

        entry = generate_setup_entry_from_graph(
            graph, setup_id="front", part_step=FRONT_STEP,
            machining_side="front", scope=["facing"],
        )
        self.assertEqual(entry.machining_side, "front")
        self.assertEqual(entry.pocket_access, "open")
        self.assertEqual(entry.opening_axis.mode, "explicit")
        # Detected opening axis is +Y for this plate family.
        vec = entry.opening_axis.vector
        self.assertGreater(abs(vec[1]), 0.99)


if __name__ == "__main__":
    unittest.main()
