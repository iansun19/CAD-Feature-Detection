"""Regression guard: every emitted pocket feature is one B-rep-connected component."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hole_detection import FaceGraph, induced_subgraph_components  # noqa: E402
from pocket_detection import (  # noqa: E402
    POCKET_CONNECTIVITY_CLASSES,
    PocketConnectivityError,
    PocketFeature,
    assert_pocket_feature_connected,
)

FRONT_CASCADE = Path("pipeline_out/96260B_front/feature_graph_cascade.json")
FRONT_NPZ = Path("pipeline_out/96260B_front/graph.npz")
REAR_CASCADE = Path("pipeline_out/96260B_rear/feature_graph_cascade.json")
REAR_NPZ = Path("pipeline_out/96260B_plate/graph.npz")


def _face_graph_from_npz(npz_path: Path, n_faces: int) -> FaceGraph:
    data = np.load(npz_path)
    edge_index = np.asarray(data["edge_index"])
    edge_attr = np.asarray(data["edge_attr"]) if "edge_attr" in data else np.zeros(
        (edge_index.shape[1], 4), dtype=np.float64,
    )
    return FaceGraph.from_edge_tensors(edge_index, edge_attr, n_faces)


def _pocket_nodes(cascade_path: Path) -> list[dict]:
    with open(cascade_path) as f:
        graph = json.load(f)
    nodes = []
    for node in graph["nodes"]:
        class_name = node.get("class_name", "")
        if class_name in POCKET_CONNECTIVITY_CLASSES:
            nodes.append(node)
    return nodes


def _assert_cascade_pockets_connected(
    cascade_path: Path,
    npz_path: Path,
    *,
    part_id: str,
) -> None:
    if not cascade_path.is_file():
        raise unittest.SkipTest(f"missing cascade output: {cascade_path}")
    if not npz_path.is_file():
        raise unittest.SkipTest(f"missing graph: {npz_path}")

    with open(cascade_path) as f:
        n_faces = int(json.load(f)["n_faces"])
    graph = _face_graph_from_npz(npz_path, n_faces)

    failures: list[str] = []
    for node in _pocket_nodes(cascade_path):
        face_ids = node["face_ids"]
        comps = induced_subgraph_components(face_ids, graph)
        if len(comps) != 1:
            failures.append(
                f"{part_id} feature_id={node['feature_id']} "
                f"class={node['class_name']} has {len(comps)} components: "
                f"{[len(c) for c in comps]}"
            )
    if failures:
        self_fail = "\n".join(failures)
        raise AssertionError(
            "pocket connectivity invariant violated:\n" + self_fail
        )


class TestPocketConnectivityGuard(unittest.TestCase):
    def test_front_cascade_pockets_connected(self) -> None:
        _assert_cascade_pockets_connected(
            FRONT_CASCADE, FRONT_NPZ, part_id="96260B_front",
        )

    def test_rear_cascade_pockets_connected(self) -> None:
        _assert_cascade_pockets_connected(
            REAR_CASCADE, REAR_NPZ, part_id="96260B_rear",
        )

    def test_guard_rejects_disconnected_feature(self) -> None:
        graph = FaceGraph(4)
        graph.neighbors[0].add(1)
        graph.neighbors[1].add(0)
        graph.neighbors[2].add(3)
        graph.neighbors[3].add(2)
        feat = PocketFeature(
            feature_id=99,
            kind="pocket",
            subtype="blind_pocket",
            face_indices={0, 1, 2, 3},
            centroid_uv=(0.0, 0.0),
            opening_axis=(0.0, 1.0, 0.0),
            wall_diameters={},
            wall_count=0,
            floor_count=0,
            sphere_count=0,
            step_plane_count=0,
            released_faces=[],
            released_by_type={},
            depth_below_top_mm=None,
            fillet_radius_mm=None,
            surface_3d=False,
            blend_ring={},
            blend_face_indices=[],
            template_match=True,
            toolpath_class="filleted_pocket",
        )
        with self.assertRaises(PocketConnectivityError) as ctx:
            assert_pocket_feature_connected(feat, graph)
        msg = str(ctx.exception)
        self.assertIn("feature_id=99", msg)
        self.assertIn("filleted_pocket", msg)
        self.assertIn("2 disconnected components", msg)
        self.assertIn("comp[0]=", msg)
        self.assertIn("comp[1]=", msg)


if __name__ == "__main__":
    unittest.main()
