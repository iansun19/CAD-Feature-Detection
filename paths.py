"""Canonical repo-root paths for fixtures and generated artifacts."""
from __future__ import annotations

from env_bootstrap import REPO_ROOT

FIXTURES_DIR = REPO_ROOT / "fixtures"
FIXTURES_STEP = FIXTURES_DIR / "step"
FIXTURES_GRAPHS = FIXTURES_DIR / "graphs"

# Well-known reference parts (renamed from awkward root filenames).
STEP_96260B_FRONT = FIXTURES_STEP / "96260B_front.stp"
STEP_96260B_REAR = FIXTURES_STEP / "96260B_rear.stp"
STEP_FISH_MOLD = FIXTURES_STEP / "fish_mold.stp"
STEP_PART1 = FIXTURES_STEP / "fixtures/step/fixtures/step/part1.step"
STEP_PART2 = FIXTURES_STEP / "fixtures/step/fixtures/step/part2.step"
STEP_PART3 = FIXTURES_STEP / "fixtures/step/fixtures/step/part3.step"
STEP_PART4 = FIXTURES_STEP / "fixtures/step/fixtures/step/part4.step"
STEP_NIST_CTC_01 = FIXTURES_STEP / "fixtures/step/fixtures/step/nist_ctc_01.step"

GRAPH_29000 = FIXTURES_GRAPHS / "fixtures/graphs/fixtures/graphs/29000_feature_graph.json"
GRAPH_NIST_CTC_01 = FIXTURES_GRAPHS / "fixtures/graphs/fixtures/graphs/nist_ctc_01_feature_graph.json"

SCHEMA_CAM_PLAN = REPO_ROOT / "schema" / "schema/schema/cam_plan.schema.json"
