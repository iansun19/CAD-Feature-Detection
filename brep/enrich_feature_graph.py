#!/usr/bin/env python3
"""
enrich_feature_graph.py — add Phase-3 params to an existing feature_graph.json.

Requires pythonocc in the active environment (see environment.yml). No torch needed.

Usage:
    python enrich_feature_graph.py fixtures/graphs/fixtures/graphs/fixtures/graphs/fixtures/graphs/29000_feature_graph.json --step MFCAD++_dataset/step/test/29000.step
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from brep.feature_graph import write_feature_graph
from brep.feature_params import enrich_graph_with_params


def main():
    ap = argparse.ArgumentParser(description="Attach geometry params to a feature graph")
    ap.add_argument("graph", help="feature_graph.json path (updated in place unless -o)")
    ap.add_argument("--step", required=True, help="STEP file matching face order")
    ap.add_argument("-o", "--output", default=None, help="output path (default: overwrite input)")
    args = ap.parse_args()

    graph_path = Path(args.graph)
    with open(graph_path) as f:
        graph = json.load(f)

    enrich_graph_with_params(graph, args.step)

    out = Path(args.output) if args.output else graph_path
    write_feature_graph(str(out), graph)
    print(f"wrote {out}  (schema v{graph['schema_version']}, {len(graph['nodes'])} nodes with params)")


if __name__ == "__main__":
    main()
