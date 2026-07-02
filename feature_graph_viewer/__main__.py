"""Generate an interactive HTML viewer for a feature graph."""
from __future__ import annotations

import argparse
from pathlib import Path

from feature_graph_viewer.build import build_viewer

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = PKG_DIR / "template.html"


def main():
    ap = argparse.ArgumentParser(
        description="Build HTML 3D viewer with numbered faces + feature graph overlay",
    )
    ap.add_argument("--part-id", default="29000", help="MFCAD++ part id (default: 29000)")
    ap.add_argument(
        "--graph", "-g",
        default="29000_feature_graph.json",
        help="feature_graph.json path",
    )
    ap.add_argument(
        "--step",
        default=None,
        help="STEP file (default: MFCAD++_dataset/step/test/<part-id>.step)",
    )
    ap.add_argument(
        "--output", "-o",
        default=None,
        help="output HTML (default: <part-id>_graph_viewer.html)",
    )
    ap.add_argument("--open", action="store_true", help="open in browser after build")
    args = ap.parse_args()

    part_id = args.part_id
    graph_path = Path(args.graph)
    step_path = Path(args.step) if args.step else None
    output_path = Path(args.output or f"{part_id}_graph_viewer.html")

    out = build_viewer(
        part_id=part_id,
        graph_path=graph_path,
        step_path=step_path,
        output_path=output_path,
        template_path=DEFAULT_TEMPLATE,
        open_browser=args.open,
    )
    print(f"wrote {out.resolve()}")
    print("  Open in a browser. Each face shows its index; colors = feature instance.")
    print("  Red lines connect adjacent features. Click a feature in the sidebar to highlight.")


if __name__ == "__main__":
    main()
