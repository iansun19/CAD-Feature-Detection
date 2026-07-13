#!/usr/bin/env python3
"""Seed Supabase tool_libraries table from local Fusion JSON exports."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import env_bootstrap  # noqa: F401, E402 � loads .env

from planning.machining_context import DEFAULT_TOOL_LIBRARIES_DIR  # noqa: E402
from tools.tool_library_store import seed_tool_libraries  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Supabase tool_libraries from disk.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_TOOL_LIBRARIES_DIR,
        help="Directory containing Fusion .json library exports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log rows that would be upserted without calling Supabase",
    )
    args = parser.parse_args()

    seeded = seed_tool_libraries(args.dir, dry_run=args.dry_run)
    action = "Would seed" if args.dry_run else "Seeded"
    print(f"{action} {len(seeded)} library(ies): {', '.join(seeded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
