#!/usr/bin/env python3
"""One-time ingest of local Fusion tool libraries into Supabase ``tools`` table.

Uses machining_context.load_tool_library() for parsing; Supabase stores normalized
Tool rows (not raw Fusion JSON). Re-running is idempotent: upsert on ``guid`` with
latest row winning on conflict.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import env_bootstrap  # noqa: F401, E402 - loads .env

from tool_store import (  # noqa: E402
    IngestFileResult,
    prepare_tool_rows_from_library,
    upsert_tool_rows,
)

logger = logging.getLogger(__name__)


def _library_paths(dir_path: Path, only_files: list[str] | None = None) -> list[Path]:
    for hsmlib in sorted(dir_path.glob("*.hsmlib")):
        logger.warning(
            "skipping unsupported .hsmlib library %s; "
            "re-export as JSON from Fusion/Toolpath",
            hsmlib.name,
        )
    paths = sorted(
        [*dir_path.glob("*.json"), *dir_path.glob("*.tools")],
        key=lambda p: p.name.lower(),
    )
    if not only_files:
        return paths

    wanted = {name.strip().lower() for name in only_files}
    filtered = [
        path
        for path in paths
        if path.name.lower() in wanted or path.stem.lower() in wanted
    ]
    missing = wanted - {path.name.lower() for path in filtered} - {
        path.stem.lower() for path in filtered
    }
    for name in sorted(missing):
        logger.warning("no library file matched --only %r", name)
    return filtered


def _log_file_result(result: IngestFileResult) -> None:
    unit_summary = ", ".join(f"{k}={v}" for k, v in sorted(result.unit_counts.items()))
    if "millimeters" in result.unit_counts:
        logger.warning(
            "METRIC LIBRARY %s: unit=millimeters (%d tools) - metric path untested on real data",
            result.path.name,
            result.unit_counts["millimeters"],
        )
    logger.info(
        "%s: ingested=%d skipped=%d holders_skipped=%d unknown_types=%d units=[%s]",
        result.path.name,
        result.tools_ingested,
        result.tools_skipped,
        result.holders_skipped,
        sum(result.unknown_types.values()),
        unit_summary or "n/a",
    )
    if result.unknown_types:
        for raw_type, count in sorted(result.unknown_types.items()):
            logger.warning("  unknown type in %s: %r x%d", result.path.name, raw_type, count)


def ingest_directory(
    dir_path: Path,
    *,
    dry_run: bool = False,
    client: object | None = None,
    only_files: list[str] | None = None,
) -> tuple[list[IngestFileResult], Counter[str], set[str], list[str]]:
    """Parse and upsert all libraries under ``dir_path``; return per-file results."""
    if not dir_path.is_dir():
        raise NotADirectoryError(f"tool library directory not found: {dir_path}")

    paths = _library_paths(dir_path, only_files)
    if not paths:
        logger.info("no .json or .tools libraries found in %s", dir_path)
        return [], Counter(), set(), []

    file_results: list[IngestFileResult] = []
    global_unknown: Counter[str] = Counter()
    all_guids: set[str] = set()
    file_failures: list[str] = []
    chunk_failures: list[str] = []
    rows_written = 0

    for path in paths:
        result = prepare_tool_rows_from_library(path, material=None)
        if result.error:
            msg = f"{path.name}: {result.error}"
            logger.error("FAILED %s", msg)
            file_failures.append(msg)
            file_results.append(result)
            continue

        _log_file_result(result)
        file_results.append(result)
        global_unknown.update(result.unknown_types)

        for row in result.rows:
            guid = row.get("guid")
            if guid:
                all_guids.add(str(guid))

        upsert_result = upsert_tool_rows(
            result.rows,
            client=client,
            dry_run=dry_run,
            source_label=path.name,
        )
        rows_written += upsert_result.upserted
        chunk_failures.extend(upsert_result.failed_chunks)

        if result.rows and upsert_result.upserted == 0:
            msg = f"{path.name}: all upsert chunks failed"
            logger.error("FAILED %s", msg)
            file_failures.append(msg)
        elif upsert_result.failed_chunks:
            msg = (
                f"{path.name}: partial upsert "
                f"({upsert_result.upserted}/{upsert_result.attempted} rows written)"
            )
            logger.error("FAILED %s", msg)
            file_failures.append(msg)

    total_rows = sum(len(r.rows) for r in file_results if not r.error)
    logger.info(
        "ingest summary: files=%d file_failures=%d chunk_failures=%d "
        "row_upserts=%d rows_written=%d unique_guids=%d",
        len(paths),
        len(file_failures),
        len(chunk_failures),
        total_rows,
        rows_written,
        len(all_guids),
    )
    if global_unknown:
        logger.warning("unknown Fusion types (global, by raw string):")
        for raw_type, count in global_unknown.most_common():
            logger.warning("  %r: %d", raw_type, count)
    if chunk_failures:
        logger.error("failed chunks (%d):", len(chunk_failures))
        for msg in chunk_failures:
            logger.error("  %s", msg)
    if file_failures:
        logger.error("failed files (%d):", len(file_failures))
        for msg in file_failures:
            logger.error("  %s", msg)

    return file_results, global_unknown, all_guids, file_failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Fusion tool libraries into Supabase tools table.",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Local directory of Fusion .json/.tools exports (not committed to git)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and log stats without calling Supabase",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="FILE",
        help="Ingest only matching library filename(s); repeatable (idempotent re-run)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    _, global_unknown, guids, file_failures = ingest_directory(
        args.directory.resolve(),
        dry_run=args.dry_run,
        only_files=args.only,
    )
    action = "Would ingest" if args.dry_run else "Ingested"
    print(
        f"{action} {len(guids)} unique tool(s) from {args.directory} "
        f"({sum(global_unknown.values())} unknown-type occurrences, "
        f"{len(file_failures)} file failure(s))"
    )
    return 1 if file_failures and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
