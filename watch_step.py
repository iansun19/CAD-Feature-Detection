#!/usr/bin/env python
"""Watch an inbox directory and auto-run the STEP -> CamPlan pipeline on drop.

A dependency-free (stdlib-only) polling watcher around ``run_step_to_plan.py``.
Drop a ``.step``/``.stp`` file into the inbox and the full perception ->
context -> planning chain runs automatically, writing the CamPlan JSON.

    ~/miniconda3/envs/mlcad/bin/python watch_step.py            # watch pipeline_in/
    ~/miniconda3/envs/mlcad/bin/python watch_step.py --once     # drain then exit

Requires the conda ``mlcad`` env (pythonocc / OCC); the repo .venv lacks OCC.
The watcher re-invokes itself via ``sys.executable`` per file, so a failure on
one part never kills the loop.

Per-file shop inputs
--------------------
The two facts the cascade cannot derive from geometry (--machining-side and
--opening-axis) cannot be guessed by an unattended watcher. Supply them with a
sidecar file next to the STEP, named ``<step-name>.args`` (e.g. ``part.step`` ->
``part.step.args``), containing extra CLI flags for run_step_to_plan.py:

    --machining-side front --opening-axis +Z

Without a sidecar the part is planned on its geometry-resolved axis; parts whose
axis cannot be resolved will fail and land in failed/ with the error logged.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
STEP_SUFFIXES = {".step", ".stp"}
LOG = logging.getLogger("watch_step")


def _is_stable(path: Path, settle: float) -> bool:
    """True once the file has stopped growing (finished copying into the inbox)."""
    try:
        first = path.stat().st_size
    except OSError:
        return False
    time.sleep(settle)
    try:
        return path.stat().st_size == first
    except OSError:
        return False


def _read_sidecar_args(step_path: Path) -> list[str]:
    """Extra run_step_to_plan.py flags from ``<step-name>.args``, if present."""
    sidecar = step_path.with_name(step_path.name + ".args")
    if not sidecar.is_file():
        return []
    text = sidecar.read_text(encoding="utf-8")
    # Ignore comment lines; join the rest and shell-split.
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
    extra = shlex.split(" ".join(lines))
    if extra:
        LOG.info("  sidecar args: %s", " ".join(extra))
    return extra


def _process(step_path: Path, processed_dir: Path, failed_dir: Path) -> bool:
    LOG.info("Processing %s", step_path.name)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_step_to_plan.py"),
        str(step_path),
        *_read_sidecar_args(step_path),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    ok = proc.returncode == 0
    dest_dir = processed_dir if ok else failed_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / step_path.name
    if dest.exists():
        dest.unlink()
    step_path.rename(dest)
    sidecar = step_path.with_name(step_path.name + ".args")
    if sidecar.is_file():
        sidecar.rename(dest_dir / sidecar.name)
    if ok:
        LOG.info("  done -> moved to %s/", dest_dir.name)
    else:
        LOG.error("  FAILED (exit %d) -> moved to %s/", proc.returncode, dest_dir.name)
    return ok


def _scan(inbox: Path, processed_dir: Path, failed_dir: Path, settle: float) -> int:
    n = 0
    for entry in sorted(inbox.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in STEP_SUFFIXES:
            continue
        if not _is_stable(entry, settle):
            LOG.info("Skipping %s (still being written)", entry.name)
            continue
        _process(entry, processed_dir, failed_dir)
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--inbox",
        type=Path,
        default=REPO_ROOT / "pipeline_in",
        help="directory to watch for dropped STEP files (default: pipeline_in/)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="seconds between inbox scans (default: 2.0)",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=1.0,
        help="seconds to confirm a file's size is stable before processing (default: 1.0)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="process everything currently in the inbox, then exit",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    inbox = args.inbox
    inbox.mkdir(parents=True, exist_ok=True)
    processed_dir = inbox / "processed"
    failed_dir = inbox / "failed"

    if args.once:
        _scan(inbox, processed_dir, failed_dir, args.settle)
        return 0

    LOG.info("Watching %s (Ctrl-C to stop). Drop .step/.stp files to plan them.", inbox)
    try:
        while True:
            _scan(inbox, processed_dir, failed_dir, args.settle)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        LOG.info("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
