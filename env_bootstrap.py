"""Set process env before torch/OCC import (macOS OpenMP duplicate lib)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
_DOTENV_LOADED = False


def load_repo_dotenv() -> None:
    """Load repo-root ``.env`` into os.environ (existing vars are not overwritten)."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    path = REPO_ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            if not key:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)

    _DOTENV_LOADED = True


load_repo_dotenv()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
