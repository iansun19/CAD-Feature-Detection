"""Set process env before torch/OCC import (macOS OpenMP duplicate lib)."""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
