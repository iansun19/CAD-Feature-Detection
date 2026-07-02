"""In-process STEP → graph tensor ingest (requires pythonocc in the active env)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_params import HAS_OCC


def require_occ(action: str = "STEP parsing") -> None:
    if not HAS_OCC:
        raise RuntimeError(
            f"{action} requires pythonocc-core in this Python environment. "
            "Install the unified env: conda env create -f environment.yml && conda activate mlcad"
        )


def ingest_step_to_npz(
    step_path: Path,
    npz_path: Path,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse STEP in-process and write graph.npz. Returns (x, edge_index, edge_attr)."""
    require_occ()
    from step_ingest import ingest_step_to_pyg

    x, edge_index, edge_attr, stats = ingest_step_to_pyg(
        str(step_path),
        num_surface_types=cfg["num_surface_types"],
        angle_reduce=cfg.get("angle_reduce", "median"),
        include_curvature=cfg.get("include_curvature", False),
    )
    if not stats.success:
        raise RuntimeError(f"STEP ingest failed: {stats.error}")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        part_id=np.array(step_path.stem),
        step_path=np.array(str(step_path.resolve())),
    )
    return x, edge_index, edge_attr
