#!/usr/bin/env bash
# Create the unified mlcad conda env (torch + pythonocc-core + PyG).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Mambaforge, then re-run this script." >&2
  exit 1
fi

ENV_NAME="${MLCAD_ENV_NAME:-mlcad}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing env: $ENV_NAME"
  conda env update -f environment.yml --name "$ENV_NAME" --prune
else
  echo "Creating env: $ENV_NAME"
  conda env create -f environment.yml --name "$ENV_NAME"
fi

echo ""
echo "Verifying imports in $ENV_NAME …"
KMP_DUPLICATE_LIB_OK=TRUE conda run -n "$ENV_NAME" python - <<'PY'
import torch
from OCC.Core.STEPControl import STEPControl_Reader
import torch_geometric

print("torch", torch.__version__)
print("pythonocc OK")
print("torch_geometric", torch_geometric.__version__)
if hasattr(torch.backends, "mps"):
    print("mps", torch.backends.mps.is_available())
print("cuda", torch.cuda.is_available())
PY

echo ""
echo "Done. Activate with:  conda activate $ENV_NAME"
echo "Then run:             python run_pipeline.py --step path/to/part.step"
