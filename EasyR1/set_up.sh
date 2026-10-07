#!/usr/bin/env bash
# Use python -m pip so it works under conda even when `pip` is not on PATH.
set -euo pipefail

PY="${PYTHON:-python3}"
if ! command -v "$PY" &>/dev/null; then
  echo "Error: $PY not found. Run: conda activate gpd_rl (or your environment) first." >&2
  exit 1
fi

"$PY" -m pip install -r requirements_b200.txt
"$PY" -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
"$PY" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install flash-attn==2.8.3 --no-build-isolation
"$PY" -m pip install -e .
"$PY" -m pip install aiofiles
"$PY" -m pip install transformers==4.57.3
