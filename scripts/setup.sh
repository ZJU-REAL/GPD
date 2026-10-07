#!/bin/bash
# GPD one-shot environment install (pip install -e . inside EasyR1)
set -eo pipefail

GPD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EASYR1_ROOT="${GPD_ROOT}/EasyR1"

echo "GPD root: ${GPD_ROOT}"
echo "Installing EasyR1 (verl) into the current Python environment ..."
echo "Tip: conda activate gpd_rl   # or set PYTHON=/path/to/python"

cd "${EASYR1_ROOT}"
if [[ -x "${EASYR1_ROOT}/set_up.sh" ]]; then
  bash set_up.sh
else
  python3 -m pip install -r requirements_b200.txt
  python3 -m pip install -e .
fi

mkdir -p "${GPD_ROOT}/outputs"

echo ""
echo "Setup complete."
echo "Next:"
echo "  export MODEL_PATH=/PATH/TO/MODEL/Qwen3-VL-4B-Instruct   # or 2B"
echo "  cd ${EASYR1_ROOT}"
echo "  bash examples/bash_4b/gpd.sh"
echo "  # bash examples/bash_2b/gpd.sh"
echo "See ${GPD_ROOT}/README.md for data paths and paper comparisons."
