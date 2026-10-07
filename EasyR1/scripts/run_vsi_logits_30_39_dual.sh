#!/bin/bash
# rows 30–39: two logits sets (full_priv + answer_only), same settings as 20–29.
set -eo pipefail

EASYR1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${EASYR1_ROOT}"

GPD_ROOT="$(cd "${EASYR1_ROOT}/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-/PATH/TO/MODEL/Qwen/Qwen3-VL-4B-Instruct}"
export DATA_DIR="${DATA_DIR:-/PATH/TO/WORKSPACE/data}"
export PROJECT_OUT="${PROJECT_OUT:-${GPD_ROOT}/outputs}"

export ROW_INDEX="${ROW_INDEX:-30}"
export NUM_ROWS="${NUM_ROWS:-10}"
export SEED="${SEED:-42}"

export OUT_FULL="${OUT_FULL:-${PROJECT_OUT}/vsi_logits_10cases_full_priv_30_39}"
export OUT_ANSWER_ONLY="${OUT_ANSWER_ONLY:-${PROJECT_OUT}/vsi_logits_10cases_answer_only_30_39}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash scripts/run_vsi_logits_10cases_dual.sh

echo ""
echo "=== postprocess rows 30–39 (CSV + merged + paste + sentence-avg + auto inline notes) ==="
CONDA_ENV="${CONDA_ENV:-gpd_rl}"
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -x "${CONDA_BASE}/envs/${CONDA_ENV}/bin/python" ]]; then
  PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
else
  PYTHON="$(command -v python3)"
fi

"${PYTHON}" scripts/vsi_logits_postprocess_batch.py \
  --cases-dir-ao "${OUT_ANSWER_ONLY}" \
  --cases-dir-fp "${OUT_FULL}"

echo "Done. Outputs:"
echo "  ${OUT_ANSWER_ONLY}"
echo "  ${OUT_FULL}"
