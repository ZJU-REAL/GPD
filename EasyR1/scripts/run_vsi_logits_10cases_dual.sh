#!/bin/bash
# Offline dump of two vsi logits sets (10 rows), aligned with the asymmetric teacher privilege used during training:
#
#   1) vsi_logits_10cases_full_priv
#        data:    vsi_10k_train.parquet  (priv = <scene_context> + <reference_answer>)
#        teacher: TEACHER_SYSTEM (prompt explicitly allows 3D + answer); no --teacher-answer-only
#
#   2) vsi_logits_10cases_answer_only
#        data:    vsi_10k_answeronly_train.parquet (priv = <reference_answer> only)
#        teacher: TEACHER_SYSTEM_ANSWER_ONLY; must use --teacher-answer-only
#
# Student side is the same for both runs: multi-frame images + Question only, no priv.
#
# Usage (from EasyR1 root):
#   bash scripts/run_vsi_logits_10cases_dual.sh
# Or with specific GPU:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_vsi_logits_10cases_dual.sh

set -eo pipefail

EASYR1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${EASYR1_ROOT}"

# GPD package root (parent of EasyR1)
GPD_ROOT="$(cd "${EASYR1_ROOT}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/PATH/TO/MODEL/Qwen/Qwen3-VL-4B-Instruct}"
DATA_DIR="${DATA_DIR:-/PATH/TO/WORKSPACE/data}"
PROJECT_OUT="${PROJECT_OUT:-${GPD_ROOT}/outputs}"

PARQUET_FULL="${PARQUET_FULL:-${DATA_DIR}/vsi_10k_train.parquet}"
PARQUET_ANSWER_ONLY="${PARQUET_ANSWER_ONLY:-${DATA_DIR}/vsi_10k_answeronly_train.parquet}"

for f in "${PARQUET_FULL}" "${PARQUET_ANSWER_ONLY}"; do
  if [[ ! -f "${f}" ]]; then
    echo "parquet not found: ${f}" >&2
    echo "Set DATA_DIR, PARQUET_FULL, or PARQUET_ANSWER_ONLY" >&2
    exit 1
  fi
done

OUT_FULL="${OUT_FULL:-${PROJECT_OUT}/vsi_logits_10cases_full_priv}"
OUT_ANSWER_ONLY="${OUT_ANSWER_ONLY:-${PROJECT_OUT}/vsi_logits_10cases_answer_only}"

ROW_INDEX="${ROW_INDEX:-0}"
NUM_ROWS="${NUM_ROWS:-10}"
SEED="${SEED:-42}"

CONDA_ENV="${CONDA_ENV:-gpd_rl}"
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -x "${CONDA_BASE}/envs/${CONDA_ENV}/bin/python" ]]; then
  PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
else
  PYTHON="$(command -v python3 2>/dev/null || command -v python)"
fi

ANALYZE="${EASYR1_ROOT}/scripts/analyze_rollout_student_teacher_logits.py"
COMMON=(
  "${PYTHON}" "${ANALYZE}"
  --model-path "${MODEL_PATH}"
  --row-index "${ROW_INDEX}"
  --num-rows "${NUM_ROWS}"
  --seed "${SEED}"
)

echo "=== [1/2] full priv: 3D+answer data + TEACHER_SYSTEM (no --teacher-answer-only) ==="
echo "  parquet: ${PARQUET_FULL}"
echo "  out:     ${OUT_FULL}"
"${COMMON[@]}" \
  --parquet "${PARQUET_FULL}" \
  --out-dir "${OUT_FULL}"

echo ""
echo "=== [2/2] answer-only priv: answeronly parquet + --teacher-answer-only ==="
echo "  parquet: ${PARQUET_ANSWER_ONLY}"
echo "  out:     ${OUT_ANSWER_ONLY}"
"${COMMON[@]}" \
  --parquet "${PARQUET_ANSWER_ONLY}" \
  --teacher-answer-only \
  --out-dir "${OUT_ANSWER_ONLY}"

echo ""
echo "Done. Plot examples:"
echo "  python scripts/plot_vsi_logits_priv_kl_per_token.py --cases-dir ${OUT_FULL} --one-panel-per-file --write-csv"
echo "  python scripts/plot_vsi_logits_priv_kl_per_token.py --cases-dir ${OUT_ANSWER_ONLY} --one-panel-per-file --write-csv"
