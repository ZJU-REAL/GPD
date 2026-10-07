#!/bin/bash
# Shared env for GPD / Mixed-15k experiments (scale=2B).
# Source after EASYR1_ROOT is set.

if [[ -z "${GPD_ROOT:-}" ]]; then
  GPD_ROOT="$(cd "${EASYR1_ROOT}/.." && pwd)"
fi
export GPD_ROOT

FRAMES_DIR="${FRAMES_DIR:-${GPD_ROOT}/data/frames}"
DATA_DIR_PRIVFIX="${DATA_DIR_PRIVFIX:-${GPD_ROOT}/data/mixed_15k_privfix}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${GPD_ROOT}/outputs}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
CONDA_ENV="${CONDA_ENV:-gpd_rl}"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -x "${CONDA_BASE}/envs/${CONDA_ENV}/bin/python" ]]; then
  PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
elif [[ -n "${CONDA_PREFIX}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found for env ${CONDA_ENV}. Run: conda activate ${CONDA_ENV}" >&2
  exit 1
fi

export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export RAY_DISABLE_MEMORY_MONITOR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPD_DATA_IMAGE_DIR="data.image_dir=${FRAMES_DIR}"
