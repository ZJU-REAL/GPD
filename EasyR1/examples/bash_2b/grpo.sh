#!/bin/bash
# =============================================================================
# Qwen3-VL-2B — Pure GRPO baseline (no teacher privilege)
# Paper method name: GPD (Geometry-Privileged Distillation)
# Data: data/mixed_15k_privfix/pure_grpo_{train,val}.parquet
# =============================================================================
#
# Usage:
#   cd EasyR1   # from the repository root
#   export MODEL_PATH=/PATH/TO/MODEL/Qwen3-VL-2B-Instruct
#   bash examples/bash_2b/grpo.sh
# =============================================================================

set -eo pipefail

EASYR1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${EASYR1_ROOT}"

source "$(dirname "${BASH_SOURCE[0]}")/gpd_env.sh"

DATA_DIR="${DATA_DIR:-${DATA_DIR_PRIVFIX}}"
TRAIN_FILES="${TRAIN_FILES:-${DATA_DIR}/pure_grpo_train.parquet}"
VAL_FILES="${VAL_FILES:-${DATA_DIR}/pure_grpo_val.parquet}"
SAVE_PATH="${SAVE_PATH:-${OUTPUTS_DIR}/checkpoint_2b_grpo}"

LEN_TARGET=${LEN_TARGET:-80}
LEN_PEN=${LEN_PEN:-0}
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
MICRO_UPDATE="${MICRO_UPDATE:-4}"
MICRO_EXP="${MICRO_EXP:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

RUN_LOG_DIR="${SAVE_PATH}/run_logs"
mkdir -p "${RUN_LOG_DIR}"
RUN_LOG="${RUN_LOG_DIR}/grpo_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== 2B | grpo | MODEL_PATH=${MODEL_PATH} | SAVE_PATH=${SAVE_PATH} ==="
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} | GPU_MEM_UTIL=${GPU_MEM_UTIL}"
  echo "PYTHON=${PYTHON} ($(date -Is))"
  "${PYTHON}" -V
  "${PYTHON}" -m verl.trainer.main \
    config=examples/gpd_config.yaml \
    data.max_response_length=1024 \
    data.mini_rollout_batch_size=4 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    ${GPD_DATA_IMAGE_DIR} \
    algorithm.privileged_rl=false \
    algorithm.use_frozen_base_kl=false \
    algorithm.frozen_base_kl_coef=0 \
    algorithm.shared_ref_student_weights=false \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=true \
    algorithm.kl_coef=0.01 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    worker.actor.micro_batch_size_per_device_for_update=${MICRO_UPDATE} \
    worker.actor.micro_batch_size_per_device_for_experience=${MICRO_EXP} \
    worker.actor.use_torch_compile=true \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.save_freq=30 \
    trainer.experiment_name=2b_grpo \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=4 \
    "$@"
} 2>&1 | tee "${RUN_LOG}"
