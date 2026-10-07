#!/bin/bash
# =============================================================================
# Qwen3-VL-2B — Answer-privileged OPSD (pure KL distillation, no GRPO PG)
# Paper method name: GPD (Geometry-Privileged Distillation)
# Data: data/mixed_15k_privfix/answer_only_{train,val}.parquet
# =============================================================================
#
# Usage:
#   cd EasyR1   # from the repository root
#   export MODEL_PATH=/PATH/TO/MODEL/Qwen3-VL-2B-Instruct
#   bash examples/bash_2b/opsd.sh
# =============================================================================

set -eo pipefail

EASYR1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${EASYR1_ROOT}"

source "$(dirname "${BASH_SOURCE[0]}")/gpd_env.sh"

DATA_DIR="${DATA_DIR:-${DATA_DIR_PRIVFIX}}"
TRAIN_FILES="${TRAIN_FILES:-${DATA_DIR}/answer_only_train.parquet}"
VAL_FILES="${VAL_FILES:-${DATA_DIR}/answer_only_val.parquet}"
SAVE_PATH="${SAVE_PATH:-${OUTPUTS_DIR}/checkpoint_2b_opsd}"

LEN_TARGET=${LEN_TARGET:-80}
LEN_PEN=${LEN_PEN:-0}
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
MICRO_UPDATE="${MICRO_UPDATE:-4}"
MICRO_EXP="${MICRO_EXP:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

RUN_LOG_DIR="${SAVE_PATH}/run_logs"
mkdir -p "${RUN_LOG_DIR}"
RUN_LOG="${RUN_LOG_DIR}/opsd_$(date +%Y%m%d_%H%M%S).log"

{
  echo "=== 2B | opsd | MODEL_PATH=${MODEL_PATH} | SAVE_PATH=${SAVE_PATH} ==="
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
    algorithm.privileged_rl=true \
    worker.actor.global_batch_size=128 \
    algorithm.privileged_kl_full_vocab_chunk_size=${FULL_VOCAB_CHUNK:-128} \
    algorithm.opsd_privileged_kl_only_no_grpo=true \
    algorithm.shared_ref_student_weights=false \
    algorithm.opsd_jsd_token_clip=${OPSD_JSD_TOKEN_CLIP:-0.05} \
    algorithm.opsd_rollout_top_k=${OPSD_ROLLOUT_TOP_K:-20} \
    algorithm.split_grpo_pg_opd_kl=false \
    algorithm.use_frozen_base_kl=false \
    algorithm.entropy_penalty_coef=0 \
    algorithm.privileged_kl_sdar_sigmoid_gate=false \
    algorithm.kl_teacher_confidence_weight_beta=0 \
    algorithm.kl_teacher_disconfidence_weight_gamma=0 \
    algorithm.priv_kl_short_threshold=0 \
    worker.reward.reward_function_kwargs.length_target=0 \
    worker.reward.reward_function_kwargs.length_pen=0.0 \
    worker.actor.optim.lr=${LR:-5.0e-6} \
    worker.actor.max_grad_norm=0.1 \
    worker.actor.dynamic_batching=false \
    worker.rollout.temperature=1.1 \
    worker.rollout.top_k=${OPSD_ROLLOUT_TOP_K:-20} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    worker.actor.micro_batch_size_per_device_for_update=${MICRO_UPDATE} \
    worker.actor.micro_batch_size_per_device_for_experience=${MICRO_EXP} \
    worker.actor.use_torch_compile=true \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.save_freq=30 \
    trainer.experiment_name=2b_opsd \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=4 \
    "$@"
} 2>&1 | tee "${RUN_LOG}"
