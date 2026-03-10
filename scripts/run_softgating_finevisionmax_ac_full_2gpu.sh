#!/usr/bin/env bash
set -euo pipefail

# 2-GPU AC-full training launcher for nanoVLM soft-gating finevisionmax config.
# Defaults replicate the requested "batch size 32 and GA 2" setup:
# global_batch_size = local_batch_size * nproc_per_node * grad_accum_steps

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTITAN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-/home/coder/edd/nanoVLM_root/nanoVLM_main/.venv/bin/activate}"

CONFIG_NAME="${CONFIG_NAME:-nanovlm_230m_momh_soft_gating_b5_tttv_nopack}"
STEPS="${STEPS:-100}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
GLOBAL_BATCH_SIZE=$(( LOCAL_BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM_STEPS ))
COMM_INIT_TIMEOUT_SECONDS="${COMM_INIT_TIMEOUT_SECONDS:-1800}"
COMM_TRAIN_TIMEOUT_SECONDS="${COMM_TRAIN_TIMEOUT_SECONDS:-300}"

source "${VENV_ACTIVATE}"
cd "${TORCHTITAN_ROOT}"

echo "[run-softgating-2gpu] config=${CONFIG_NAME}"
echo "[run-softgating-2gpu] steps=${STEPS}"
echo "[run-softgating-2gpu] nproc_per_node=${NPROC_PER_NODE}"
echo "[run-softgating-2gpu] local_batch_size=${LOCAL_BATCH_SIZE}"
echo "[run-softgating-2gpu] grad_accum_steps=${GRAD_ACCUM_STEPS}"
echo "[run-softgating-2gpu] global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "[run-softgating-2gpu] comm_init_timeout_seconds=${COMM_INIT_TIMEOUT_SECONDS}"
echo "[run-softgating-2gpu] comm_train_timeout_seconds=${COMM_TRAIN_TIMEOUT_SECONDS}"

torchrun --standalone --max-restarts=0 --nproc_per_node="${NPROC_PER_NODE}" -m torchtitan.train \
  --module nanoVLM \
  --config "${CONFIG_NAME}" \
  --training.steps "${STEPS}" \
  --training.global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --training.local-batch-size "${LOCAL_BATCH_SIZE}" \
  --comm.init-timeout-seconds "${COMM_INIT_TIMEOUT_SECONDS}" \
  --comm.train-timeout-seconds "${COMM_TRAIN_TIMEOUT_SECONDS}" \
  --activation-checkpoint.mode full \
  --metrics.log_freq 1 \
  --metrics.no-enable-wandb \
  --checkpoint.no-enable
