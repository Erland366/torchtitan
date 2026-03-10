#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/home/coder/.cache/huggingface}"

TASKS="${TASKS:-coco2017_cap_val,vqav2_val,ocrbench,scienceqa,docvqa_val}"
LIMIT="${LIMIT:-2000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
CKPT_PATH="${CKPT_PATH:-}"
CKPT_FORMAT="${CKPT_FORMAT:-auto}"
MODEL_BACKEND="${MODEL_BACKEND:-huggingface}"
FALLBACK_BACKEND="${FALLBACK_BACKEND:-torchtitan_plugin}"
OUT_DIR="${OUT_DIR:-eval_results/torchtitan}"
RUN_NAME="${RUN_NAME:-torchtitan-eval-$(date +%Y%m%d-%H%M%S)}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Set CKPT_PATH to a checkpoint folder (HF or DCP)." >&2
  exit 1
fi

python scripts/nanovlm_downstream_eval.py \
  --checkpoint_path "${CKPT_PATH}" \
  --checkpoint_format "${CKPT_FORMAT}" \
  --tasks "${TASKS}" \
  --limit "${LIMIT}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --model_backend "${MODEL_BACKEND}" \
  --fallback_backend "${FALLBACK_BACKEND}" \
  --output_dir "${OUT_DIR}" \
  --run_name "${RUN_NAME}"
