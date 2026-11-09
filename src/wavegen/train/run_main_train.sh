#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-WaveGen_SF_Emilia_24k_50Hz_IBDrop02}"
DATASET_PATH="${DATASET_PATH:-data/Emilia_ZH_EN_pinyin}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-19200}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MASTER_PORT="${MASTER_PORT:-29500}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

accelerate launch --num_processes "$NUM_PROCESSES" \
  --mixed_precision "$MIXED_PRECISION" --main_process_port "$MASTER_PORT" \
  --module wavegen.train.train --config-name "$CONFIG_NAME" \
  datasets.path="$DATASET_PATH" datasets.num_workers="$NUM_WORKERS" \
  datasets.batch_size_per_gpu="$BATCH_SIZE_PER_GPU" \
  optim.mixed_precision="$MIXED_PRECISION" "$@"
