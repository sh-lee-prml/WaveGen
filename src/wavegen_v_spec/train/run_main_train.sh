#!/usr/bin/env bash
set -euo pipefail
accelerate launch --num_processes 8 --mixed_precision bf16 -m wavegen_v_spec.train.train "$@"
