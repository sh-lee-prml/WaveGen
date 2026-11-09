#!/usr/bin/env bash
set -euo pipefail

# Run from the WaveGen repository root; CLI options override the example TOML.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -m wavegen_v_spec.eval.eval_infer_batch \
  --eval-config "${SCRIPT_DIR}/eval_infer_batch.toml" "$@"
