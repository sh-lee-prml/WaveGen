#!/usr/bin/env bash
set -euo pipefail

# Edit the adjacent TOML for local datasets and metric models first.
# Generate only:
#   bash src/wavegen/eval/eval_infer_batch_example.sh --infer-only
# Generate and score both CFG conditions:
#   bash src/wavegen/eval/eval_infer_batch_example.sh --full-eval
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/eval_infer_batch.sh" "$@"
