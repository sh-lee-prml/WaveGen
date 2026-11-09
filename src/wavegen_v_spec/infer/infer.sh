#!/usr/bin/env bash
set -euo pipefail
# Install this standalone project first: python -m pip install -e .
python -m wavegen_v_spec.infer.infer_cli "$@"
