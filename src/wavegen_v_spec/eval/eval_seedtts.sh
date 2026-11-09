#!/usr/bin/env bash
set -euo pipefail

# Score one generated condition; pass the metadata and local metric model paths.
python -m wavegen_v_spec.eval.eval_seedtts_testset "$@"
