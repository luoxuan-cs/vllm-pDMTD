#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/workspace/vllm}"
cd "${VLLM_ROOT}"
export VLLM_USE_MODELSCOPE=False
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${VLLM_ROOT}/.venv/bin/python" tools/dmtd/benchmark.py
