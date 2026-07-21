#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/workspace/vllm}"
cd "${VLLM_ROOT}"
export VLLM_USE_MODELSCOPE=False

PYTHON="${VLLM_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run tools/dmtd/setup.sh first." >&2
  exit 2
fi

"${PYTHON}" -m pytest \
  tests/v1/dmtd/test_oracle.py \
  tests/v1/dmtd/test_refresh_oracle.py \
  tests/v1/dmtd/test_dmtd_qwen3.py \
  tests/v1/dmtd/test_dmtd_scheduler.py \
  -v
