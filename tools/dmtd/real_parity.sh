#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/workspace/vllm}"
# Default Norefresh; set DMTD_MODEL_PATH to Causal-Parallel-Refresh for the
# real-history Refresh variant, e.g.:
#   DMTD_MODEL_PATH=/workspace/parallel-eval/models/Causal-Parallel-Refresh \
#     tools/dmtd/real_parity.sh
MODEL_PATH="${DMTD_MODEL_PATH:-/workspace/parallel-eval/models/Causal-Parallel-Norefresh}"
cd "${VLLM_ROOT}"
export VLLM_USE_MODELSCOPE=False
export VLLM_ENABLE_V1_MULTIPROCESSING=0

PYTHON="${VLLM_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${PYTHON}; run tools/dmtd/setup.sh first." >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
  echo "DMTD checkpoint not found at ${MODEL_PATH}" >&2
  exit 2
fi

echo "DMTD real parity against: ${MODEL_PATH}"

DMTD_MODEL_PATH="${MODEL_PATH}" \
VLLM_USE_V2_MODEL_RUNNER=1 \
"${PYTHON}" -m pytest tests/v1/dmtd/test_dmtd_e2e.py -v -s
