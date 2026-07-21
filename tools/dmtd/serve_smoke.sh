#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/workspace/vllm}"
MODEL_PATH="${DMTD_MODEL_PATH:-/workspace/parallel-eval/models/Causal-Parallel-Norefresh}"
PORT="${DMTD_PORT:-8000}"
LOG_FILE="${DMTD_SERVER_LOG:-/tmp/dmtd-vllm.log}"
cd "${VLLM_ROOT}"
export VLLM_USE_MODELSCOPE=False

VLLM_BIN="${VLLM_ROOT}/.venv/bin/vllm"
if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "Missing ${VLLM_BIN}; run tools/dmtd/setup.sh first." >&2
  exit 2
fi

VLLM_USE_V2_MODEL_RUNNER=1 "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name dmtd-qwen3-4b \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len 2048 \
  --no-enable-prefix-caching \
  --kv-cache-memory-bytes 4294967296 \
  --tensor-parallel-size 1 \
  --max-num-seqs 8 \
  --port "${PORT}" >"${LOG_FILE}" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    cat "${LOG_FILE}" >&2
    exit 1
  fi
  sleep 2
done

curl -sf "http://127.0.0.1:${PORT}/v1/models"
curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dmtd-qwen3-4b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "temperature": 0,
    "max_tokens": 8
  }'
