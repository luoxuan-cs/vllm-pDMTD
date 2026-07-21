#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/workspace/vllm}"
cd "${VLLM_ROOT}"
export VLLM_USE_MODELSCOPE=False

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv venv --python 3.12 --allow-existing .venv

if [[ "${DMTD_BUILD_FROM_SOURCE:-0}" == "1" ]]; then
  uv pip install -e . --torch-backend=auto
else
  VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
fi

uv pip install -r requirements/lint.txt
uv pip install -r requirements/test/cuda.in

.venv/bin/python - <<'PY'
from pathlib import Path

import torch
import transformers
import vllm

root = Path("/workspace/vllm").resolve()
loaded = Path(vllm.__file__).resolve()
if root not in loaded.parents:
    raise SystemExit(f"Expected local vLLM under {root}, loaded {loaded}")
print(f"vLLM: {vllm.__version__} ({loaded})")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY
