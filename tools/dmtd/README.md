# DMTD validation

Run these commands inside the `ms-swift` GPU container:

```bash
cd /workspace/vllm
tools/dmtd/setup.sh
tools/dmtd/unit.sh
tools/dmtd/real_parity.sh
tools/dmtd/serve_smoke.sh
tools/dmtd/benchmark.sh
```

`setup.sh` must report a `vllm.__file__` under `/workspace/vllm`. Set
`DMTD_BUILD_FROM_SOURCE=1` if the image's precompiled extension is incompatible
with the local checkout.

Point `DMTD_MODEL_PATH` at any DMTD checkpoint:

- `/workspace/parallel-eval/models/Causal-Parallel-Norefresh` (default; shadow history)
- `/workspace/parallel-eval/models/Causal-Parallel-Refresh` (real-KV refresh)
- `/workspace/parallel-eval/models/Bidirectional-Parallel-Norefresh`
  (shadow history, bidirectional current shadow cycle)
- `/workspace/parallel-eval/models/Bidirectional-Parallel-Refresh`
  (real-KV refresh, bidirectional current shadow cycle)

Example:

```bash
DMTD_MODEL_PATH=/workspace/parallel-eval/models/Causal-Parallel-Refresh \
  tools/dmtd/real_parity.sh
```

The correctness oracle intentionally performs full-sequence Transformers
forward passes with `use_cache=False`. Bidirectional variants pad the current
cycle with MASK IDs before reading the final real position. Refresh adds an
explicit cycle oracle under `tests/v1/dmtd/oracle.py` (`RefreshCycleOracle`) for
cache/visibility control-flow. These checks are slow relative to native decode.

See `RESULTS.md` for the recorded environment, test matrix, BF16 parity
boundary, cache invariants, measured throughput and unsupported features.
