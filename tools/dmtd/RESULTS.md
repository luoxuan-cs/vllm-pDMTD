# DMTD validation record

Date: 2026-07-20

## Baseline

- vLLM source: `/workspace/vllm`
- vLLM commit: `58b2012aa26f2b85560ecd6988c8fa2a773804c1`
- runtime version:
  `0.23.1rc1.dev1311+g58b2012aa.d20260721`
- checkpoints (select with `DMTD_MODEL_PATH`):
  - `/workspace/parallel-eval/models/Causal-Parallel-Norefresh` (default; `shadow+causal`)
  - `/workspace/parallel-eval/models/Causal-Parallel-Refresh` (`real+causal`)
  - `/workspace/parallel-eval/models/Bidirectional-Parallel-Norefresh`
    (`shadow+bidirectional`)
  - `/workspace/parallel-eval/models/Bidirectional-Parallel-Refresh`
    (`real+bidirectional`)
- checkpoint size: 4,022,468,096 parameters / 8,044,936,192 bytes
- GPU: one NVIDIA L40S, 46,068 MiB
- PyTorch: `2.11.0+cu130`
- Transformers: `5.13.1`

The editable installation resolved `vllm.__file__` to
`/workspace/vllm/vllm/__init__.py`.

## Correctness results

- DMTD oracle/model/cache/scheduler unit suite: 50 passed
  (Norefresh + Refresh oracle, Refresh cycle planner shapes, cycle-aligned
  scheduler clip, bidirectional masks and complete-cycle planner expansion).
- Registry import and architecture coverage: 2 passed.
- Real 4B GPU matrix (both checkpoints via `DMTD_MODEL_PATH`):
  - Causal-Parallel-Norefresh: 5 passed (exact short tokens; longer ULP≤0.125).
  - Causal-Parallel-Refresh: 5 passed (same matrix; short-sequence BF16 ties
    under continuous batching accepted via one-ULP teacher-forced check).
  - prompt lengths 1, 3, 4 and 5;
  - 2, 5 and 9 generated tokens;
  - two continuously batched requests at different cycle phases;
  - a 17-token prompt chunked at 8 tokens across the 16-token block boundary.
- Refresh defaults to merged8 causal parallel-layer forward
  (`DMTD_REFRESH_BACKEND=merged`); `two_pass` remains available. Parallel-layer
  FA runs per-request inside the batch to keep independent shadow blocks
  correct; the sequential layers stay batched.
- Dummy-weight V2 initialization and generation: passed.
- OpenAI-compatible `/v1/models` and `/v1/chat/completions`: passed.
- Both bidirectional 4B checkpoints: offline multi-request generation and
  OpenAI-compatible serve smoke passed with native FlexAttention.
- Ruff check/format on changed Python files: passed.
- ShellCheck on the DMTD entry scripts: passed.

The two-token real-weight rollout is exactly token-identical to the
full-recompute Transformers oracle. Longer BF16 rollouts encounter logits on
ties or one-ULP boundaries: the persistent PagedAttention execution can choose
a different tied token than full recomputation. Every selected token was
teacher-forced through the full-recompute model and was at most one BF16 ULP
(`0.125` at the observed logit scale) below its maximum. The tests deliberately
report this instead of claiming false bitwise argmax equivalence across
different attention execution shapes.

## Cache invariants exercised

- Scheduler reserves exactly three lookahead slots without a speculative
  configuration or draft-token bookkeeping.
- Norefresh: layers 0–27 persist shadow KV; Refresh: layers 0–27 persist only
  refreshed real KV (one-cycle lag), with temporary physical shadow slots.
- Layers 28–35 write only the current real-token KV.
- Refresh prefill never crosses a cycle boundary in one schedule step.
- Steady Refresh cycle uses merged8
  `[real(c-4:c) | head, M, M, M]`; first cycle is shadow4 only.
- A cycle head constructs `[real, MASK, MASK, MASK]` once and reuses its four
  hidden states for the sequential layers; refresh hiddens never enter them.
- Bidirectional Norefresh expands every touched cycle to a complete shadow
  block; suffix-only recomputation is invalid because all four hidden states
  depend on all four shadow slots.
- Bidirectional Refresh keeps refresh/real queries causal while only the
  current shadow cycle is fully connected, including in merged8.
- Request removal discards the cycle buffer; buffer loss rebuilds the full
  current shadow block (Refresh does not suffix-only recompute).
- No rejection sampler, rollback, bonus token or acceptance-rate path is used.
- Bidirectional variants require and automatically select FlexAttention;
  explicitly selecting another backend fails closed.

## Bidirectional numerical note

The bidirectional checkpoints are numerically ill-conditioned under BF16:
one-ULP-level parallel-layer differences between the reference's dense
FlexAttention and vLLM's paged FlexAttention are strongly amplified around
parallel layer 6. Their
greedy IDs therefore do not satisfy the causal checkpoints' one-ULP
teacher-forced parity criterion, especially under continuous batching. The
visibility matrices, cycle/cache invariants, complete-cycle oracle, real-weight
offline generation, and serving paths pass; exact greedy parity is not claimed.

## Performance

The included four-prompt, 32-output-token-per-prompt benchmark measured:

- full-recompute Transformers oracle: 35.10 tokens/s;
- native vLLM eager path: 377.59 tokens/s;
- measured speedup: 10.76x.

This is a correctness-baseline comparison against the intentionally slow
oracle, not a claim against a standard cached Transformers baseline.

## Enabled and constrained features

Enabled and validated:

- native V2 model runner;
- standard sampler and direct token commit;
- FlashAttention/PagedAttention KV cache;
- shared block table with per-layer shadow/real ownership;
- chunked prefill;
- continuous batching;
- OpenAI-compatible serving.

Intentionally disabled until separately validated:

- prefix caching;
- speculative decoding;
- asynchronous scheduling;
- pipeline/context parallelism and dual batch overlap;
- KV connectors;
- quantized weight formats;
- CUDA Graph/`torch.compile`.

Tensor parallel was not run because the host exposes one GPU. No custom kernel
was added; profiling should precede any kernel work.
