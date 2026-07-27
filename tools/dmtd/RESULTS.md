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

## Compilation optimizations (added later)

CUDA graphs and async scheduling are enabled for DMTD; `torch.compile` remains
off. See `tools/dmtd/bench_dmtd_ablation.py` and `tools/dmtd/bench_compare.py`.

Everything is captured under `CUDAGraphMode.FULL_DECODE_ONLY` with
`CompilationMode.NONE`, in two independent families.

The runner's own graph covers the sequential layers plus the assembly in front
of them. That part is shape-invariant: each scheduled token takes exactly one
row out of two persistent tables (the cycle buffer and the parallel-output
table) by index, whether or not the parallel layers ran this step. So one graph
serves cycle-head and mid-cycle decode steps alike, and
`DMTDQwen3ModelState.requires_eager_step` only vetoes replay while a request is
still prefilling.

The parallel layers run before that forward, under graphs the model state owns
(`capture_extra_graphs` / `DMTDParallelCudaGraphManager`), one family per group.
This works because a decode cycle head is a *uniform* batch: every participating
request contributes exactly `DECODE_ROWS_PER_REQ[group]` rows -- `tau` for the
Norefresh variants and Causal-Refresh's shadow block, `2*tau` for a
Causal-Refresh cycle, `tau + tau` split across two groups for
Bidirectional-Refresh. That is the same property DFlash gets from freezing
`num_query_per_req` at config time, so the existing `uniform_token_count`
dispatch dimension covers it and no shared descriptor change was needed. Graphs
are bucketed on the participating-request count and padded up with inert rows
(`PAD_SLOT_ID`, `seq_len == 0`, flat `query_start_loc`), following DFlash's
padding recipe.

Two details are load-bearing for correctness:

- Refresh emits its refresh block on every cycle so the shape stays uniform, but
  on the first cycle -- when there is nothing to overwrite -- those rows have
  their KV write suppressed. Re-running a position reproduces it only to within
  bf16 rounding (a different query length makes FlashAttention tile
  differently), and perturbing prefill's real KV moves later greedy tokens.
- Padded rows and mid-cycle cycle-buffer writes are aimed at scratch entries: an
  extra request slot in `cycle_hidden` and an extra row in the parallel-output
  table. The write vectors are a fixed length so a captured graph copies the
  same number of rows on every replay.

Fixed 512-token prompt, 256 output tokens, batch 1, one L40S:

| variant | eager | +sequential graphs | +parallel graphs | +async | total |
| --- | --- | --- | --- | --- | --- |
| Causal-Parallel-Norefresh | 84.36 | 107.97 | 136.99 | 166.60 | 1.97x |
| Causal-Parallel-Refresh | 81.29 | 106.70 | 136.88 | 166.17 | 2.04x |
| Bidirectional-Parallel-Norefresh | 78.99 | 102.56 | 136.32 | 164.18 | 2.08x |
| Bidirectional-Parallel-Refresh | 55.53 | 69.64 | 101.61 | 118.74 | 2.14x |

Against a Qwen3-4B baseline running everything it supports (including
`torch.compile`, which DMTD does not):

| model | tok/s | vs Qwen3-4B |
| --- | --- | --- |
| Qwen3-4B | 82.68 | 1.00x |
| Causal-Parallel-Norefresh | 166.59 | 2.01x |
| Causal-Parallel-Refresh | 165.83 | 2.01x |
| Bidirectional-Parallel-Norefresh | 164.37 | 1.99x |
| Bidirectional-Parallel-Refresh | 118.70 | 1.44x |

Greedy token ids are identical between eager, CUDA graph, and async scheduling
runs for all four variants (`tests/v1/dmtd/test_dmtd_e2e.py`:
`test_cudagraph_decode_matches_eager`, which also asserts the parallel-layer
graphs were actually replayed, and
`test_async_scheduling_matches_sync_scheduling`). They are also identical to the
pre-optimization engine: `tools/dmtd/smoke_generate.py --expect-json
tools/dmtd/smoke_backup.json` matches for all four variants in all three modes
(eager, graphed, graphed + async), and top-5 logprobs match to the bit.

The eager column above is itself faster than before this work (e.g.
Causal-Parallel-Norefresh 78.35 -> 84.36) because the sequential-layer input is
now assembled with index-based gathers instead of boolean masking, removing
about a dozen device synchronizations per step, and because each group's
attention inputs are now refilled in place instead of reallocated per step.

## Enabled and constrained features

Enabled and validated:

- native V2 model runner;
- standard sampler and direct token commit;
- FlashAttention/PagedAttention KV cache;
- shared block table with per-layer shadow/real ownership;
- chunked prefill;
- continuous batching;
- OpenAI-compatible serving.

- full CUDA graphs for every decode step, covering both the sequential layers
  and the parallel-layer groups;
- asynchronous scheduling.

Intentionally disabled until separately validated:

- prefix caching;
- speculative decoding;
- pipeline/context parallelism and dual batch overlap;
- KV connectors;
- quantized weight formats;
- `torch.compile` (measured at about +5% alone and +1.6% on top of CUDA graphs,
  so the lowest-value item on the list);
- CUDA graphs for prefill steps, and for the rare non-uniform decode steps: a
  multi-token decode, a Refresh request's very first cycle when its prompt is
  shorter than `tau`, and a mid-cycle resume after a schedule gap. Each falls
  back to eager for that one step.

Tensor parallel was not run because the host exposes one GPU. No custom kernel
was added; profiling should precede any kernel work.
