# Short-window attention experiment

`src/cuda/Attention.cuh` implements an optional fused attention path selected by
`--attention fused`. Staged attention remains the default because it is faster
on the tested Main configurations. Both paths use the same model, weights,
absolute positions, ring cache and numerical policy.

This is original project code informed by the pinned
[FlashInfer decode implementation](https://github.com/flashinfer-ai/flashinfer/blob/07cc0d60a9ba1aca87b7b803ab1810ca5e3738d6/include/flashinfer/attention/decode.cuh).
No FlashInfer package or upstream kernel has been built or installed. Its
online-softmax and approximate exp2 paths have not been copied. Existing
attributed NVIDIA warp reductions remain shared primitives.

## Computation and ownership

The fused path supports the predefined head dimensions 16 and 32 and windows
up to 512. Q/K arrive already rotated. Each block has 128 threads and owns an
entire head; conventional/graph execution launches one block per head.
Persistent execution assigns heads to its resident blocks; one-block execution
visits heads serially. Idle blocks still join the surrounding cooperative-grid
barriers in the forward pass.

For a position `t`, valid history is `min(t + 1, window)`. The first physical
slot is `(t + 1 - valid) % window`. Each later chronological slot is formed by
addition and at most one subtraction, since `firstSlot + j < 2 * window`.
This avoids repeated 64-bit modulo in the inner loops. Layer/head offsets
remain part of every address; the routine reads the cache without modifying it.

| Phase | Work assignment and synchronization |
|---|---|
| Stage K | Threads cooperatively load up to 128 rows with float2/half2 reads into a shared `[128][33]` tile. Padding avoids the row-stride bank conflicts of an unpadded 32-wide tile. |
| Score | After a block barrier, each active thread computes one QK row dot in FP32. A barrier protects the tile before reuse. |
| Normalize | Warp zero computes stable max subtraction, `expf`, denominator and explicitly normalized FP32 probabilities. A block barrier publishes them. |
| Stage/reduce V | Reuse the tile for V, with barriers between loaders and consumers. Four warps partition chronological positions; each lane accumulates one output coordinate. |
| Publish | Warp zero combines four FP32 partial sums, rounds the output to FP16 in mixed mode, then a barrier protects shared scratch before the next head. |

The routine uses 19,584 shared bytes per block, including the padded tile,
query, probabilities and partial sums. The tested persistent kernel reports
78 registers/thread and 32 local bytes. Actual occupancy permits 440 cooperative
blocks on this device; the tested 110-block grid is legal. The standalone
attention kernel reports 50 registers/thread and zero local bytes. These are
resource attributes, not measured occupancy or bandwidth.

Host orchestration replaces stages 6–8 with one kernel. The complete fused and
persistent forward paths replace their three barriers with one after the
combined operation. Main therefore changes from 38 to 34 numerical kernel
launches per conventional graph step. Diagnostic tracing uses the selected
attention path and retains its `layers.N.attention` field.

## Precision and validation

All dots, exponentials and reductions remain FP32, with `--fmad=false` and no
fast-math change. Mixed K/V are physically FP16, converted on load. Query values
retain their earlier cast boundaries; probabilities do not round to FP16.
Attention output rounds at the existing boundary. Tiled QK and V reductions
change FP32 accumulation order; numerical acceptance is checked with the
existing tolerances rather than assuming bitwise agreement with staged mode.

The current implementation passed:

- Independent FP64 primitive comparisons for both head dimensions and cache
  precisions; windows 1/7/32/64/256/512; partial, full and wrapped histories;
  positions through 1,000,000; grids 1/4/110. Unused slots/layers are poisoned and
  output guards checked.
- Memcheck, synccheck and racecheck on the registered primitive test: zero
  errors/hazards (`results/tiled-attention-{tool}.log`).
- Sixteen full-model fixture cases: all sizes and both modes for graph110 and
  persistent110, plus Main graph4 and persistent1. Scores, hidden state and K/V
  pass the existing policy (`results/tiled-attention/summary.json`).
- Exact diagnostic-versus-normal execution for staged and fused Main mixed
  mode, including the attention field, score and full final state
  (`results/tiled-attention-ctest-final.log`).
- Native constructor checks reject an unsupported 64-wide head and a 513-slot
  window when fused attention is selected.
- Sixteen fresh-process loading regressions across LAZY/EAGER, staged/fused,
  FP32/mixed and one-/110-block persistence (`results/tiled-attention-cold-loading`).

Early test-build and sanitizer attempts that omitted the new registered test
are retained and do not count as validation. Registered-test listings and the
verified logs identify the actual checks above.

## Cold-start fix

The first persistent experiment exited on its idle budget. Bounded diagnostics
failed with LAZY loading and passed with EAGER for both attention paths.
`Runtime::info()` was querying attributes of the cold attention kernel after
starting the persistent worker. Such loading can require context synchronization;
see [NVIDIA's lazy-loading guidance](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/lazy-loading.html).

The runtime now obtains and caches those attributes before starting the worker.
The same LAZY/EAGER regressions then pass. A copy-warmup-only attempt failed and
was removed. The fix does not force EAGER loading or extend the idle budget.
Failed diagnostics remain in `results/fused-attention-loading-*` and
`results/cuda-loading-regression-copy-warm-only-failed*`.

## Short results and decision

Default clocks, affinity core 96, Main, scalar GEMV, 110 blocks, graph/copy or
persistent/mapped, warmup 256. Each condition has one 1k run and three separate
varied-order 10k runs. The table gives the median run p99, without pooling:

| Main mode / backend | Staged (us) | Tiled fused (us) | Retained pre-change staged binary (us) |
|---|---:|---:|---:|
| FP32 graph | 185.287 | 232.457 | 185.347 |
| FP32 persistent | 220.497 | 261.247 | 223.367 |
| Mixed graph | 194.487 | 233.426 | 194.217 |
| Mixed persistent | 224.447 | 264.636 | 226.416 |

All 64 numerical/timing runs passed sampled GPU-process checks. No live CPU
oracle or CPU baseline was rerun during timing. The current and retained
binaries have distinct hashes; compare current staged/fused first. Raw records,
commands, CSVs and monitoring are in `results/tiled-attention/`.

The initial untiled prototype was substantially worse (FP32 graph median p99
418.205 us, persistence 458.694 us). The bounded-index-only 1k diagnostic improved
it but still lost; raw data are in `results/fused-attention-ring-index-smoke`.
All losing results are retained.

Four finite 512-event Nsight Systems captures are in
`results/attention-profiles-tiled/`. They include partial and full histories;
graph captures also include constructor warm-up. Compared with the first prototype, the profiled attention
kernel median decreases from about 138 to 45.5 us; these instrumented timings
are explanatory diagnostics, not host-service measurements or final speedup
evidence. The graph still performs its attention work in only four blocks on
Main. More history parallelism is a future experiment, not a demonstrated cause
or a promised improvement. No hardware-counter data were collected; the earlier
counter-permission denial remains documented.

Keep staged attention as the default. This experiment does not establish
Small/Large performance, final 100k-by-five comparisons, or long-stream safety.
Reproduce with `uv run --no-sync python -m scripts.check_fused_attention --benchmark`
and `uv run --no-sync python -m scripts.profile_attention`.
