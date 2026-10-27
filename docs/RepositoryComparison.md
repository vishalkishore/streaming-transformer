# Source comparison and correctness review — 27 October 2026

New web research: [PersistentDesignResearch.md](PersistentDesignResearch.md)
compares Hazy Research, Cohere and NVIDIA write-ups with our staged runtime,
and gives a concrete instrumentation/fusion/prefetch/dependency sequence.

The graph/persistence explanation and bounded size/grid/window survey are now in
[Crossover.md](Crossover.md). No persistence-winning crossover has been found;
losing cases are part of the engineering result.

Latest continuation: a project-owned tiled attention prototype is implemented
as `--attention fused`, informed by FlashInfer. It passes 16 model cases,
16 cold-loading regressions and memcheck/synccheck/racecheck. It still loses on
Main, so staged attention remains default. Three 10k run medians (us):

| Mode / backend | Staged | Tiled fused |
|---|---:|---:|
| FP32 graph110 | 185.287 | 232.457 |
| FP32 persistent110 | 220.497 | 261.247 |
| Mixed graph110 | 194.487 | 233.426 |
| Mixed persistent110 | 224.447 | 264.636 |

All 64 numerical/timing runs passed sampled process monitoring. CPU baselines
were not repeated. The initial untiled version was worse and is retained as
diagnostic evidence. A cold CUDA metadata-query stall was fixed by preloading
attributes before starting persistence. See [Attention.md](Attention.md) for
the algorithm, exact validation, profiler evidence, failed attempts and limits.

Latest numerical update: the earlier reduction A/B below was rerun with an
empty-GPU check and continuous sampled process monitoring. All 20 guarded runs
were valid (`results/nvidia-reductions-ab-exclusive`). Three 10k repetitions now
give median graph p99 184.857 → 185.688 us (0.45% higher), and persistent p99
236.847 → 230.007 us (2.89% lower). These supersede the unguarded diagnostic
numbers below; graph still wins. No CPU baseline was repeated.

Our model contract remains authoritative. This review inspects pinned upstream
source; it does not claim that their binaries were built, their tests passed
locally, or their published timings transfer to our Blackwell GPU. The inspected
file/blob manifest is [repository-review.json](repository-review.json).

## Transformer operations: mann-uofg/infer

Reviewed commit `aa51492a5ec4907cd70a71f2c628d73d7825ef9e`:
[GEMV](https://github.com/mann-uofg/infer/blob/aa51492a5ec4907cd70a71f2c628d73d7825ef9e/src/kernels/gemv.cu),
[decode](https://github.com/mann-uofg/infer/blob/aa51492a5ec4907cd70a71f2c628d73d7825ef9e/src/kernels/decode.cu),
[reference](https://github.com/mann-uofg/infer/blob/aa51492a5ec4907cd70a71f2c628d73d7825ef9e/reference/llama.py),
[kernel tests](https://github.com/mann-uofg/infer/blob/aa51492a5ec4907cd70a71f2c628d73d7825ef9e/tests/kernel_test.cu),
and [engine tests](https://github.com/mann-uofg/infer/blob/aa51492a5ec4907cd70a71f2c628d73d7825ef9e/tests/python/test_engine.py).

| Mechanism | Upstream | Our implementation / decision |
|---|---|---|
| GEMV | Vector loads, several rows per warp, shared input staging, prefetched next loads | `Primitives.cuh::matvec` owns one row per warp and loads scalar elements. Vector loads and rows-per-warp are concrete optimization candidates; verify alignment, tails, register pressure and accumulation order. |
| Fusion | Optional RMSNorm input staging and fused gate/up activation or residual output | Separate explicit stages preserve our FP16 cast boundaries. Implement fusion only with those boundaries retained. |
| Decode attention | Split-KV, tiled shared K/V, online softmax; optimized kernel statically requires head dimension 64 | Main head dimension is 32. Current separate score, stable softmax and value reductions are correct for our shapes. A new 32-wide tiled kernel needs its own oracle and grid sweep. |
| RoPE | Coordinates `i` and `i + D/2` | We rotate adjacent pairs. Copying upstream unchanged changes the model. |
| KV indexing | Absolute position indexes a bounded maximum-sequence allocation; attention reads the prefix | Our allocation is a circular local window. Preserve absolute RoPE positions, physical modulo slots, eviction and valid length per layer. |
| Precision | BF16 weights/cache; explicit FMA and fast exponential in optimized paths | Our mixed mode uses FP16 at documented intermediate boundaries, with FP32 reductions and `--fmad=false`. Upstream arithmetic and tolerances cannot be imported unchanged. |
| Graphs | Device-held position permits capture/replay | Already implemented locally. Both graph and persistent paths own state on device. |
| Correctness | Primitive comparisons, layer residuals, prefill versus decode and generated-token agreement | We compare score and full hidden/K/V snapshots, independent full-sequence versus streaming references, resets and wraparound. Token equality is inapplicable to our scalar-output model. |

The inspected tree has no license file, and the inspected implementation files
have no reuse grant. Use these as design references; no source from this repo has
been copied. The independent reference structure is useful, but its global
max-relative-error threshold would not replace our component and per-layer
state checks. Neither project's bounded tests establish million-command safety.

## Host/device runtime: NVIDIA dl-lowlat-infer

Reviewed commit `a718567b47e8d80ac38d58ac76df8d98a0eaab3d`:
[PingPongModelPersistent](https://github.com/NVIDIA/dl-lowlat-infer/blob/a718567b47e8d80ac38d58ac76df8d98a0eaab3d/src/cpp/nvlstm_inf/PingPongModelPersistent.cuh),
[ModelPersistentBase](https://github.com/NVIDIA/dl-lowlat-infer/blob/a718567b47e8d80ac38d58ac76df8d98a0eaab3d/src/cpp/nvlstm_inf/ModelPersistentBase.cuh),
[GEMV/signaling helpers](https://github.com/NVIDIA/dl-lowlat-infer/blob/a718567b47e8d80ac38d58ac76df8d98a0eaab3d/src/cpp/nvlstm_inf/ss_gemv_impl/gemv_lstm_utils.cuh),
and [ping-pong tests](https://github.com/NVIDIA/dl-lowlat-infer/blob/a718567b47e8d80ac38d58ac76df8d98a0eaab3d/test/cpp/nvlstm_inf/ping_pong_model_test.cu).

Its acquire polling, release signaling and explicit exit handshake support the
same basic ownership pattern we use. Some paths use exchange/CAS or alternate
flags; our aligned system-scope atomic load/store sequence protocol intentionally
keeps its existing semantics. Our completion ID is separate from the score, so
zero or repeated scores cannot masquerade as missing completion.

Upstream finalization requests exit, waits for acknowledgement, synchronizes the
CUDA stream and joins its helper thread. Our `Runtime::stop` likewise sends STOP
and synchronizes before resources are freed; lifecycle tests cover finite exit,
reset and sequence exhaustion. Long-stream validation remains pending.

The upstream model is LSTM, and it includes precomputation, GDRCopy options and
architecture-specific implementations. Its README example targets SM90a; those
paths are not SM120 compatibility evidence. Reuse of portable Apache-2.0 helpers
is possible, but replacing our working protocol with theirs offers no established
advantage. Profile our transport direction/backoff variants first.

## Task scheduling: Cohere megakernel

Reviewed commit `67d0b9ca22ea3652796b715d1d1863459e0e2c3c`:
[megakernel](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/megakernel.cuh),
[runtime](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/runtime.cu),
[ABI](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/decode/abi.h),
and [layer tests](https://github.com/cohere-ai/cohere-megakernel/blob/67d0b9ca22ea3652796b715d1d1863459e0e2c3c/src/tests/test_decode_layers.py).

Tasks encode input-barrier indices/thresholds and output-counter arrivals, with
producer/consumer warp roles and asynchronous tile staging. Our complete forward
pass instead uses block or cooperative-grid barriers between explicit stages.
The source motivates measuring dependency stalls and fusing independent QKV or
gate/up work; it does not establish that replacing our barriers is safe or faster.

Its build emits SM90a, uses Hopper-specific tensor/TMA machinery, and fixes the
attention ABI to head dimension 128. Its BF16 paged-cache/MoE workload and host
service are different. The runtime's host stop flag is also a different mechanism
from our GPU mailbox STOP command. Preserve our validated synchronization until
a separately tested dependency protocol has an ordering argument. Apache-2.0
permits attributed adaptation; a wholesale port is outside this project's scope.

## Local action and evidence

1. Added original regression tests for adjacent-pair RoPE and uniform attention
   over the entire retained history, including eviction, wraparound and reset.
   These target the concrete semantic hazards found in this review.
2. Adapted the small, licensed NVIDIA FasterTransformer warp reductions rather
   than importing an entire engine. Attribution and modifications are in
   [the vendor note](../third_party/fastertransformer/README.md).
3. Preserve the strongest graph comparison: preceding five 10k Main FP32 runs
   measured median p99 183.38 us for graph110 versus 236.72 us for persistent110.
   This is a losing persistent result and short diagnostic evidence only.
4. Next experiments: aligned vector loads/rows per warp, fusion with explicit
   casts, then resident tiles where occupancy permits. Change one mechanism at
   a time; validate before 1k/10k GPU-only timing. Do not rerun the CPU baseline
   for every optimization.

Fresh adaptation acceptance and before/after timing are recorded in
[LAB_NOTEBOOK.md](../LAB_NOTEBOOK.md). Final million-command, adversarial/window/
PyTorch matrices and 100k-by-five workload/power studies remain outstanding.

The adaptation passed 36 changed-backend fixture conditions plus four tuned-grid
Main conditions, native CTest and 52 Python tests. In the three guarded 10k
Main FP32 A/B repetitions, median p99 changed from 236.85 to 230.01 us for
persistent110 (2.89% lower), and 184.86 to 185.69 us for graph110 (0.45% higher).
The graph still wins. These short runs support retaining
the primitive for further investigation, not a final performance claim.

## Next primitive references: llama.cpp and FlashInfer

Inspected pinned [llama.cpp mmvf.cu](https://github.com/ggml-org/llama.cpp/blob/de5122d879d00d2895c36be7e0cee94105aeeabd/ggml/src/ggml-cuda/mmvf.cu)
and its MIT license. Our optional `--matvec vector` adaptation uses aligned
float2 activation/FP32-weight loads and half2 FP16-weight loads converted to
FP32. It keeps one warp per row, the existing reduction and output casts, and
explicit multiply/add under `--fmad=false`. It does not adopt upstream half
accumulation, gate fusion or configurable multi-warp rows in this experiment.
Odd dimensions and unaligned input/weight pointers fall back to scalar loads.
The same implementation is available to custom graph and persistent paths;
scalar remains the default until the experiment is evaluated. Attribution,
license and modification details are in `third_party/llama_cpp`.

Inspected [FlashInfer decode.cuh](https://github.com/flashinfer-ai/flashinfer/blob/07cc0d60a9ba1aca87b7b803ab1810ca5e3738d6/include/flashinfer/attention/decode.cuh)
and [norm.cuh](https://github.com/flashinfer-ai/flashinfer/blob/07cc0d60a9ba1aca87b7b803ab1810ca5e3738d6/include/flashinfer/norm.cuh).
Their vectorized shared-memory loading, online attention state and chunking
informed our project-owned single-block/head prototype. No FlashInfer source
was vendored and the upstream project was not built. Our tested prototype
handles 16-/32-wide heads, the chronological ring view and already rotated Q/K,
while retaining explicit normalized FP32 probabilities. Online rescaling and
split-KV remain separate future experiments.

The pinned [Mirage task tree](https://github.com/mirage-project/mirage/tree/63af900000d8928a885f7af7889a3448621e18a9/include/mirage/persistent_kernel/tasks/blackwell)
contains explicit SM100 attention/linear kernels. This tree inspection does not
establish SM120 support; scheduler adoption remains deferred. None of these
upstream projects was benchmarked locally by this review.

The pair-load experiment passed 12 full-model vector conditions and native
FP64-dot/alignment/tail tests. All 44 GPU-run sidecars were valid. Three 10k Main
repetitions produced these median host-service p99 values:

| Mode / backend | Scalar (us) | Vector (us) |
|---|---:|---:|
| FP32 graph110 | 187.837 | 196.617 |
| FP32 persistent110 | 223.557 | 231.177 |
| Mixed graph110 | 197.618 | 197.668 |
| Mixed persistent110 | 226.508 | 230.827 |

Scalar remains the default: vectorization did not improve these Main runs.
Both variants above use the same rebuilt executable. Runtime dispatch increases
compiled register use relative to the preceding binary, so these numbers must
not be mixed with the earlier NVIDIA-reduction A/B. This experiment does not
measure isolated GEMV latency or establish performance on other model sizes.
Raw evidence: `results/vector-matvec`; reproduce with
`uv run --no-sync python -m scripts.check_vector_matvec --benchmark`.
