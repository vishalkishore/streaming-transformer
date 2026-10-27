# Experiment notebook

## 27 October 2026 — bounded crossover survey

Added `docs/Crossover.md`, guarded `scripts/check_crossover.py`, an audited plot
script and two aggregation regressions. The source analysis counts cooperative
barriers and explains phase imbalance/resource costs without claiming measured
barrier durations. No numerical routines/defaults were changed in this step.

72 initial 1k size/grid/window conditions and 72 accepted 10k conditions have
valid sampled GPU-exclusivity evidence and benchmark SHA256
`f682dc713f82711dd991650851ef20940bca9e2bd108b11f49bf88c3b47a3194`.
Three clean varied-order 10k host-service p99 medians, graph/persistent (us):
Small FP32 96.139/120.008, mixed 101.089/124.458;
Main FP32 184.387/221.246, mixed 195.326/224.996;
Large FP32 466.961/508.921, mixed 474.291/514.010.
Persistence also loses preloaded device means by 7.2–13.2%; live signaling alone
does not explain the gap. No persistence-winning crossover found. Smaller 4/16
block grids slow both backends on Small/Main; Main windows 32/512 also favor graph.

A Python suite accidentally overlapped the original 10k repeat 0; all 24 runs
in that repetition are explicitly excluded, retained and replaced after tests.
Accepted repetitions are 1/2/3, not the unfiltered original summary. Use
`results/crossover-survey-10k/accepted-summary.json` and the two
`results/crossover-survey*-figure.json` audits. CPU baselines were not repeated.
Fresh Python tests: 60 pass (rerun serially after all GPU work). Current full
release gates, resident weights, long stress, power/arrival studies and final
100k x five runs remain pending. No native rebuild was needed for these scripts.


## 27 October 2026 — fused attention, cold-loading fix and tiled follow-up

Implemented optional `--attention fused` with project-owned device code informed
by pinned FlashInfer decode source. It shares the same operator between sequence/
graph and fused/persistent execution. FP32 accumulation, expf, explicit FP32
probability normalization, physical FP16 K/V and the existing output cast remain.
Q/K are already rotated. No upstream package was installed or source vendored.
Staged attention and scalar GEMV remain defaults.

The first prototype fused stages 6–8 into one 128-thread block per head, with
pair loads, shared probabilities and four V-reduction warps. Full model checks
passed after the runtime fix below, but short Main FP32 p99 was 418.205 us for
graph and 458.694 us for persistence, versus 184.727/220.207 us staged in the same
binary. Both mixed results lost too. All 64 first-prototype numerical/timing
sidecars passed; raw records: `results/fused-attention`. Its 512-event graph
profile attributed 62% of kernel time to attention, with median 138.08 us per
attention launch. This identifies where time accumulated, not a hardware-counter
explanation of why.

Before completing that experiment, persistent correctness initially stalled.
The worker eventually exited on its existing idle budget. Two-command fresh
process diagnostics failed under LAZY loading but passed under EAGER, in both
attention modes. `info()` queried the cold attention kernel after the worker
started. Cached those attributes in the constructor before launching persistence;
all 16 LAZY/EAGER × precision × attention × one-/110-block regressions then pass.
See NVIDIA's lazy-loading guidance linked in `docs/Attention.md`. No EAGER override
or longer timeout is part of the fix. Warming copies alone failed and was removed.
Failed direct diagnostics and the copy-only attempt are retained. A timeout-wrapper
diagnostic was rejected by the GPU-process guard and was not used as evidence.

An early build also retained an old test object because the test was edited while
compilation was running. The new test was absent from the binary; early CTest
passes did not validate it and the first filtered sanitizer ran zero cases.
Forced a rebuild, checked the actual registered-test list and repeated all
accepted checks. Only verified and tiled logs below count for the new primitive.
`results/fused-attention-registered-tests.txt` records the corrected listing.

Next isolated address computation: replace per-history uint64 modulo with one
first-slot modulo plus bounded subtraction. Primitive tests passed; four 1k
diagnostics improved the candidate but still lost (graph fused 318.176 us versus
staged 203.757 us). No 10k/final claim is made for this intermediate version.
Raw records: `results/fused-attention-ring-index-smoke`.

The current tiled follow-up cooperatively stages 128 K/V rows into shared
`[128][33]` storage, computes one QK row per thread, and retains the four-warp V
reduction. The same tile is reused with explicit producer/consumer barriers.
Support is explicitly head dimensions 16/32 and windows up to 512. This changes QK
FP32 summation order; no tolerance was weakened. Shared memory rises from 2560
to 19584 bytes/block. Actual persistent resources are 78 registers/thread,
32 local bytes and 440 legal cooperative blocks on SM120; standalone attention
uses 50 registers and zero local bytes. The tested 110-block grid is legal.

Fresh completed validation for the tiled implementation:

- 58 Python tests passed (`results/tiled-attention-pytest.log`).
- 16 cold-loading cases passed (`results/tiled-attention-cold-loading`).
- Registered FP64 primitive checks cover both precisions and dimensions,
  windows 1/7/32/64/256/512, partial/full/wrapped histories, positions through 1m,
  grids 1/4/110, poisoned unused data and output bounds.
- Memcheck/synccheck: zero errors; racecheck: zero errors/warnings/hazards
  (`results/tiled-attention-{tool}.log`). These tools executed the actual test.
- 16 full-model cases passed: all sizes × both modes × graph110/persistent110,
  plus Main graph4 and persistent1. Full hidden/K/V/score policy is unchanged.
- Fresh final CTest passed, including an extended diagnostic test comparing
  traced and normal staged/fused Main mixed execution exactly, with the expected
  attention field and full final state (`results/tiled-attention-ctest-final.log`).

Timing kept Main, scalar GEMV, grid 110, core 96, default clocks, warmup 256 and
copy/mapped transport fixed. One 1k diagnostic and three varied-order 10k runs
per condition/mode/backend; no CPU oracle or CPU baseline rerun. A retained
pre-change staged binary controls for compiler/resource changes.

| Main mode / backend | Current staged median p99 (us) | Tiled fused | Retained staged control |
|---|---:|---:|---:|
| FP32 graph | 185.287 | 232.457 | 185.347 |
| FP32 persistent | 220.497 | 261.247 | 223.367 |
| Mixed graph | 194.487 | 233.426 | 194.217 |
| Mixed persistent | 224.447 | 264.636 | 226.416 |

All 64 current numerical/timing sidecars are valid; there are no dropped events.
The audit confirms every non-control run uses the current benchmark executable
and no run has a live CPU oracle (`results/tiled-attention-audit.json`).

Four finite 512-event Nsight captures passed in `results/attention-profiles-tiled`.
The graph captures contain 513 numerical steps including constructor warm-up;
staged has 34 stage kernels/step plus input/norm/score/advance, whereas fused has 28
stage kernels plus two attention launches and those four support kernels. The
net reduction is 38→34 kernels/step. The tiled attention launch median is 45.504 us
versus the original prototype's 138.081 us, under profiling and mixed history
lengths. These are not host-service statistics or hardware bandwidth evidence.
Source headers and binaries of the abandoned intermediate variants are retained
locally; the raw graphs and failed results are not dropped.

Decision: the tested one-block/head approach still loses to staged attention.
Keep it explicitly selectable, keep staged default, and investigate more history
parallelism only as a separate measured experiment. Small/Large performance,
full final matrices, million-command stress, final 100k-by-five studies and the
report remain pending. This is not release acceptance or a persistence speedup.

## 27 October 2026 — exclusive GPU rerun and pair-load GEMV experiment

The user required an empty GPU before testing and requested recalculation of
the preceding reduction results. No project benchmark was active when checked;
no unrelated process was stopped. Added `bench/gpu_guard.py`: wait for no compute
or graphics clients, serialize workspace launchers, allow only the launched
process tree and monitor once per second. Competition or query failure rejects
a run. JSON sidecars retain evidence and the sampling limitation. Benchmark,
native/window/adversarial/PyTorch validation launchers use this guard. Final
study caching now requires an unchanged valid process-evidence artifact under
the new policy; old cached runs cannot satisfy the new identity.

Recalculated the preceding NVIDIA-reduction A/B with the same retained binaries,
Main FP32, 110 blocks, warmup 256, core 96 and default clocks. No CPU baseline or
live CPU oracle was rerun. All 20 process-evidence sidecars are valid (four
correctness runs and sixteen 1k/10k timing runs). Raw data and driver:
`results/nvidia-reductions-ab-exclusive/`. Three 10k repetitions give median
p99 **236.847 → 230.007 us** for persistence (2.89% lower), and
**184.857 → 185.688 us** for graph (0.45% higher). These supersede the earlier
unguarded diagnostic numbers below, not a final five-by-100k study.

Next, followed the user's pinned llama.cpp/FlashInfer references. Adapted only
llama.cpp's MIT-licensed FP32-accumulating float2/half2 load loops from
`de5122d879d00d2895c36be7e0cee94105aeeabd`, with license and changes retained in
`third_party/llama_cpp`. Runtime option `--matvec vector` selects the pair-load
experiment; scalar is still the default. Input/weight alignment and even column
counts are required, otherwise scalar fallback is used. FP16 weights convert
to FP32 before multiplication; intermediates and output cast boundaries remain
unchanged. Adjacent-pair lane assignment changes the reduction order. No new
package was installed. FlashInfer and Mirage remain inspected references only.

Hypothesis: pair loads reduce load/loop instruction pressure. Kept warp-per-row
ownership, precision, transport and synchronization fixed; no gate fusion,
multi-warp row ownership or online attention was added. A runtime branch keeps
both variants in the same executable. This increases compiled register use:
graph stage 40 →44 and persistent 56 →78 compared with the preceding binary;
reported local bytes remain 32. Do not attribute changes between these binaries
to vector loads alone. The within-binary comparison below isolates flag choice,
but does not measure standalone GEMV latency or establish the cause of a loss.

Native primitive tests compare scalar/vector against FP64 dots across all
Small/Main/Large projection/FFN/score shapes plus odd widths, with aligned and
independently misaligned input/weight pointers, FP32/mixed storage and grids
1/4/110. Native CTest passed under the GPU guard. Twelve full-model vector
conditions passed: three sizes × two modes × graph110/persistent110, each
3W+9 events including fixture resets and complete state comparisons.
`scripts/check_vector_matvec.py --benchmark` reproduces these gates followed by
the short Main A/B. An initial invocation rejected an omitted CLI allow-list
entry before CUDA work; fixed it and retained the rejection logs in
`results/vector-matvec-cli-rejected*`.

All 44 guarded numerical/timing runs passed process monitoring in
`results/vector-matvec/`. Timing: one 1k run per variant/backend/mode, followed by
three varied-order 10k repetitions, same 110-block grid, graph/copy and
persistent/mapped, core 96, warmup 256, default clocks. No CPU model active during
timing, and GPU runs were serial.

| Main mode / backend | Scalar median p99 (us) | Vector median p99 (us) |
|---|---:|---:|
| FP32 graph | 187.837 | 196.617 |
| FP32 persistent | 223.557 | 231.177 |
| Mixed graph | 197.618 | 197.668 |
| Mixed persistent | 226.508 | 230.827 |

Decision: no pair-load speedup on the tested Main configurations. Keep scalar
as the default and retain vector as an explicitly selected experiment. This
does not rule out vectorization with different row ownership or compile-time
specialization. Graph still wins. Python tests pass 58 cases, including waiting,
graphics/compute visibility, process ownership, contamination/query failure and
cache evidence checks (`results/gpu-guard-vector-pytest-final.log`). Final native
log: `results/vector-matvec-ctest-final.log`. Million-command stress remains
stopped and no final performance claim is made.

## 27 October 2026 — pinned source review and NVIDIA reduction adaptation

User direction: inspect infer, NVIDIA dl-lowlat-infer and Cohere; borrow useful
licensed code, preserve this model and validate with short runs before stress.
The pinned file/blob manifest is `docs/repository-review.json`; the detailed
comparison and reuse decisions are in `docs/RepositoryComparison.md`. Upstream
repositories were inspected, not built or benchmarked locally. No license file
was found in the inspected infer tree, so it remains a design reference.

Adaptation: NVIDIA FasterTransformer commit
`df4a7534860137e060e18d2ebf019906120ea204`, warpReduceSum/warpReduceMax, specialized
to FP32 and renamed in `third_party/fastertransformer/WarpReduce.cuh`. Preserve
the copyright and Apache-2.0 license; the vendor README records the source hash
and modifications. No dependency install was needed. Native provenance now
includes third-party sources. This is independent of the three reviewed engines.

Hypothesis: the XOR tree makes the result available in all lanes and eliminates
the extra broadcast in RMSNorm and the softmax maximum/denominator. Keep row-dot
reductions, FP16 casts, FP32 arithmetic and ring state unchanged. No fast-math or
FMA policy change. This is one reduction-mechanism experiment, not a fused model
or scheduling rewrite. The all-lane test covers signed cancellation on grids
1/4/110; existing softmax/value tests cover ragged lengths and wrapped slots.
New independently written Python tests guard adjacent-pair RoPE and uniform
attention over all retained values through eviction/wrap/reset.

Environment: RTX PRO 5000 Blackwell, SM120, 110 SMs; CUDA compiler 13.0.88 and
runtime 13.0, host compiler GCC 13.3.0. Release build, architecture 120, `-O3`,
`--fmad=false`, line info and ptxas resource reporting. Exact build output:
`results/nvidia-reductions-build-final.log`.

Validation, completed serially before timing:

- Native CTest: both targets passed (`results/nvidia-reductions-ctest.log`).
- 36/36 changed-backend fixture cases: Small/Main/Large × FP32/mixed × graph4,
  cuBLAS-graph4, fused1/4 and persistent1/4
  (`results/nvidia-reductions-check/summary.json`). This is not a fresh 54-case
  all-backend matrix. Previous 54-case acceptance belongs to the previous binary.
- Four additional Main FP32/mixed graph110/persistent110 correctness cases
  passed (`results/nvidia-reductions-ab/check-*.json`).
- 52 Python tests passed (`results/repository-review-pytest.log`).

Short A/B: retained old executable
`build-blackwell/stream-bench-before-nvidia-reductions` versus new executable;
Main FP32, grid 110, graph/copy and persistent/mapped, worker affinity core 96,
default dynamic clocks, warmup 256, reset semantics unchanged. Each variant had
a 1k diagnostic then three varied-order 10k repetitions. CPU baseline/oracle
was not rerun during timing; all timing results record `cpu_oracle_active=false`.
GPU jobs were serial. Raw JSON/CSV, commands, telemetry, locality snapshots and
the runnable experiment driver are in `results/nvidia-reductions-ab/`.

| Backend | Before p99, three 10k runs (us) | NVIDIA adaptation p99 (us) | Median change |
|---|---|---|---|
| Graph110 | 186.168, 187.188, 185.049 | 185.328, 186.238, 185.349 | 186.168 → 185.349, 0.44% lower; ranges overlap |
| Persistent110 | 238.278, 236.758, 240.918 | 230.238, 230.558, 229.588 | 238.278 → 230.238, 3.37% lower |

Persistent kernel register use increased 53 → 56/thread; reported local bytes
remain 32 and shared bytes0. Graph stage registers remain40. The short reduction
gain is measured; its attribution to the eliminated broadcasts remains a
hypothesis without internal phase profiling. Persistence still loses to the
strongest tested graph configuration. Retain the adaptation for further study;
do not turn this three-repeat diagnostic into final 100k-by-five evidence.

The earlier broad graph sweep (`results/quick-main-tuned`) completed five 10k
comparisons: median graph110 p99 183.38 us versus persistent110 236.72 us.
That result supersedes comparisons against the weak graph4 configuration.
CPU's earlier single 1k p99 was38.89 us and was not remeasured here. All losing
results remain available. Million-command stress was not restarted. Remaining
release gates include long stress, additional matrices, final workload/power
studies, supported profiles and the report; borrowed code does not waive them.

## Continuation — raw analysis, provenance and release audit

Added `bench/analyze.py` to recompute host distributions from individual samples
using the native nearest-rank definition. It preserves offered/completed/drop
counts, reports deadline failures including drops, and rejects correctness or
device runs as host performance evidence. All four distributions exactly match
the native output for `results/acceptance-release/main-fp32-persistent-4`; this
is an analyzer check, not a performance result
(`results/analysis-raw-verification.json`). Telemetry is filtered to the measured
interval using recorded clock correlation; unavailable metrics remain absent,
and wall-clock offset changes above 5 ms invalidate correlation. Power is a
sample mean with count/span, not inferred whole-run energy.

Study resume stamps now include SHA256 identities of resolved shared libraries
and the uv-reported GPU package set. PyTorch result provenance includes source,
interpreter and resolved Git/dirty-state identities. Twelve linked native
libraries were resolved and hashed successfully
(`results/continuation-provenance.json`). No immutable commit exists yet.

All 30 Python tests passed (`results/analysis-pytest.log`), including raw-drop,
tail-definition and telemetry-boundary regressions. The requirement-level audit
is in `docs/ReleaseChecklist.md`; it explicitly lists unfinished studies and
report questions. The first million-command stress case and the waiting serial
validation followup remain live. No new GPU performance measurements, system
clock changes or final performance claims were made.

## Continuation — complete native matrix and independent trace checks

The final `mixed-depth-rms-v3` acceptance policy is documented in
`docs/NumericalPolicy.md`. Its derivation includes native operation-level checks
and independent FP32/FP64 confirmation; rejected policies and original scalar
failures remain in the result archives. The confirmation passed 18/18 held-out
cases (`results/numerical-policy-confirmation.json`). The fresh native matrix
passed all 54 Small/Main/Large × precision/backend conditions
(`results/acceptance-release/summary.json`). These are correctness runs, not
service latency measurements. Four million-command persistent cases are running
serially; no completed stress gate is claimed in this entry.

An audit found that the extended reference checker still used the superseded
sqrt-depth score rule and tested full-sequence agreement without reset. It now
uses the same frozen score policy, resets both independent implementations after
`2*window+3` events, and restarts masked absolute positions at the boundary.
CPU-only checks passed all 36 size/precision/input-family combinations
(`results/extended-independent/summary.json`, `.log`). Native adversarial checks
and the window and PyTorch matrices remain pending. CPU reference checks did not
start a second GPU client or collect performance evidence.

The study runner now requires the complete distinct 54-case matrix plus all four
million-command stress cases, matching executable hashes, complete event counts,
correct backend/configuration labels, nonempty checked state, and zero accepted
numerical/nonfinite failures. It also includes separate backend-specific
saturation curves; common-rate comparative schedules remain identical across
backends. Synthetic gate and schedule regression tests exercise missing,
duplicated, mislabeled, truncated and erroneous evidence and verify rate/count
matching. All 27 Python tests passed (`results/continuation-pytest.log`). No new
performance study has been run, and the full objective remains incomplete.

## 27 October 2026 — Blackwell migration and numerical gates

The user selected the available RTX PRO 5000 Blackwell, SM120, for final results. Earlier A5000/A6000 scope and results remain historical. No competing GPU jobs were stopped and no clock, affinity, NUMA or power policies were changed.

Environment: driver 580.105.08; CUDA toolkit 13.0.88; GCC 13.3.0; CMake 3.28.3; Ninja 1.11.1. Oracle dependencies were installed using `uv` in `.venv`, including Torch 2.6.0+cpu. Full software/hardware output is recorded in `results/blackwell-environment.json` and `results/blackwell-hardware.json`.

The initial transferred SM86 build compiled after replacing removed `cudaDeviceProp` clock fields with device-attribute queries, but native backends were rejected by the runtime's SM86-only device guard. Preserve `results/destination-validation` as an unsuccessful migration attempt. The final target is compiled natively with `-DCMAKE_CUDA_ARCHITECTURES=120`, `-O3`, `--fmad=false`, line information and ptxas resource output. Fast math and TF32 remain disabled. No SM86-to-Blackwell timing comparison is claimed.

Small/Main FP32/mixed fixtures were regenerated with the original seeds and reset indices. Tensor/archive/trace hashes were verified; see `results/blackwell-fixture-hashes.jsonl`. The initial SM120 matrix is in `results/blackwell-validation`; the matrix following the final native changes is in `results/blackwell-validation-final`.

Numerical hypothesis: FP32 reduction differences near FP16 rounding midpoints propagate through mixed residuals. Native pre/post-cast and common-operand FP64 checks support this mechanism. All 16 isolated native linears satisfied the conservative FP32 dot bound on the Main trace. See `docs/NumericalDiagnosis.md` for the concrete first disagreement and limitations. Tolerances remain unchanged and Main mixed remains unaccepted.

The sequence-exhaustion review found that consuming the final request ID could prevent STOP publication. Reserve the final ID for STOP and reject RUN/RESET before that point. New native tests cover one-/four-block exhaustion and clean STOP, runtime reconstruction, repeated zero scores, consecutive resets, finite RUN limits and idle exit. This bounded coverage is separate from the outstanding million-command gate.

The CPU snapshot regression initially assumed distinct Eigen allocations would be bit-identical. That assumption failed; the failure log is retained. Replaying one allocation with snapshots on/off passes Small/Main in both modes. Python tests also check deterministic regeneration and transfer-corruption detection.

Compute Sanitizer memcheck passed the FP32/mixed primitive test with zero errors (`results/blackwell-sanitizer-primitives.log`). It subsequently passed all seven native tests, including the one-/four-block lifecycle tests, with zero errors (`results/blackwell-sanitizer-native.log`). This bounded coverage is not complete sanitizer acceptance or proof of the CPU/GPU mailbox protocol. Final profiles, long stress, transport repetitions, baseline tuning, Large/scaling studies and final performance results remain pending.

Final checks for this session: 14 Python tests passed; all seven native tests passed; the 36-condition fixture matrix passed 27 and retained the nine Main mixed hidden failures. Every Main mixed score and KV element passed the unchanged thresholds. No stress or final study was run after the failing matrix.

The repository has no commit yet. Benchmark output now reports a null resolved commit, actual Git status, source snapshot hash and executable hash instead of treating `.git/HEAD`'s symbolic-ref text as a commit. Create an immutable commit and retain dirty-state identification before collecting final evidence.
