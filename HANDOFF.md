# Project handoff — updated 27 October 2026

## Latest steering — CPU once, short GPU experiments first

The user explicitly stopped the old million-command run and its waiting
supervisor. PIDs 58099/57248/60274 are no longer active. Interrupted logs are
preserved; no million-command completion is claimed. Do not restart that old
CPU-per-event workflow. `results/stopped-validation-user-request.json` records
the stop. A CPU-only `stream-oracle` now exports all scores and sparse full-state
snapshots once per precision/trace. GPU replays use `--expected ORACLE_DIRECTORY`
without constructing a CPU model. `scripts/validate.py` and adversarial validation
reuse those stored results, with archive/config/checkpoint/trace/reset identity
checks and correctness progress every 10,000 events.

The stored-oracle bounded regression passed independent reference agreement,
three GPU replays per precision and four invalid-oracle rejections per precision
(`results/stored-oracle-regression`). Python tests pass 44 cases. The user then
requested 1k/10k experiments before long stress. `scripts/quick_bench.py` runs
Main FP32 serially, measuring CPU only once; `--skip-cpu` reuses that baseline.
No million-command rerun has been started.

GPU softmax now uses a warp per head; attention-value accumulation uses a warp
per output coordinate and distributes history across lanes. These change FP32
reduction order, so the earlier 54/54 matrix is historical evidence for the
previous binary, not full acceptance of the new binary. The new Main FP32/mixed
graph/cuBLAS-graph/persistent one/four-block matrix passed 8/8. New primitive tests
cover ragged softmax lengths, large logit offsets, history lengths and wrapped
value slots across one/four/110 blocks. Check the final CTest log and fresh full
matrix before broadening correctness claims.

Short default-clock, affinity-pinned Main FP32 results (single runs; not final
repeated evidence): initial persistent110 10k p99 707.41 us, after warp softmax
590.58 us, after parallel values 236.56 us. Tested cuBLAS graph4 changed from
877.42 to 764.02 to 529.16 us. CPU was measured once at 1k p99 38.89 us and still
wins this configuration. Raw records: `results/quick-main*`. One-block persistent
did not improve with the value-work distribution; preserve that losing result.
The graph baseline is not yet fully tuned, and different-time runs need repeated
controls before final improvement claims.

The source snapshot is public at https://github.com/vishalkishore/streaming-transformer,
with published main `e8ecd375fc9c6e7e88ad817f8f93839da3b14ebe`. Current local changes
after publication are not yet pushed. Initial local commit remains on
`local-initial-snapshot`. Research references and applicability limits are in
`docs/ImplementationReferences.md`. The older active section below describes
superseded jobs and numerical checks; this section takes precedence.

## Active continuation — numerical acceptance before final studies

The full objective remains active. Small/Main/Large fixtures and all 32/64/128/256/512 window variants now exist. A separate uv-managed `.venv-gpu` contains Torch 2.14.1+cu130 with SM120 support; `.venv` retains the Torch 2.6.0 CPU oracle. `bench/torch_bench.py` implements eager, Inductor and explicit CUDA Graph comparisons with device-held positions and persistent KV state. Its final full correctness matrix and performance studies are not yet complete.

Actual CUDA diagnostic snapshots passed common-operand checks for all 16 Main and 30 Large linears, normalization, activation and residual operations. The RoPE absolute-position test passes through position 1,000,000. Primitive tests now inherit the runtime's `--fmad=false` flag. CPU/GPU performance runs must remain serial; correctness may use the CPU oracle but is not latency evidence.

`results/acceptance-v2/summary.json` is terminal: **45/54 passed**. All nine Large mixed conditions failed its provisional metrics; stress was correctly skipped. The later `mixed-depth-rms-v3` policy passed all 18 independent confirmation cases (`results/numerical-policy-confirmation.json`) and all 54 native Small/Main/Large × precision/backend conditions (`results/acceptance-release/summary.json`). The four million-command stress cases are running sequentially; at this update the first Main FP32 one-block case is active, and no stress completion is claimed. Read [docs/NumericalPolicy.md](docs/NumericalPolicy.md) for the exact rules and preserved rejected metrics. Do not treat an in-progress matrix, a policy proposal or raw diagnostic timing as final acceptance/performance evidence.

Known environment restrictions: Nsight Compute returns `ERR_NVGPUCTRPERM`; `numactl --membind=1` returns operation-not-permitted; CPU governor files are read-only. No global clock/governor changes have been made. Continue supported checks and record the exact unavailable measurements.

Latest analysis work: `bench/analyze.py` recomputes nearest-rank distributions,
retains overload/drop counts, filters telemetry using host clock correlation,
and optionally exports completed-event CDFs. All four distributions match an
actual native raw CSV exactly (`results/analysis-raw-verification.json`), without
treating that correctness timing as performance evidence. All 30 Python tests
pass (`results/analysis-pytest.log`). Study stamps now hash resolved native shared
libraries and the uv-reported GPU package set; PyTorch includes Git/source and
interpreter provenance. `docs/ReleaseChecklist.md` tracks the full remaining
scope. Final cross-run summaries, actual plots and report still require final
measurement data.

Latest continuation: all 36 independent full-sequence/streaming reference cases passed
across three sizes, two modes and six trace families, with explicit resets
(`results/extended-independent/summary.json`). This CPU-only run is separate from
the pending native adversarial matrix. All 27 Python tests passed
(`results/continuation-pytest.log`), including new evidence-gate and load-schedule
regressions. The runner now has distinct common-rate and backend-specific
saturation phases; neither has produced final measurements.

`scripts/validation_followup.py` is queued behind the live validation PID 57248
(`results/validation-followup.log`). It verifies process identity while waiting,
then requires the full 54-case matrix and four successful million-command cases
from the same executable before running fresh CTest, memcheck, synccheck, native
adversarial traces, window validation and the PyTorch matrix sequentially. It
stops on failure or another GPU client. Inspect the actual process/log and
`results/validation-followup/summary.json` before claiming any stage completed;
the existence of the script or queue is not evidence of those checks passing.

Remaining completion work includes numerical acceptance, all four million-command persistent stress cases, extra trace/window and PyTorch validation, baseline tuning, transport/noise and supported profiler studies, final repeated service/arrival/power measurements, analysis/plots, architecture/transport/measurement documentation, a reproducible demo and requirement-by-requirement release audit. `bench/runner.py` is resumable study infrastructure, not evidence that these runs have completed.

The earlier sections below are historical snapshots; their pending-status notes and result counts do not supersede this active continuation.

## Latest session: final target changed to Blackwell

The user explicitly selected **NVIDIA RTX PRO 5000 Blackwell, SM120, 48 GB** for final results. This supersedes the A5000 target below and the original A6000 scope. The detected device has 110 SMs. Keep historical hardware results separate; no final performance result has been collected.

Dependencies were installed with **uv**, as requested. The oracle `.venv` uses Python 3.12.15 and PyTorch 2.6.0+cpu to retain the existing reference arithmetic. Do not use pip directly for installs/removals. A Blackwell-compatible PyTorch GPU comparison remains unimplemented.

The deleted **four Small/Main FP32/mixed fixtures have been regenerated** and all archive/tensor/trace/reset checks pass. Repeat with `uv run --no-sync python -m scripts.fixtures`; verify without rewriting with `--verify-only`. See `results/blackwell-fixture-hashes.jsonl`. Large fixtures were not generated.

Build and execution changes:

- CMake discovers the installed toolkit/compiler rather than forcing the previous server's CUDA 12.5/GCC 12 paths. The default architecture is 120; explicit SM86 builds remain available.
- CUDA 13 removed two clock fields from `cudaDeviceProp`; the probe now queries the corresponding device attributes.
- Runtime supports the selected SM120 target and records compiled architectures.
- New `stream-diagnose` and `scripts/diagnose_native.py` capture actual CPU operation operands and pre/post-cast results, compare against FP32/FP64 references, and check common-operand FP64 linear arithmetic. See `docs/NumericalDiagnosis.md`.
- Fixed sequence exhaustion: reserve the final sequence ID for STOP, reject further RUN/RESET before consuming it, and allow deterministic near-exhaustion initialization via `RuntimeOptions::initialSequence`.
- Added lifecycle regression checks for one-/four-block exhaustion, STOP, reconstruction, repeated zero scores, consecutive resets, finite and idle exit. These are bounded tests, not the million-command gate.
- Benchmark provenance now records resolved Git commit (currently null: still no commit), dirty status, source snapshot SHA256 and executable SHA256. The validation runner uses the active interpreter when generating fixtures and skips stress if fixture correctness fails.

Fresh evidence on Blackwell:

| Check | Result |
|---|---|
| Native SM120 build | `build-blackwell/`, CUDA 13.0.88, GCC 13.3.0, driver 580.105.08; passed. |
| Python tests | `results/blackwell-pytest-final.log`: 14 passed. |
| Native tests | `results/blackwell-ctest-final.log`: seven GoogleTest cases, including primitive and lifecycle coverage; passed. |
| Full fixture matrix after native changes | `results/blackwell-validation-final/summary.json`: 27/36 passed. All Small and Main FP32 passed. All nine Main mixed conditions still fail 1–3 raw hidden coordinates; scores and complete KV snapshots pass. Tolerances unchanged. |
| Native mixed operation diagnosis | `results/blackwell-mixed-diagnosis/report.json`: the first isolated input cast disagreement straddles an FP16 midpoint by about 1.9e-8. All 16 common-operand native linears satisfy a conservative FP32 dot-product bound over 777 events. Main mixed remains unaccepted. |
| Compute Sanitizer primitives | `results/blackwell-sanitizer-primitives.log`: memcheck passed, zero errors. See the notebook for any additional sanitizer results. |
| Compute Sanitizer native suite | `results/blackwell-sanitizer-native.log`: all seven cases passed under memcheck, zero errors, including one-/four-block lifecycle tests. This is bounded coverage, not proof of the mailbox memory model or the million-command gate. |

The initial destination matrix used SM86 and was refused by the old runtime guard; it is retained in `results/destination-validation`, and must not be counted as model correctness failures. An early trace test assumed independently allocated Eigen buffers would be bit-identical; that assumption failed, with the failure log retained. The corrected test uses one CPU allocation with instrumentation on/off and passes Small/Main both modes exactly.

Next work remains **numerical gates → transport/lifecycle stress → baseline tuning → profiles → final studies → report**. Isolate GPU operation boundaries and normalization/activation/attention behavior before accepting a revised tensor policy. Do not weaken the current tolerances to erase the raw mixed failures. Do not run final studies or claim the million-command gate yet.

Current setup and commands: [docs/BlackwellSetup.md](docs/BlackwellSetup.md). Investigation: [docs/NumericalDiagnosis.md](docs/NumericalDiagnosis.md). Experiment log: [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md). The sections below are preserved as the **historical 26 October handoff**; their hardware, memory limitations, paths and unimplemented-status notes describe that earlier workspace.

## Historical handoff — 26 October 2026

The implementation is **partially complete**. The native backends exist and have passed bounded checks, but the full implementation plan and its final experiment gates are not complete. Continue from this workspace; do not restart the project or describe it as a finished v1.0.

The user selected the **RTX A5000, SM86** as the current target. The original plan names an A6000; report the actual A5000 hardware. Keep existing GPU jobs running. The user explicitly declined stopping the other workloads to free memory.

## Implemented and checked

| Component | Current evidence |
|---|---|
| C++20/CUDA CMake/Ninja project | `build/` compiled successfully with CUDA 12.5.82, CUDA host compiler GCC 12, host C++ GCC 13.3, architecture 86. |
| Hardware probe | `build/stream-probe`; records cooperative launch, watchdog, SM/register/shared-memory resources and mapped-memory capabilities. Existing A5000: 64 SMs, cooperative launch supported, kernel timeout disabled. |
| Python references | Independent streaming ring-buffer and full-sequence per-layer local-causal implementations; deterministic input families and an FP64 check. Latest `results/handoff-pytest.log`: **13 passed**. These pytest cases use small dimensions, not the whole final scaling matrix. |
| Checkpoint interface | Versioned JSON manifest, row-major `[out, in]` weights, contiguous tensor bytes, shapes/dtypes/offsets/SHA256; native loader checks hashes and expected model shapes. |
| Frozen fixtures | Small and Main in FP32/mixed, deterministic weights and traces, expected scores, per-layer hidden states and complete KV snapshots. Fixtures cover wraparound, eviction and reset. |
| CPU baseline | Preallocated Eigen implementation; latest change vectorizes attention over the physical ring. Fresh Main FP32 check: all 777 events, scores, hidden states and complete KV snapshots passed. Maximum score absolute error: 2.15e-6. CPU configuration/hash unit tests also pass. |
| Native GPU implementations | cuBLAS, cuBLAS graph, custom conventional CUDA, custom graph, fused single/cooperative block execution, persistent single/cooperative block execution. Mixed uses FP16 weights/KV; scratch is FP32 with explicit FP16 rounding. |
| Bounded cross-backend checks | `results/validation/summary.json`: **27/36 conditions passed**: all 18 Small conditions (FP32/mixed), all 9 Main FP32 conditions. Nine Main mixed conditions failed the initial hidden-state tolerance; see below. These checks predate the final CPU attention optimization. |
| Primitive checks | GoogleTest CUDA tests for GEMV, RMSNorm, RoPE, cache insertion/attention, SwiGLU and residuals in FP32/mixed. `results/ctest.log` records the earlier full native pass. |
| Persistent protocol | Separate aligned request/completion IDs, system-scope release/acquire load/store, RUN/RESET/STOP, finite event limit and idle budget; occupancy checked for cooperative grids. |
| Transport smoke | `results/transport-tight.json`: **100,000 CPU-initiated exchanges**, zero checksum/sequence failures. This was on a shared GPU and overlapped validation; it is not a final isolated transport result. |
| Benchmark infrastructure | CLI modes for correctness, service, scheduled arrivals, finite device traces and transport. Scheduled runner has an independent producer, bounded preallocated queue, explicit drops, and intended/enqueue/submission/completion timestamps. Raw CSV and JSON outputs are implemented. |
| Study tooling | `bench/study.py` implements serial comparison matrices, changed backend order, telemetry, common graph-anchored rates, bursts, grid/window/idle/thread sweeps. It has not been run or fully validated. |

Important source files: `reference/model.py`, `reference/export.py`, `include/{Model,Cpu,Runtime}.hpp`, `src/cpu/Cpu.cpp`, `src/cuda/{Primitives,Resources}.cuh`, `src/cuda/Runtime.cu`, `src/runtime/Model.cpp`, `bench/Main.cpp`, `bench/study.py`, `scripts/{validate,environment,diagnose_mixed}.py`.

## Unresolved correctness and execution gates

1. **Main mixed precision is not accepted.** Scores and KV snapshots passed the starting mixed tolerance, but 1–4 raw residual hidden coordinates failed per backend. For the initial CPU implementation, hidden max absolute error was 0.0078125 and relative L2 was about 4.26e-4. Some near-zero residual coordinates differ by 0.0021–0.0029, exceeding the starting absolute bound. No tolerances have been changed.
2. `results/mixed-reduction-diagnosis.json` demonstrates sensitivity to FP32 versus FP64 arithmetic with identical FP16 cast boundaries: first divergence already occurs in input projection; two residual coordinates fail the same starting test between these two Python computations. This supports a rounding/reduction hypothesis but does not by itself close the native correctness investigation. The optimized CPU mixed diagnosis is also preserved in `results/mixed-diagnosis-eigen.*`.
3. **The million-command gate has not passed.** First attempt found a benchmark-checker bug: fixture reset indices and periodic stress resets were both applied, while expected positions followed only the periodic resets. The checker now excludes fixture resets for `--expected cpu`. The retry failed at CUDA initialization with out-of-memory, before processing events. Preserve this distinction; do not claim a runtime position bug or a successful million-event run.
4. Another GPU job subsequently consumed the remaining memory. Last observation: about 458 MiB free, two vLLM workers plus `tile_matching.py`; even a minimal CUDA allocation probe failed. GPU checks must resume after memory is available or on the destination server. Do not terminate those jobs.
5. The last CPU attention optimization passed fresh Main FP32 validation, but the complete cross-backend matrix and GPU primitive suite have not been rerun after that change.

## Remaining plan work

- Finish Main mixed diagnosis at operation/cast boundaries; inspect error distributions and justify any tensor-specific acceptance policy rather than hiding the original failures. Validate Large and the remaining trace/reset/position-sensitive cases.
- Rerun all correctness gates on the destination GPU. Complete at least 1,000,000 checked RUN commands, adversarial resets, finite/idle exits, repeated output values, restart and clean STOP for persistent variants. Add explicit lifecycle and sequence-exhaustion regression coverage.
- Complete M0 transport measurements: GPU-initiated acknowledgements, backoff, empty launch/graph, pinned copy, direction/poll/publication analysis, CPU use, repetitions and before/after noise controls. Only the CPU-initiated tight-poll smoke has passed so far.
- Run Compute Sanitizer and bounded Nsight Systems/Compute profiles. Verify actual registers/spills/occupancy/traffic, cooperative grid legality and finite replay behavior. No sanitizer or profiler run is complete.
- Tune the strongest graph and CPU baselines, sweep legal persistent block counts and evaluate resident-weight staging only with supporting resource/profile evidence. Current implementation streams weights; there is no resident-weight experiment yet.
- Add the PyTorch eager/compile/explicit CUDA Graph benchmark and identify its actual math/compiler backend. This comparison is not implemented yet.
- Validate and then run the complete Small/Main/Large, window, precision, queue/load/burst, idle/power and CPU crossover studies. Final conditions require at least 100,000 measured events and five runs with varied order. No final performance comparison or speedup claim is supported yet.
- Add controlled affinity/NUMA/clock comparisons with restored settings, matched telemetry and uncontaminated runs where possible. Current records are shared-GPU/default-clock diagnostics.
- Finish experiment provenance: exact commit plus dirty-state identification, complete environment/control/clock metadata, startup costs, error/sample summaries, telemetry analysis and plots. The native runner currently records `.git/HEAD` text rather than a resolved immutable commit hash; fix this before final evidence.
- Finish architecture/transport/measurement documentation, `LAB_NOTEBOOK.md`, analysis/report, reproducible demonstration and release checklist. The `docs/` directory is currently empty. `bench/study.py` is a runner, not the finished experiment notebook or report.

## Direction for the next implementation session

Proceed in this order: **numerical gates → transport/lifecycle stress → baseline tuning → profiles → final studies → report**. Keep the existing model and cache contract; use the same primitives for custom graph and persistent paths. Inspect the mixed failures before changing tolerances. Retain unfavorable results and separate shared-GPU smoke results from isolated final comparisons.

First rebuild and rerun the checks below on a GPU with sufficient free memory. Then address the unresolved items above in milestone order. Do not launch the entire study matrix until correctness is accepted and the runner/provenance have been checked.

## Transfer

From the current server, create a portable archive including sources, fixtures and raw results, excluding machine-specific environments and build artifacts:

```bash
tar -C /root \
  --exclude='vishal-cuda/.venv' \
  --exclude='vishal-cuda/build*' \
  --exclude='vishal-cuda/.pytest_cache' \
  --exclude='*/__pycache__' \
  -czf /tmp/vishal-cuda.tar.gz vishal-cuda
scp /tmp/vishal-cuda.tar.gz USER@SERVER:~/
ssh USER@SERVER 'tar -xzf ~/vishal-cuda.tar.gz -C ~'
```

Replace `USER@SERVER`. The archive preserves approximately 794 MiB of uncompressed fixtures and all existing results. The virtual environment and CUDA binaries must be rebuilt on the destination. Git was initialized, but no commit has been created.

## Destination setup and checks

Use a driver-compatible CUDA toolkit; the tested toolkit was 12.5.82 with driver 550.163.01. A newer server may use a different supported combination, which must be recorded. The project CMake defaults contain this server's CUDA/GCC paths; pass explicit compiler paths if the destination differs.

```bash
cd ~/vishal-cuda
sudo apt-get update
sudo apt-get install -y cmake ninja-build g++-12 libeigen3-dev nlohmann-json3-dev libgtest-dev libssl-dev
# Install a compatible NVIDIA CUDA toolkit separately if nvcc is absent.
# Install uv if it is absent.
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pytest matplotlib
uv pip install --python .venv/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build -j 6
build/stream-probe results/destination-hardware.json
.venv/bin/python scripts/environment.py --output results/destination-environment.json
.venv/bin/python -m pytest -q
ctest --test-dir build --output-on-failure
.venv/bin/python scripts/validate.py --output results/destination-validation
# Expect the currently unresolved Main mixed hidden-state failures to remain
# until the numerical investigation is resolved.
```

Run the independent mixed investigation from the repository root as a module:

```bash
.venv/bin/python -m scripts.diagnose_mixed
```

The script currently writes `results/mixed-reduction-diagnosis.*`; preserve the existing records before replacing them if testing a new implementation.

After correctness is accepted, the stress runner is available:

```bash
.venv/bin/python scripts/validate.py --stress 1000000 --output results/destination-stress
```

It runs the fixture matrix first, then Main persistent one-/four-block stress in both numerical modes. Do not treat an unsuccessful subprocess as a completed gate.
