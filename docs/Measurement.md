# Measurement contract

The model consumes a normalized FP32 feature vector and produces one FP32 score. This is not exchange-feed-to-decision latency. Loading, allocation and compilation are outside steady-state samples and reported as startup costs. GPU performance paths compute no CPU model oracle; CPU baselines and GPU conditions run serially.

| Measurement | Start | End | Included |
|---|---|---|---|
| Host service | CPU normalized event ready for submission | Corresponding score visible to CPU | Input staging/transfer, complete forward, synchronization/publication |
| Scheduled arrival | Intended arrival on the monotonic host clock | Corresponding score visible to CPU | Injector delay, bounded queue waiting and service |
| Device only | Preloaded trace/device state | Finite device forward loop complete | GPU compute/state work; conventional paths also include their documented D2D staging |

One monotonic host clock supplies intended/enqueue/submission/completion timestamps. CUDA events measure device-only execution. No host/GPU timestamp subtraction or sum of stage p99 values substitutes for end-to-end latency.

## Samples, resets and offered load

Final conditions require at least 100,000 measured completions after warmup and at least five separate runs with backend order varied. Individual samples are saved as CSV. p50/p95/p99, diagnostic p99.9, worst value, rate and CPU time are retained. Short tuning, operation diagnostics and correctness runs are labeled separately and cannot support final speedup claims.

All backends use hash-identified weights/configurations/input traces, the same numerical mode and reset policy. Warmup is followed by reset before measured service runs. Separate FP32 and mixed comparisons are mandatory; compare CPU and GPU in matched FP32 as well.

Scheduled replay uses an independent producer pinned to a separate physical core when allowed. It does not wait for inference completion before scheduling another event. Its preallocated bounded queue exposes overflow/dropped IDs, injector delay, queueing and deadline misses. Percentiles of completed events must always be shown with offered/completed/drop counts. A run with fewer than the required completions cannot pass the final sample gate, even if it offered 100,000 inputs.

Common absolute rates are anchored to a named measured graph capacity at approximately 25/50/75/90/110 percent, with identical schedules across backends. Bursts group arrivals at matched mean offered rate. Backend-specific saturation curves are labeled separately; equal percentages of different backend capacities are not a common workload.

## Controls, telemetry and limits

Retain default-clock/uncontrolled measurements before affinity/NUMA/clock comparisons. Record physical core topology, affinity, first touch, memory policy, governor, turbo policy, actual GPU clocks, temperature, other activity and power where available. Requested controls are not evidence that they took effect. Restore settings changed for an experiment.

The current environment permits worker affinity but rejects NUMA binding (`Operation not permitted`), exposes read-only governor settings and denies Nsight Compute performance counters (`ERR_NVGPUCTRPERM`). These restrictions must be present in the final limitations. No privileged driver reload or interference with other GPU jobs is authorized or required.

Telemetry uses the same interval and collector on CPU and GPU conditions. `bench.study.run` records wall/monotonic clock correlation around each subprocess so raw host sample boundaries can select the corresponding telemetry interval. Power/clock summaries must state their sample count and exclude compilation/startup when analyzing steady state.

`python -m bench.analyze PREFIX... --output results/analysis.json --plot`
recomputes nearest-rank percentiles from raw CSV, retaining offered/completed/drop
counts and deadline failures including drops. It rejects correctness and device
runs as host performance evidence. Telemetry summaries cover the measured host
interval, preserve unavailable metrics, and reject correlation after a wall-clock
offset change above 5 ms. Reported power is a sampled mean with its sample count
and observed span; it is not an energy integral or a complete idle-power study.
The generated CDF describes completed events and labels the dropped fraction.

For a complete study, use `python -m bench.summarize results/final-study --output
results/final-summary.json`, then `python -m bench.plot_study
results/final-summary.json --output results/figures`. Summary generation verifies
the saved raw-artifact hashes and retains archived/failed attempts. Each center
is the median of separate-run p99 values; error bars are their minimum/maximum,
not a confidence interval. Paired comparisons preserve individual winning and
losing runs. Missing repetitions, short runs, changed implementation identities,
null commits or increased drops prevent a repeatable-improvement candidate.
That candidate flag still requires the baseline/noise-control and release audit.

The `noise` study phase compares default-clock uncontrolled and affinity runs
with the same seven transport/poll conditions plus fixed four-block Main graph
and persistent paths. It does not claim NUMA/governor controls were applied.
Transport runs retain an explicit warmup prefix in raw CSV; the summary discards
it before computing host percentiles. Host wait/ack observations and GPU full
round-trip cycles remain separate; cycles are not converted using an assumed
fixed frequency. Transport CPU usage includes that warmup, and its lack of
per-event monotonic timestamps prevents steady-state telemetry correlation.
Model runs use their ordinary reset-after-warmup boundary.

Default-clock low-load and idle-gap runs test a hypothesis: a resident spinning kernel may change clock/power state and first-event latency. Where supported and restorable, matched clock-control runs are needed to separate clock effects from launch, traffic or barrier effects. Without that evidence, do not attribute an observed difference exclusively to persistence or clock behavior.

## Provenance and profiling

The study stamp hashes the resolved native shared-library files. PyTorch
conditions record interpreter and source hashes, resolved commit/dirty state,
actual Torch/CUDA versions and compiler configuration. Their resume identity
includes the uv-reported installed package set, in addition to the frozen lock
file and benchmark sources. A stamp identifies a run; it does not prove its
correctness or sufficient repetitions.

Resume also verifies hashes of both raw CSV and JSON artifacts. Missing or changed
artifacts cause a new attempt; earlier files and attempt records are archived
before rerunning. Final studies require the native 54-case/four-stress gate and
the complete sequential native/adversarial/window/PyTorch/sanitizer followup,
including current PyTorch source and numerical settings.

Every native result records resolved commit, dirty status, source snapshot SHA256 and executable SHA256, as well as config/checkpoint/trace hashes, precision, backend, transport, block count, polling policy, warmup and count. A null commit is not final immutable provenance. Resuming a study must check executable and command identity and must not reuse failed/incomplete results.

Nsight Systems captures bounded timelines. Nsight Compute uses application replay for host-fed execution or a labeled finite preloaded trace; ordinary kernel replay does not recreate host acknowledgments. Profiler timings are not final host latency measurements. Compiler resources/CUDA occupancy are not dynamic occupancy or memory-traffic counters. Cache capacity does not prove cache hits, and model bytes divided by time are never called measured DRAM bandwidth.
