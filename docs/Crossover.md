# Why the graph currently wins

The current implementation has no demonstrated persistence-winning crossover.
This is a result for our streamed-weight runtime on the RTX PRO 5000 Blackwell,
not a general claim about persistent kernels. Scalar GEMV and staged attention
remain the defaults. No CPU inference executes during these GPU measurements.

## What the implementation establishes

The graph and persistent backends share numerical primitives, FP32 reductions,
cast boundaries and stateful ring-cache semantics. The graph orders phases using
kernel dependencies; persistence calls a cooperative grid barrier between phases.
For staged attention, `forward()` executes `17 * layers + 4` barriers: 21 Small,
38 Main and 72 Large. The finite preloaded loop adds one input-staging barrier per
event. The live RUN path adds four protocol/input/completion barriers around the
forward pass. These are source counts, not measured barrier durations.

Graphs submit a captured sequence rather than making a separate host submission
for every numerical kernel. Persistence therefore competes against graph replay,
while paying its own grid synchronization and live CPU/GPU mailbox costs. It
still reads weights and scratch through the memory hierarchy; this implementation
does not keep weight tiles in shared memory between events. Removing launches
alone does not establish lower latency.

The current GEMV assigns one output row per warp. A Main 128-row projection
therefore gives row work to 128 of the 440 warps in a 110-block grid, while every
cooperative block must reach the phase barrier. This illustrates phase imbalance;
it is not a measured utilization percentage for the complete model. Graph
kernels also have inactive warps at this grid size, but do not retain the whole
cooperative grid across subsequent stages.

The current executable's persistent kernel has 78 registers/thread, 32 local bytes
and 19,584 shared bytes/block, including the optional attention implementation.
The same resources apply when staged attention is selected through runtime
branching. Compiler resources can differ from the separate graph kernels.
Register pressure, inter-phase waiting and available arithmetic parallelism are
plausible contributors; unavailable hardware counters prevent assigning an exact
fraction of the gap to each. Local bytes are a resource attribute, not measured
spill traffic. No DRAM-bandwidth claim follows from these tests.

## Bounded experiments

Reproduce the initial size survey with:

```sh
uv run --no-sync python -m scripts.check_crossover
uv run --no-sync python -m scripts.check_crossover --sizes small main --blocks 4 --output results/crossover-survey-b4
uv run --no-sync python -m scripts.check_crossover --sizes small main --blocks 16 --output results/crossover-survey-b16
uv run --no-sync python -m scripts.check_crossover --sizes main-w32 main-w512 --output results/crossover-survey-windows
```

All 72 initial 1k-event conditions completed with valid sampled exclusivity
sidecars and the same benchmark executable SHA256
`f682dc713f82711dd991650851ef20940bca9e2bd108b11f49bf88c3b47a3194`.
The initial sweep covers Small/Main/Large, both precisions, both backends and both
timing boundaries at 110 blocks; Small/Main at 4/16 blocks; and Main windows
32/512 at 110 blocks. These are fixed grids for diagnosis, not proof that each
backend is fully tuned. All initial pairs favor the graph. Reducing to 4/16
blocks makes both backends slower than 110 blocks on Small/Main, consistent with
loss of useful parallelism outweighing any smaller-grid synchronization benefit.
It does not isolate barrier latency.

The follow-up has **three clean varied-order 10k repetitions per condition**,
72 accepted runs. Median per-run host-service p99 (us), without pooling samples:

| Model | Precision | Graph | Persistent | Persistence penalty |
|---|---|---:|---:|---:|
| Small | FP32 | 96.139 | 120.008 | 24.8% |
| Small | Mixed | 101.089 | 124.458 | 23.1% |
| Main | FP32 | 184.387 | 221.246 | 20.0% |
| Main | Mixed | 195.326 | 224.996 | 15.2% |
| Large | FP32 | 466.961 | 508.921 | 9.0% |
| Large | Mixed | 474.291 | 514.010 | 8.4% |

Preloaded device mean penalties range from 7.2% to 13.2%. The smaller relative
service gap on Large is an observed trend, not evidence of a future crossing.
No independent workload dimension or untested model size is extrapolated.

![Three clean 10k repetitions](../results/crossover-survey-10k-figure.png)

Accepted records: `results/crossover-survey-10k/accepted-summary.json`.
Audit: `results/crossover-survey-10k-figure.json`. All 72 sidecars are valid and
share the executable identity above. The first repetition accidentally overlapped
a 3.46-second Python suite including CPU reference inference. Its entire 24-run
repetition is excluded, despite valid GPU-only process evidence; it is preserved
with `measurement-exclusion.json`. Repetitions 1/2 plus a separate replacement
repetition 3 supply the three accepted runs. Tests finished before the replacement.
This illustrates why GPU process visibility alone is insufficient to enforce
CPU/GPU measurement separation.

Reproduce the accepted ordering directly with:

```sh
uv run --no-sync python -m scripts.check_crossover --events 10000 --runs 3 --start-repeat 1 --output results/crossover-reproduction
uv run --no-sync python -m scripts.plot_crossover results/crossover-reproduction/summary.json --output results/crossover-reproduction-figure
```

At 110 blocks, single-run 1k host-service p99 values (microseconds):

| Model | Precision | Graph | Persistent |
|---|---|---:|---:|
| Small | FP32 | 98.448 | 120.247 |
| Small | Mixed | 105.488 | 125.078 |
| Main | FP32 | 203.766 | 234.235 |
| Main | Mixed | 203.806 | 241.195 |
| Large | FP32 | 498.399 | 538.788 |
| Large | Mixed | 499.829 | 546.408 |

Main window 32 gives graph/persistent p99 168.316/199.526 us FP32 and
168.167/204.826 us mixed; window 512 gives 228.305/258.635 us FP32 and
228.325/262.694 us mixed. Persistence also loses in the preloaded device boundary,
so live host signaling cannot explain the entire observed gap.

![Initial short size, grid and window survey](../results/crossover-survey-figure.png)

The figure/audit are generated by `scripts/plot_crossover.py`; it rejects invalid
exclusivity evidence, active CPU oracles, drops and mixed executable identities.
The process monitor samples once per second and cannot rule out every transient
client. Raw commands, CSVs, JSON, telemetry and process observations are retained
beside each summary in `results/crossover-survey*`.

## Timing boundaries and interpretation

Device mode records a CUDA-event average over a finite preloaded trace starting
with empty caches. Persistence uses a finite fused loop, with no mailbox polling;
graph mode includes per-event D2D input staging and graph launches. Event time can
include device idle gaps if host enqueue cannot keep up. Service mode measures
host input-ready to host output-visible completion after 256 warm-up events and
a reset. Graph uses pinned-copy transport; persistence uses mapped transport.
These are different experiments, not additive components. In particular, do not
subtract a device mean from service p99 to call the difference mailbox overhead.

Warm-up/reset means initial cache fill is included in both measured traces;
1k-event Large runs spend a significant fraction filling its 512-position cache.
The size survey changes layers, width and window together, so it cannot identify
an independent causal effect for each dimension. The separate Main window cases
hold architecture fixed and use documented checkpoint variants.

## Remaining crossover work

The three varied-order 10k repetitions remain bounded diagnostics. Final claims
still require tuned legal grids including 1/8, final
100k x five host-service studies, matched CPU comparisons, scheduled arrivals,
low-load idle/power costs and an actual resource-validated resident-tile
experiment. A finite global-memory-weight loop losing does not rule out benefits
from resident weights or a different task schedule. Retain an explicit 'no
crossover found in tested conditions' result if those experiments also lose.

Implementation guidance from primary engineering write-ups is collected in
[PersistentDesignResearch.md](PersistentDesignResearch.md). The comparison
distinguishes our current staged runtime from overlapped task-based megakernels.
