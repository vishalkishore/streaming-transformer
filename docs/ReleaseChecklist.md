# Completion audit

The full implementation-plan objective is **not complete**. This checklist maps
its requirements to current evidence and remaining work. A script's existence,
queued job or short diagnostic is not a completed experiment. Historical Ampere
measurements do not establish final Blackwell performance.

| Plan requirement | Current evidence | Remaining release gate |
|---|---|---|
| Final hardware and approved toolchain (§3, §5) | Native SM120 build; hardware/environment manifests; uv environments | Freeze immutable source commit and final environment alongside final runs |
| Fixed architecture, weights, shapes and cast boundaries (§4) | Reference/native implementations; Small/Main/Large fixtures; documented numerical policy | Verify transfer package and final recorded hashes |
| Independent streaming and masked references (§4, M1) | 36/36 size/mode/family cases with reset; `results/extended-independent/summary.json` | Preserve commands, source identity and raw diagnostics in final package |
| FP64/local arithmetic investigation (§8) | Native common-operand Main/Large diagnostics; 18/18 independent numerical-policy confirmations | Retain rejected policies and original failures in report |
| CPU, cuBLAS, conventional custom CUDA and graph backends (M2–M3) | 54/54 native correctness matrix | Final tuning and matched performance comparisons |
| Shared primitives and complete fused/persistent forward (M3–M4) | Native matrix includes fused/persistent one/four blocks | Final fresh native tests and sanitizer checks |
| CPU optimization without measured-path allocation (§6) | Eigen implementation; all-six-config no-allocation test | Thread tuning, physical-core placement, matched FP32 comparisons |
| Per-layer ring state, eviction, wraparound and reset (§4, §8) | Complete fixture state snapshots in native matrix | Native adversarial families and all 120 window conditions |
| Device absolute positions and long streams (§4, §8) | Position primitive tests through 1,000,000 | Four million-command checked Main stress cases; report drift/reset evidence |
| RUN/RESET/STOP, repeated scores, exhaustion, finite/idle exit (§6, M4) | Bounded lifecycle tests and sequence-exhaustion fix | Fresh complete native suite and stress results |
| Supported aligned system-scope mailbox (§6) | Explicit load/store ordering; transport documentation | Final protocol/transport repetitions and supported sanitizer evidence |
| Direction-specific transport, backoff, empty launch/graph and copies (M0) | Transport CLI and historical diagnostics | Isolated Blackwell repetitions, CPU use, before/after noise-control comparison |
| PyTorch eager/compile/explicit graph (M3) | Benchmark implemented with explicit mixed accumulation/casts | Full 18-condition correctness matrix, actual profiler backend identification, final comparisons |
| Legal grids {1,4,8,16,SM count} (§6, M4–M5) | Occupancy checks and tuning runner | Run sweeps; preserve losing configurations and compiler resources |
| Streamed versus resident weight tiles (§6, M5) | Current weights streamed; static resource metadata | Evaluate feasible residency experiment with spills/workspace included; do not infer physical SM placement |
| Registers, spills, occupancy and traffic (§6, §9) | Compiler/CUDA resource attributes; bounded Nsight Systems capture | Supported profiles/instrumentation; explicitly record Nsight Compute counter denial |
| Distinct device/service/arrival measurements (§9) | Native modes and explicit boundaries | Collect matched final device diagnostics and host studies; quantify instrumentation overhead |
| Strongest comparable baseline and repeatable p99 result (§2, M5) | Tuning infrastructure only | Tune graph/CPU/persistent; test predefined workloads; retain wins and losses, no promised 20% result |
| 100,000 completions × five varied-order runs (§9) | Runner enforces counts; no final matrix yet | Complete final conditions and verify individual raw samples/repetition coverage |
| Common rates, bursts, queue/drop visibility (§9) | Independent producer; bounded queue; schedule regression checks | Run common absolute rates 25/50/75/90/110%, matched bursts and overload |
| Separate backend-specific saturation curves (§9) | Runner and schedule tests | Collect and plot curves without confusing them with common-rate comparisons |
| Model/window/mode/CPU crossover (M5) | Fixtures prepared for 30 window variants | Complete size/window/precision studies and plot headline crossover results |
| Affinity, NUMA, governor and clock controls (§9) | Affinity available; hard NUMA rejected; governors read-only | Before/after permitted controls, topology/locality metadata and unchanged-setting/restoration records |
| Default versus controlled clock idle/power study (§9) | Default-clock diagnostic infrastructure; no global changes | Establish supported/restorable clock controls or document exact restriction; measured low-load/gap latency, power and CPU use |
| Telemetry correlation and trustworthy percentiles (§9) | Raw analyzer; nearest-rank agreement; startup/drop/clock regression tests | Analyze final measured intervals and per-run variability; do not add stage p99s or infer DRAM bandwidth |
| Exact provenance, startup and raw artifacts (§10–§11) | Binary/source/library hashes, interpreter/package metadata, startup fields | Immutable commit, validated resume records, complete final artifact inventory |
| Architecture/transport/measurement explanations (§11) | Draft documents in `docs/` | Review against final implementation and actual measurements |
| Notebook, plots, report and brief demo (M5, §11) | Notebook, analyzer and demo script exist | Execute demo, produce final figures/report and audit seven questions below |

The final report must answer each question with measured evidence and clearly
identified limits:

1. Why does persistence help or lose relative to the strongest graph?
2. How does latency change with model size and window?
3. What limits weight residency and block count?
4. How are cross-block and CPU/GPU ordering established?
5. What happens under bursts, reset and shutdown?
6. Where does CPU execution win?
7. How much low-load benefit is associated with clock/power behavior, and what
   power cost does persistence impose?

Known restrictions are Nsight Compute `ERR_NVGPUCTRPERM`, denied hard NUMA
binding and read-only CPU governors. These restrict specific evidence; other
implementation and measurement work remains possible. No final speedup, energy
benefit or complete-release claim is currently supported.
