# Experiment notebook

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
