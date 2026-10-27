# Persistent CUDA Runtime for Streaming Transformer Inference

Implementation brief v1.1 — 26 October 2026

Target amendment — 27 October 2026: the user selected the available NVIDIA RTX PRO 5000 Blackwell (48 GB, SM120) for final results, replacing the original A6000 and interim A5000 targets. Build natively for SM120 with CUDA 13.0. Preserve the model, numerical modes, cache semantics and measurement gates below. References to A6000/SM86 in the original brief describe the initial hardware scope; final claims must identify the actual Blackwell hardware. Historical Ampere results are separate from final Blackwell evidence.

## 1. Abstract

Build a C++/CUDA inference runtime that processes an incoming numeric event with a compact causal transformer and returns one score. The runtime retains each layer's keys and values between events. Its main execution backend is a persistent GPU kernel: a program launched once that waits for new inputs, executes the complete transformer forward pass, and publishes each result.

Evaluate the same model through conventional CUDA kernels, CUDA Graphs, and persistent execution. The primary metric is host input-ready to host output-visible p99 latency. Also measure sustainable arrival rate, queueing, CPU usage, GPU resource use, and numerical error. The engineering contribution is the runtime, its GPU work distribution and synchronization, and a reproducible explanation of when persistence helps or loses.

This is a compact-model inference systems project. Random, frozen weights are suitable for numerical and execution benchmarks; their outputs must not be presented as useful financial predictions. Supporting a trained checkpoint uses the same export interface and does not require changing the runtime.

## 2. Objective and success criteria

Answer one question: can persistent execution reduce the latency of a complete streaming transformer on the NVIDIA RTX PRO 5000 Blackwell (SM120) compared with a strong CUDA Graph implementation?

Deliver:

1. A fully specified streaming transformer and a trusted reference.
2. Working C++ backends for conventional CUDA, CUDA Graphs, and persistent execution.
3. An optimized CPU reference baseline.
4. Correct cache updates, reset, shutdown, and long-stream execution.
5. Reproducible latency distributions and an experiment notebook.

The performance goal is a repeatable p99 improvement over the strongest comparable GPU baseline on at least one predefined workload. A 20% improvement is a useful working target, not a promised result or a reason to discard unfavorable runs. Treat the CPU/graph/persistent crossover curves as headline results: report where each backend wins, with the same host-visible output requirement. Include the power cost of persistence rather than treating a spinning GPU as a free optimization.

Persistence and transformer attention are existing ideas. Claim ownership of the implementation and demonstrated findings; do not claim a new attention algorithm or assume a publication-level contribution.

## 3. Fixed initial scope

Use one NVIDIA RTX PRO 5000 Blackwell 48 GB GPU. Compile explicitly for SM120 with the installed driver-compatible CUDA 13 toolkit. Keep benchmark GPU execution exclusive and CPU baseline runs separate. Retain A6000/A5000 SM86 records as historical diagnostics; do not split this small model across GPUs.

The core workload is one ordered event stream. Every input requires one output and advances model state. There is no wait to collect a batch. This is a defined streaming workload, not an assertion that Jump uses batch size one. Test sustained and bursty input arrivals so the workload is not artificially restricted to low traffic.

Include one architecture, fixed tensor shapes per run, bounded attention caches, FP32 and one mixed-precision mode, and a native C++ command-line interface.

Defer multiple concurrent streams, general model compilation, LLM text generation, speculative decoding, quantization below FP16, distributed execution, custom allocators, exchange-feed parsers, web serving, GDRCopy, and Hopper-specific features. These are not prerequisites.

## 4. Model contract

### Default model

| Setting | Value |
|---|---|
| Input | One normalized FP32 vector of 32 features |
| Output | One FP32 score |
| Transformer layers | 2 |
| Hidden width | 128 |
| Attention | 4 query/key/value heads; head dimension 32 |
| Attention history | At most 256 positions, including the current event |
| Feed-forward network | SwiGLU, intermediate width 256 |
| Normalization | Pre-norm RMSNorm, epsilon 1e-5; final RMSNorm |
| Position encoding | RoPE, base 10,000, adjacent coordinate pairs, absolute event positions |
| Bias/dropout | No linear biases; dropout disabled |
| Parameters | 332,544 |
| FP16 weights / KV state | About 0.634 MiB / 0.250 MiB, excluding workspace and metadata |

For each event, execute input projection; then, in every layer, RMSNorm, Q/K/V projection, RoPE, KV-cache insertion, attention, output projection with residual addition, RMSNorm, SwiGLU feed-forward, and residual addition. Finish with RMSNorm and a linear score head, with no sigmoid. For RoPE, coordinate pair i uses angular frequency 10,000 raised to (-2i / head_dimension); compute the angle from the absolute event position. Rotate in FP32 before the configured output cast.

The small footprint is intentional: it exposes launch, synchronization, cache, and execution overheads. Do not describe this as a large-language-model serving engine.

### Streaming semantics

Let event index t start at zero. At each layer:

1. Compute Q, K, V for the current event only.
2. Rotate Q and K using absolute position t.
3. Write the new K and V into slot t modulo W.
4. Attend to positions max(0, t-W+1) through t.
5. Preserve historical K/V exactly as computed when those events arrived.

Store K and V separately with layout [layer, head, window_slot, head_dimension]. Each layer owns its own cache. Store the event position in device memory; never move the whole cache to append an event. Reset invalidates old positions and returns the position to zero.

This is a stateful transformer with local causal attention at every layer. It is not equivalent to independently recomputing the latest W raw inputs from an empty state. Earlier information can remain in higher-layer representations after a cache entry is evicted.

Build two matching references: an explicit PyTorch streaming step and a full-sequence PyTorch forward pass with the same per-layer local causal mask. Do not use a one-row causal mask blindly when queries and cached keys have different lengths; construct or validate the allowed positions explicitly.

### Numerical modes

Start with FP32 weights, activations, KV state and accumulation. Disable TF32 for correctness comparisons.

Then add a separately named mixed-precision configuration: FP16 weights, intermediate activations and KV state; FP32 dot-product accumulation, normalization reductions, softmax and final score accumulation. Specify every cast boundary in the reference. Round K/V before attention if the runtime stores them in FP16. Compare execution backends within the same numerical mode.

Use stable softmax with maximum subtraction. Keep aggressive fast-math disabled until its errors and latency have been evaluated separately.

### Predefined scaling cases

| Case | Layers | Width | Heads | FFN width | Window | Parameters | FP16 weights | FP16 KV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small | 1 | 64 | 4 | 128 | 64 | 43,264 | 0.0825 MiB | 0.0156 MiB |
| Main | 2 | 128 | 4 | 256 | 256 | 332,544 | 0.6343 MiB | 0.2500 MiB |
| Large | 4 | 256 | 8 | 512 | 512 | 2,632,192 | 5.0205 MiB | 2.0000 MiB |

Main is the performance target. Freeze a Small reference checkpoint too, so it can support an early persistent-execution proof. Evaluate both single-block and multi-block Main; Large reveals scaling limits after those work. Persistent execution does not imply that all weights fit in registers. Measure actual register allocation, shared memory, spills and cache traffic.

## 5. Languages, libraries and tools

| Component | Choice | Purpose |
|---|---|---|
| Native runtime | C++20 | Model ownership, buffers, submission, timing, benchmark CLI |
| GPU implementation | CUDA C++ | GEMV, attention, fusion and persistent execution |
| GPU APIs | CUDA Runtime API | Streams, memory, events, graphs, cooperative launches |
| Synchronization | Cooperative Groups and libcu++ cuda::atomic_ref | Block/grid coordination and documented mailbox ordering |
| Linear algebra baseline | cuBLAS / cuBLASLt | Strong conventional GPU matrix-operation baseline |
| Model reference | Python 3.12 and PyTorch | Reference, frozen checkpoints, torch.compile and export |
| CPU baseline | Eigen3 | Optimized native FP32 matrix/vector operations |
| Serialization | Python exporter + binary tensors + nlohmann/json manifest | Fixed, explicit model loading without LibTorch |
| Build | CMake and Ninja | Native build and CTest integration |
| Correctness | GoogleTest and pytest | Kernel, streaming-state and backend comparisons |
| Analysis | NumPy and Matplotlib | Error metrics, percentile analysis and plots |
| Profiling | Nsight Systems, Nsight Compute, NVTX | Timeline, resource and bottleneck evidence |
| GPU checking | Compute Sanitizer | Memory, initialization and synchronization checks where supported |

Use a driver-compatible CUDA toolkit, with CUDA 13.0 as the proposed starting version, and a supported host compiler. At milestone 0 record and lock the exact working driver, toolkit, PyTorch, compiler and dependency versions. CUDA Runtime interoperability is not dependent on the PyTorch C++ ABI because the first interface is a standalone executable.

Triton, FlashInfer, CUTLASS, TensorRT and pybind11 are not required for the first release. The persistent backend needs device-callable routines you own; ordinary host cuBLAS calls cannot simply be placed inside it. A Python binding can follow the native implementation.

## 6. Runtime design

Preallocate model weights, one state object, KV caches, scratch activations, and communication buffers. There must be no allocations, logging, file I/O or model loading inside the measured event path.

Use common numerical primitives exposed through ordinary kernel wrappers and device functions. This allows the graph and persistent variants to share implementation rather than comparing unrelated algorithms.

### Single-block Main and grid-size experiments

Implement a single-block Main variant as well as the cooperative variant. It streams weights from the memory hierarchy and uses block-local synchronization; it does not require all weights to fit within one SM. Main performs about 925,952 multiply/add FLOPs at a full 256-position cache, excluding scalar nonlinearities and normalization overhead. This is a workload-size estimate, not a latency prediction.

Sweep legal block counts from {1, 4, 8, 16, detected_SM_count}, deduplicating and filtering against actual launch/resource limits. An all-SM grid is one experiment, not the default architecture. For one block, use block-local coordination; for multiple blocks, use the validated cooperative design.

Run the sweep with weights streamed first. Then investigate the smallest legal grid that can retain the intended weight tiles without spills. Include activations, KV workspace, per-thread register limits and compiler-generated allocation in the footprint calculation. Do not infer residency from summed register/shared-memory capacities, or assume N blocks occupy N distinct SMs. Compare the additional memory traffic of streaming against the barriers and resource pressure of residency.

For the multi-block persistent backend, initially use a fixed cooperative grid and explicit phase barriers. Partition projection and FFN output rows across blocks; assign attention heads or head/context tiles to blocks. Begin with a simple schedule and use profiling to justify changes. Keep FP32 partial reductions and document which block owns each output.

Before launch, query cooperative-launch support and compute the legal grid size from the actual kernel's occupancy and dynamic shared memory. Every participating block must be able to reside concurrently. All participating threads must take the same barrier sequence. A homemade atomic grid barrier is deferred until the correct cooperative implementation exists and a measured need justifies it.

### Host/GPU handoff

First implement and stress-test a ping-pong transport without a model. Its result is a baseline for this transport, not an absolute hardware lower bound.

The low-latency transport uses a tiny mapped pinned mailbox with separate, aligned request and completion sequence words and separate input/output storage. Use an explicitly supported system-scope atomic_ref load/store protocol; do not assume volatile is sufficient, and do not require CPU/GPU read-modify-write atomics on a platform that does not support them. Check the exact allocation, operation and memory-order guarantees of the installed toolkit before using this backend.

Protocol:

1. CPU writes input and command, then publishes the request sequence with release ordering.
2. GPU coordinator observes the new sequence with acquire ordering, stages input into device workspace and coordinates the grid.
3. GPU computes the model, publishes the output, then releases the completion sequence.
4. CPU acquires the completion sequence and reads the output.

There is one outstanding command per state. Separate RUN, RESET and STOP commands. A repeated score, zero, infinity or NaN must never be treated as a completion signal. Bound sequence counters and test reinitialization before wraparound.

If the target platform cannot provide the required mailbox guarantees, retain the ordered-copy/event transport as the correctness baseline and implement a documented supported transport before publishing persistent host-latency results. Do not silently emulate an unsupported synchronization primitive.

For controlled C++ comparisons, use the same legal transport wherever possible; record transfer differences explicitly. Compare against a conventional pinned-copy path too, since mapping is not automatically faster.

### Transport direction and polling experiments

Separate request polling, output publication, and complete round-trip experiments. Include CPU-initiated and GPU-initiated round trips with their acknowledgement paths stated explicitly. Compare tight polling against a bounded backoff policy, including its effect on traffic, CPU/GPU utilization, response tails and power. Keep the memory-order guarantees identical.

Treat expensive request reads and cheap completion writes as a hypothesis, not a universal PCIe property. GPU timing of a posted write is not the time at which the CPU observes it. Use a single clock domain per measurement; publish absolute one-way leg times only if clock correlation is established. Do not obtain one-way latencies by halving a round trip.

### Bounded execution and profiler modes

The hardware probe must query cudaDevAttrKernelExecTimeout and record display attachment, OS and driver mode. Prefer a headless Linux GPU, but verify its actual timeout policy rather than assuming headless means unlimited runtime. Do not automatically change watchdog/TDR settings or switch driver modes.

Add a finite command mode that exits after N processed events, as well as explicit STOP handling. Keep the bounded launch duration within any active limit, including waits for inputs; a count limit alone does not bound wall time. Use a documented idle/time budget or a self-contained finite fixture when needed.

For Nsight Compute, use application replay for finite host-fed runs, or a separately labeled finite device-only trace mode with inputs preloaded on the GPU. Ordinary kernel replay can omit the host responses needed by an interactive kernel; merely exiting after N events does not fix that. Device-only profiling measures compute/state handling and is not a replacement for host-latency measurements.

## 7. Implementation milestones

Estimate 10–12 weeks at approximately 20 hours/week, with buffer if the communication or persistent kernel requires redesign. Complete each gate before moving on.

### M0 — Hardware and communication baseline: week 1

Create the build, hardware probe and benchmark metadata schema. Record CPU model, GPU architecture, SM count, register/shared-memory limits, PCIe link, NUMA topology, clocks, driver/toolkit, execution-timeout policy and cooperative-launch capabilities.

Implement empty launch, graph replay, pinned-transfer and mailbox ping-pong measurements. Validate request IDs and payload checksums across at least 100,000 exchanges. Save latency samples, CPU usage and the chosen protocol.

Run the initial noise-control comparison and direction/backoff transport tests described in section 9. As optional external calibration, spend at most half a day attempting NVIDIA's standalone ping-pong and small-model examples on SM120, if they build with the available stack without introducing architecture-specific dependencies outside this scope. Record the exact commit and timing boundaries. This is an external reference experiment, not an LSTM backend in this project, and cannot block the transformer milestones. Do not compare different model latencies as if the workloads were identical.

Exit: a supported transport works reliably, and its costs are understood. This runtime uses conventional and cooperative CUDA on SM120; Hopper thread-block clusters are not required.

### M1 — Model contract and oracle: weeks 2–3

Implement Main and the Small test configuration in PyTorch, including explicit per-layer local causal attention and the streaming ring buffer. Cross-check streaming against full-sequence masked execution before CUDA work.

Export a frozen seed-controlled checkpoint, inputs and expected intermediate outputs. Use a JSON manifest with tensor names, dimensions, dtypes, byte offsets, version and hashes, plus contiguous binary tensor data. Fix linear-weight orientation to row-major [out_features, in_features].

Implement deterministic bounded traces: random normal inputs, zeros, constant values, impulses, alternating signs, and repeated patterns. Check reset boundaries and cache wraparound. These are execution fixtures, not a forecasting-quality claim.

Exit: two independent references agree and a concrete file contract exists. Training a useful forecasting model is a later consumer of this interface, not a prerequisite for testing the runtime.

### M2 — Conventional native implementation: weeks 4–5

Implement model loading, the C++ CPU/Eigen step, and a conventional GPU path using cuBLAS for projections and FFNs with custom normalization, RoPE, ring-cache update and attention kernels.

Attention starts with a simple stable implementation that handles the entire allowed history and the current event correctly. Compare every layer, cache entry and score to the oracle, including multiple cache revolutions. Make warm-up, reset and stop explicit.

Capture the existing cuBLAS-based sequence in an initial CUDA Graph using fixed buffers and device-held state. This brings a graph comparison into the early release without waiting for custom matrix/vector routines.

Exit: publish v0.1 with correct CPU, conventional GPU and initial graph execution, the transport study, oracle checks and raw timings. This is the first usable project release, targeted for the end of week 5.

### M3 — Own CUDA primitives and graph backend: weeks 6–7

Implement device-callable matrix/vector, normalization, RoPE, attention, activation and residual routines, initially with global-memory weights and workspace. Validate them separately from fusion.

Target a bounded single-block persistent Small proof by the end of week 6, using the same transport and a checked trace. This is an early feasibility checkpoint, not the million-command completion gate. It is acceptable to start in FP32 without resident weights.

Capture the same CUDA sequence into a graph using fixed buffer addresses. Keep position, valid length and ring indices in device state. Graph replay must not capture a stale event index or add allocations as the cache grows.

Benchmark PyTorch eager, torch.compile, explicit PyTorch graph capture when supported, native cuBLAS, native custom kernels, and the custom CUDA Graph path. Verify the actual attention/compiler backend and record it.

Exit: the graph baseline is correct and reasonably tuned. Isolate the effect of custom math from the effect of graph replay.

### M4 — Complete fused and persistent execution: weeks 8–10

Extend the Small proof to a complete one-block Main forward pass, then a cooperative multi-block Main variant. Before keeping each variant alive, run its fused forward pass once per event; this makes synchronization and numerical debugging easier. Apply the legal grid-size sweep instead of assuming the largest grid is fastest.

Extend the correct fused implementation into a loop that waits for RUN/RESET/STOP. Start from global-memory weights. Evaluate resident weight tiles only after measuring footprint and occupancy. Do not promise that all model weights stay in registers.

Gate: at least 1,000,000 checked RUN commands without stale/duplicate completion or state drift outside the documented tolerance. Include adversarial reset timing and clean shutdown. Run relevant sanitizers, while recognizing that they do not prove cross-CPU/GPU protocol correctness.

Exit: one long-lived GPU program performs the entire streaming transformer and returns correct host-visible scores.

### M5 — Focused optimization and evidence: weeks 11–12

Change one mechanism at a time: block count, rows per block, attention work assignment, fusion boundary, register/shared staging, or a genuinely unnecessary barrier. Recheck correctness and capture the profile that explains the result.

Run Small/Main/Large and window sweeps. Repeat the final comparisons in both numerical modes. Compare the strongest tuned graph baseline, the persistent backend and the optimized CPU path.

Run the default-clock versus controlled-clock idle experiment, and report latency/power tradeoffs alongside the CPU/GPU crossover. A reduction caused by keeping clocks elevated must be distinguished from reductions in launches, memory traffic or barriers.

Finish the notebook, raw results, profiles, plots, limitations and a short demonstration. If persistence loses for a case, retain that result and explain the crossover.

### Release cut lines

| Checkpoint | Publishable scope | Claims supported only after measurement |
|---|---|---|
| v0.1, target week 5 | Transport study, two references, CPU/CUDA/initial graph runtime | Implemented a correct streaming transformer and characterized its execution/communication costs |
| Small proof, target week 6 | Bounded single-block persistent Small | Demonstrated persistent execution and validated state/communication on the Small configuration |
| v0.2, after M4 | Single-block and cooperative Main, stress gate | Implemented complete persistent streaming inference; report actual error and latency |
| v1.0, after M5 | Grid, crossover, load and power studies | State the measured improvement and precisely identify winning and losing workloads |

At week 6, review feasibility and remaining effort. If the Small proof is not correct, publish the completed v0.1 work and time-box diagnosis before expanding to Main. This is a scope-control checkpoint, not an automatic calendar-based abandonment rule. A partially working persistent kernel must not be described as a completed runtime.

## 8. Correctness gates

Check linear operations, RMSNorm, RoPE, stable attention and SwiGLU individually; then compare complete layer states and scores.

Required cases include first event, partially filled cache, exactly-full cache, first eviction, many revolutions, reset, startup/shutdown, and unchanged consecutive output scores. Track correct cache-slot ownership and positions at every layer.

Use FP64 calculations on small fixtures as an additional numerical oracle. Initial normalized-output tolerances may start at atol=1e-5/rtol=1e-4 for FP32 and atol=2e-3/rtol=2e-2 for mixed precision, but approve tolerances only after inspecting error distributions and operation-level behavior. Record maximum absolute error, relative L2 error and non-finite counts. Never relax a tolerance just to admit an unexplained failure.

For very long streams, include position-sensitive cases and report error growth. Cache-history equivalence must be checked separately from the arithmetic tolerance of a single forward pass.

## 9. Benchmark contract

### Three distinct measurements

1. Device execution: inputs/state already on the GPU; diagnose compute and synchronization.
2. Host service latency: normalized input available in CPU memory until the corresponding result is readable by the CPU. Include handoff, input staging, all model operations and output publication.
3. Arrival latency under load: intended event-arrival time until host-visible output, including queueing and injection delay.

Exclude model loading, allocation and compilation only after reporting their startup cost separately. Normalization/raw market-data feature extraction lies outside the defined input boundary; do not call this exchange-feed-to-decision latency. Any future work moved off the event path must be counted in sustainable throughput and backlog tests.

### Replay methodology

Use one submitting worker and an independent scheduled producer for load tests. Pre-generate the arrival schedule. Retain intended arrival, actual enqueue, submission and completion timestamps. Do not postpone scheduled arrivals merely because the previous event was slow.

Use a bounded, preallocated queue and report overflow, dropped inputs, injector delay and deadline misses. Overload must be visible; do not report attractive percentiles while silently dropping events. For comparative runs, choose common absolute arrival rates anchored to a named reference backend's measured capacity, around 25%, 50%, 75%, 90% and above that capacity. Use identical scheduled traces across compared backends; do not label different absolute arrival rates as the same workload merely because each is 25% of a different backend's capacity. Add a bursty trace at matched mean load and separate backend-specific saturation curves.

Use the same input trace, state resets, checkpoint, precision and output requirement across backends. Tune CPU thread count on the small model and record it. The matched FP32 GPU/CPU comparison is separate from a mixed-precision GPU comparison.

Collect at least 100,000 measured events per final condition after warm-up, repeated in at least five separate runs with backend order varied. Use longer runs for p99.9 claims. Save individual samples or lossless chunked samples. Report p50/p95/p99/p99.9, worst observed value, service rate, queueing, error, CPU use and GPU power where readable.

Time host endpoints using one monotonic host clock. Do not subtract host and GPU timestamps without clock correlation. Device stage instrumentation belongs in a separate build; quantify and remove its overhead for final latency runs. Do not add the p99 values of separate stages and present the sum as end-to-end p99.

### Noise controls and idle/power hypothesis

Apply controls incrementally and save the uncontrolled measurements too:

- Pin the submitting worker and arrival producer to separate physical CPU cores; avoid placing them on sibling hardware threads.
- Establish NUMA-local CPU and memory policy before allocating/first-touching host buffers, and verify locality where the OS exposes it.
- Record the CPU governor, turbo policy and GPU clocks. Use a performance governor or supported GPU clock locks only where allowed, save the original settings, and restore any settings changed for an experiment.
- Match warm-up, thermal steady state and cache policy. Record other GPU activity and temperature; do not treat requested clock locks as proof that actual clocks stayed fixed.
- Measure empty-transport and Main latency variability before and after the controls, with the same repetitions. Report this observed timing variability rather than an absolute physical noise floor.

At low offered load, test whether a spinning persistent kernel avoids idle clock/power transitions that affect the graph backend. Compare graph versus persistent under default dynamic clocks and, where supported, under matched fixed clocks. Include the first event after increasing idle gaps and a common low-load scheduled trace. Record average idle/load power, actual clock telemetry and CPU use in long runs; keep telemetry overhead comparable across backends.

The expected effect is a hypothesis. If the advantage changes with clock control, quantify that contribution without assuming it explains every difference. A persistent compute kernel is distinct from the driver's persistence-mode setting.

### Profiler interpretation

Use Nsight Systems for host submissions and conventional GPU timelines; Nsight Compute for register use, spills, occupancy and memory traffic. A persistent kernel hides internal stages from an ordinary kernel timeline, so use deliberate device-side instrumentation to study its barriers and phases.

Small weights/KV may be served from L2. Do not calculate model-weight bytes divided by time and label it achieved DRAM bandwidth without actual traffic evidence. Record cache state and distinguish cache-resident steady-state workloads from cold-start diagnostics.

## 10. Repository layout and commands

Suggested directories:

    reference/       PyTorch model, streaming oracle, export and trace generation
    include/         ModelConfig, ModelState, Runtime, transport and weight format
    src/cpu/         Eigen baseline
    src/cuda/        Primitives, conventional wrappers, graph and persistent backends
    src/runtime/     Ownership, communication, reset, shutdown
    bench/           Native runner, scheduled replay and result schema
    tests/           Kernel, model-state and transport tests
    configs/         small.json, main.json, large.json
    results/         Raw records, environment manifests and summaries
    docs/            Architecture, measurement contract and final analysis
    LAB_NOTEBOOK.md  Hypotheses, experiments, failures and explanations

Implement a compact CLI contract:

    stream-bench --config configs/main.json --backend graph --weights model/ --trace trace.bin
    stream-bench --config configs/main.json --backend persistent --blocks 1 --weights model/ --trace trace.bin
    stream-bench --config configs/main.json --backend persistent --blocks 8 --max-events 1000 --weights model/ --trace trace.bin
    stream-bench --config configs/main.json --backend cpu --weights model/ --trace trace.bin

Add explicit modes for correctness, service latency, scheduled replay, ping-pong and finite device-only profiling. Results must identify commit, environment, model/config/checkpoint hashes, numerical mode, backend, block count, transport, poll policy, clock/control regime, warm-up and sample count.

## 11. Evidence that makes the project interview-ready

Keep one notebook entry per meaningful experiment: hypothesis, baseline, single change, numerical check, timing distribution, profile explanation and next decision. Preserve failed attempts.

The final report should answer:

- Why did persistence help or lose relative to a graph?
- Where did latency move as the model and window grew?
- What limited weight residency and block count?
- How was cross-block and CPU/GPU ordering established?
- What happens under bursts, reset and shutdown?
- At what model sizes did CPU execution win?
- Did any low-load benefit come from different clock/power behavior, and what did it cost in power?

Publish a working repository, one frozen model fixture, correctness commands, raw results, an architecture explanation and a brief demo. Final résumé claims must use measured numbers and identify hardware/workload. No STAC audit, profitable trading model or universally fastest runtime is implied.

## 12. References and their role

These references inform the design; none establishes that Jump uses this runtime.

- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html): graph execution and reduced submission overhead.
- [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) and [execution-control requirements](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXECUTION.html): grid synchronization and legal cooperative launch sizing.
- [Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html): architecture guidance; query actual SM120 resource attributes and occupancy rather than assuming data-center Blackwell limits. The [Ampere guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) applies to historical SM86 results.
- [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/): application replay for host-dependent kernels, clock/cache controls and profiling limits.
- [CUDA runtime device attributes](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html): query the execution-timeout attribute rather than relying on display attachment alone.
- [CCCL memory model](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html) and [atomic_ref](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html): allocation/scope/operation guarantees for communication.
- [NVIDIA low-latency reference](https://github.com/NVIDIA/dl-lowlat-infer): communication and measurement patterns. The model is LSTM-specific; use it as a reference, not as a transformer implementation.
- [NVIDIA capital-markets post](https://developer.nvidia.com/blog/achieving-single-digit-microsecond-latency-inference-for-capital-markets/): timing-boundary context; its reported precomputation is excluded from critical-path timing.
- [Cohere megakernel](https://github.com/cohere-ai/cohere-megakernel) and [design write-up](https://cohere.com/blog/megakernels): complete-transformer task scheduling. Its implementation is H100-specific, not an SM120 starting binary.
- [mann-uofg/infer](https://github.com/mann-uofg/infer): an example of numerical checks, raw results, profiling and limitations.
- [Jump AI](https://www.jumptrading.com/ai-ml): relevance of real-time inference and hardware constraints; not evidence of specific internal model or kernel choices.

## 13. Review decisions

| Review suggestion | Decision |
|---|---|
| Single-block Main and grid sweep | Added. Compare streaming and measured residency; no assumed winner or mandatory all-SM grid. |
| CPU competitiveness | Strengthened an existing requirement into a headline crossover result. |
| Watchdog and finite profiling | Added timeout probing, bounded execution and separate profiling modes. A finite event count alone is not sufficient for host-dependent kernel replay. |
| Noise controls | Added incremental CPU/NUMA/clock controls and an uncontrolled comparison; no system changes are required without support/permission. |
| Split transport directions | Added component and direction-initiated round trips/backoff; omitted unverified universal read/write-cost claims and uncorrelated one-way timings. |
| Idle-GPU hypothesis | Added a clock-controlled latency/power experiment, with identical offered workloads. |
| NVIDIA calibration / extra LSTM cell | Optional, capped standalone reference run only. No LSTM cell is added to our runtime. |
| Early release and week-six cut line | Added v0.1 graph capture by week 5 and a Small feasibility checkpoint by week 6. Rejected an automatic stop solely because a date was missed. |
| CMake/JSON/style conflict | Retain the approved C++20/CMake/JSON stack. Cooperative launch does not itself require CMake; build/test integration justifies that choice. Do not rewrite unseen CLAUDE.md files or import unspecified style rules. |

The model, cache semantics, languages and core deliverables remain unchanged. The first implementation task is M0: a hardware probe and correct ping-pong transport.
