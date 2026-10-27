# Persistent inference design research — 27 October 2026

The current implementation is a correct staged cooperative runtime, with bounded
validation recorded elsewhere. It is not yet the overlapped task pipeline used
by the successful megakernels below. This is an optimization gap; the timing
loss itself does not identify a numerical correctness defect.

## Primary implementation write-ups

| Read | What it teaches | Application here |
|---|---|---|
| [Hazy Research: Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) | Fused instruction groups; overlapping weight loads across instructions; shared-memory page reuse; counters for actual dependencies, including chunked MLP dependencies | Our stage-by-stage grid barriers retain global waiting and our weights lack a cross-stage prefetch pipeline. Start with small fused groups before a general scheduler. |
| [Cohere: Inside the megakernel serving engine](https://cohere.com/blog/megakernels) | A porting recipe: competitive standalone operators, a common task interface, dependency counters, task placement; controller/producer/consumer roles | Use the dependency design as a reference. Its Hopper-specific task layout and parallel attention/MoE model are not a drop-in replacement for our sequential dense transformer. |
| [NVIDIA: Constant-time CUDA Graph launch](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/) | Replay batches driver work; distinguishes CPU launch overhead from device runtime and total completion time | Our baseline already captures operations. Counting removed host kernel launches overstates the remaining opportunity. The article's measurements are on a different GPU/toolkit. |
| [Hazy Research: ThunderMLA](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla) | Scheduling partial attention/reduction work, reusing schedules, and internal timing traces for loads, computation and dependency waits | Add bounded internal timing before assigning blame to individual phases. A long-context attention result does not establish a win for our 256-position window. |

The [Hazy implementation README](https://raw.githubusercontent.com/HazyResearch/Megakernels/main/README.md)
selects H100 or B200 for its demo build. That is not an SM120 support statement.
The [CUDA 13.0 Blackwell guide](https://docs.nvidia.com/cuda/archive/13.0.0/blackwell-tuning-guide/index.html)
lists a 99 KiB per-block shared-memory maximum for compute capability 12.0.
Query actual device attributes and kernel occupancy before choosing our tile
sizes; H100/B200 resource assumptions cannot be imported unchanged.

Only source/design inspection was performed in this research step. None of
these upstream engines was installed, built or benchmarked here. No upstream
code was copied. Existing pinned source/license records remain authoritative.

## What explains our current result

Measured: Main FP32 graph/persistence service p99 is 184.387/221.246 us across
three clean 10k repetitions. The preloaded means also favor graph, with no live
mailbox polling. See [Crossover.md](Crossover.md) for raw records and boundaries.

Source inspection: our Main forward has 38 cooperative phase barriers, before
live protocol overhead. Every block participates even when only some warps have
row work. Adjacent Q/K/V projections execute as separate phases despite sharing
the same normalized input; gate/up are similarly independent before activation.
Our persistent execution also shares a compiled function containing optional
attention code and its resources. Whole-kernel resources can constrain stages
that do not use those resources themselves.

Inference: we have traded graph kernel boundaries for software grid barriers
without yet gaining much cross-operation overlap or on-chip reuse. This matches
mechanisms discussed in the write-ups, but it is not a measured decomposition of
the 36.859 us service-p99 gap. Device means cannot be subtracted from host p99.
The tiny model may also offer less work to amortize scheduling than those authors'
LLM workloads. No bandwidth bottleneck has been established by hardware counters.

## Concrete implementation sequence

1. Measure a bounded, instrumented forward: capture per-block phase entry,
   computation completion and barrier exit using device timestamps. Export after
   the run. Compare instrumented and uninstrumented totals to quantify overhead;
   do not mix unsynchronized clocks across SMs or infer host/device intervals.
2. Separate experimental attention/GEMV variants at compile time where needed so
   the staged baseline does not reserve optional shared storage unnecessarily.
   Record registers, local memory and occupancy for every compiled variant.
3. Fuse small independent groups: Q/K/V projections; gate/up plus activation;
   projection plus residual where dependencies permit. Keep each existing FP16
   rounding boundary and FP32 reduction order explicit. Share the resulting
   routines between graph and persistent paths so graph remains a strong baseline.
4. Add a bounded weight-prefetch or resident-tile experiment after checking
   shared-memory limits, occupancy and storage for synchronization/scratch.
   Measure whether staging costs are amortized. Do not claim weights reside on
   a particular physical SM merely from block indexing.
5. Only then prototype producer-specific dependency signals on one complete
   layer. Define publication ordering, counter generations/reset and exact
   arrival counts. Keep the existing system-scope host mailbox protocol. Device
   dependency counters need their own validated memory-order protocol; copied
   volatile polling is not a substitute. Preserve a runnable cooperative baseline.

Validate changed primitives, then hidden/K/V/score state with wraparound and
reset. Use an empty GPU, serial CPU reference work, short 1k/10k A/B runs and
retained losing cases. A scheduler or resident design remains an experiment,
not a promised crossover. No further GPU run was needed for this research note.
