# Implementation references inspected on 27 October 2026

These are design references, not measured results for this runtime. No external
implementation has been installed, benchmarked or incorporated by this review.
Our model, casts, ring-cache semantics, sequence-ID protocol and host-visible
measurement boundary remain authoritative.

| Reference | Useful implementation detail | Important difference |
|---|---|---|
| [NVIDIA dl-lowlat-infer](https://github.com/NVIDIA/dl-lowlat-infer) | Persistent host-fed inference, ping-pong transport, weight residency and small/medium/large work placement. Inspect `src/cpp/nvLstmInf`, especially `ss_gemv_impl`, `ModelPersistentBase.cuh` and `PingPongModelPersistent.cuh`. | LSTM rather than transformer. Its documentation includes Blackwell architecture builds; this does not establish compatibility or latency for our model. |
| [Cohere megakernel](https://github.com/cohere-ai/cohere-megakernel) and [design explanation](https://cohere.com/blog/megakernels) | Full transformer decode in a persistent kernel, common task interface, warp specialization and dependency-driven scheduling. | The repository explicitly describes its tested target as H100 SM90a. Its task graph and serving system are much larger than this project. |
| [Mirage Persistent Kernel](https://github.com/mirage-project/mirage/tree/mpk) | GPU task scheduling and task-level profiling inside a megakernel. | A compiler/runtime framework, with multi-GPU facilities; adopting it would expand our fixed implementation scope. |
| [Blackwell microGPT implementation](https://github.com/CG-8663/talos-vs-macbook-vs-gx10/blob/main/bench_cuda_persistent.cu) | Small single-warp implementation with shared-memory weights, KV state, cooperative reductions and one persistent launch. | Its source explicitly excludes host round trips from timing. Its 4,192-parameter model is much smaller and has different operators. The [cuda-samples discussion](https://github.com/NVIDIA/cuda-samples/issues/418) is a proposal, not an accepted NVIDIA sample. |
| [mann-uofg/infer](https://github.com/mann-uofg/infer) | Concrete strong graph baseline: device-held position, vectorized GEMV, fused normalization/residual/SwiGLU, split-KV attention and correctness diagnostics. Inspect [`gemv.cu`](https://github.com/mann-uofg/infer/blob/main/src/kernels/gemv.cu) and [`decode.cu`](https://github.com/mann-uofg/infer/blob/main/src/kernels/decode.cu). | Llama-sized BF16/quantized workloads, a different RoPE pairing and numerical implementation. Its use of explicit FMA/fast exponentials cannot silently replace our current arithmetic. |

The [NVIDIA article](https://developer.nvidia.com/blog/achieving-single-digit-microsecond-latency-inference-for-capital-markets/)
also describes precomputation and an output-value timing signal. For this runtime,
precomputation must be counted in sustainable service capacity, and a floating
score must never replace a completion sequence. GDRCopy is outside the fixed
scope. Neither its LSTM numbers nor GPU-only microGPT throughput predicts our
host-service latency.

Our next experiments should first improve and profile the numerical primitives,
then investigate shared/register staging where the actual footprint permits it.
Warp softmax and distributed history reduction address observed serial work;
QKV/gate-up fusion and fewer genuinely unnecessary barriers are later candidates.
Any staged-weight or task-scheduling experiment needs its own resource,
correctness and before/after timing evidence. Keep the cooperative barrier
baseline until an alternative has a measured justification and a proven ordering
protocol. Replacing it directly with another repository's atomic barrier would
skip those requirements.
