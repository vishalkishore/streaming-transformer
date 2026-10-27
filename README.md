# Streaming transformer CUDA runtime

A stateful local-causal transformer with an Eigen CPU baseline, conventional cuBLAS/custom CUDA, CUDA Graph, fused and persistent execution backends. Final benchmark target: NVIDIA RTX PRO 5000 Blackwell, SM120 (selected by the user on 27 October 2026). Earlier A5000 SM86 results remain historical diagnostics.

**Work in progress:** an earlier 54-condition native Small/Main/Large matrix and
36 independent reference trace conditions passed. New experiments have their
own bounded checks documented below. Mixed acceptance follows the
investigated [numerical policy](docs/NumericalPolicy.md), retaining original raw
failures. Million-command persistent stress, the additional GPU validation
matrices and final performance studies remain open. No speedup is claimed.

Read [HANDOFF.md](HANDOFF.md) for verified status, known failures, setup, transfer commands and the next steps. The full contract is [the implementation plan](deliverables/streaming_transformer_implementation_plan.md); coding conventions are in [AGENTS.md](AGENTS.md).

Build and check from the repository root after installing the approved native
development dependencies and a driver-compatible CUDA 13 toolkit:

```bash
uv sync --frozen
cmake -S . -B build-blackwell -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-blackwell -j 6
uv run --no-sync python -m pytest -q
ctest --test-dir build-blackwell --output-on-failure
uv run --no-sync python -m scripts.demo
```

The Small frozen fixtures are included. Large expected-state archives, generated
results, build trees and virtual environments are excluded from Git. Regenerate
and verify the larger fixtures before their validation:

```bash
uv run --no-sync python -m scripts.fixtures --sizes small main large
uv run --no-sync python -m scripts.fixtures --sizes small main large --verify-only
uv run --no-sync python -m scripts.prepare_windows
```

The complete expected cache histories are large; all window variants require
substantial disk space. The demo checks correctness and does not compare speed.
The latest CPU-side suite passed 43 tests; final GPU acceptance is tracked
separately in the handoff.

Read the [architecture](docs/Architecture.md), [mailbox ordering](docs/transport.md),
[measurement contract](docs/Measurement.md), [Blackwell setup](docs/BlackwellSetup.md),
[experiment notebook](LAB_NOTEBOOK.md) and [release checklist](docs/ReleaseChecklist.md).
Raw local result paths in these documents describe the working experiment archive;
they are not downloadable artifacts in this source snapshot. The final report
and published measurement archive are still pending.

CUDA warp all-reductions reuse [NVIDIA FasterTransformer code](third_party/fastertransformer/README.md)
under its preserved Apache-2.0 license.

An optional `--matvec vector` experiment adapts MIT-licensed
[llama.cpp pair loads](third_party/llama_cpp/README.md). Scalar remains the default;
the first Main comparisons showed no gain. GPU launchers now wait for an empty
GPU and record competing-process checks. See the
[source comparison](docs/RepositoryComparison.md) for results and limitations.

`--attention fused` selects the optional shared-memory tiled attention
experiment. It passed bounded model/state and sanitizer checks but is slower
than staged attention on Main; staged remains default. See
[Attention.md](docs/Attention.md) for measurements and the cold-start fix.
