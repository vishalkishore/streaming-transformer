# AGENTS.md

## Project

Build the Persistent CUDA Runtime for Streaming Transformer Inference described in [the implementation plan](deliverables/streaming_transformer_implementation_plan.md).

The final target is one NVIDIA RTX PRO 5000 Blackwell, SM120, selected by the user on 27 October 2026. The original plan used an A6000 and the earlier handoff used an A5000, both SM86; retain their results as historical diagnostics. Implement a compact stateful transformer with conventional CUDA, CUDA Graph and persistent execution backends, plus an optimized CPU baseline. Preserve the model, numerical modes, cache semantics and measurement boundaries in the plan.

Follow the current user request. This file defines coding conventions and working practices; the implementation plan defines the model and experiments. Implement the current milestone rather than expanding the project into a general inference framework.

## Style reference

Use AayushBade14's crochet repository as the primary C++ style reference, with CUDA_Softmax for readable kernel structure and TinyRaster for explicit resource ownership and small component files.

The repositories do not use one uniform style. The rules below deliberately normalize their conventions for this project. They are an adaptation, not the author's official style guide.

Observed conventions to retain:

- Short, concrete component names.
- PascalCase classes and component files.
- camelCase methods in crochet.
- Two-space indentation in crochet and CUDA_Softmax.
- Straightforward loops and explicit computation stages.
- Separate headers and implementations for host components.
- Small purpose comments explaining each subsystem or kernel phase.

Use crochet's attached-brace style consistently; do not mix in TinyRaster's next-line braces. crochet uses both mName and m_name members; choose mName consistently here.

## C++ and CUDA formatting

- Use two spaces for indentation. Do not use tabs.
- Keep opening braces on the declaration/control line and closing braces on their own line.
- Use braces for conditional and loop bodies, including one-line bodies.
- Use one statement per line. Short trivial accessors may stay on one line.
- Put a space after commas and around binary operators.
- Use compact control syntax such as if(condition){ and for(int i = 0; i < size; i++){.
- Attach pointer/reference markers to the type: float* data, const ModelConfig& config.
- Use nullptr, explicit constructors when appropriate, const methods for observers, and initializer lists.
- Use #pragma once in project-owned headers.
- Include a component's own header first in its implementation. Include dependencies directly.
- Do not add using namespace std to headers or shared implementation files.

Example of the chosen class layout:

    #pragma once

    #include <cstdint>

    namespace Stream {
      class ModelState{
        public:
          void reset();
          std::uint64_t getPosition() const;

        private:
          std::uint64_t mPosition = 0;
      };
    }

Implementation style:

    #include "ModelState.hpp"

    namespace Stream {
      void ModelState::reset(){
        mPosition = 0;
      }

      std::uint64_t ModelState::getPosition() const{
        return mPosition;
      }
    }

Use Stream as the project namespace. Add a small nested namespace such as Stream::Cuda where it makes ownership or device code clearer; do not build a deep namespace hierarchy.

## Names and files

| Item | Convention | Examples |
|---|---|---|
| Classes, structs, enums | PascalCase | Runtime, ModelConfig, KvCache, Backend |
| Host methods and functions | camelCase | loadModel, getPosition, runTrace |
| Public data fields and locals | camelCase | hiddenSize, validLength, outputScore |
| Private data members | m + PascalCase | mPosition, mWindowSize, mDeviceInput |
| Enum values and named constants | UPPER_SNAKE_CASE | PERSISTENT, MAX_WINDOW_SIZE |
| CUDA kernels and device helpers | snake_case | update_cache, attention_step, reduce_sum |
| Local host/device buffer pointers | h_ / d_ when useful | h_input, d_input, d_output |
| Host headers/sources | PascalCase.hpp / PascalCase.cpp | Runtime.hpp, Runtime.cpp |
| Device headers/sources | PascalCase.cuh / PascalCase.cu | Attention.cuh, Persistent.cu |

Short mathematical/kernel names such as i, row, head, slot, idx, q, k and v are fine in a small clear scope. Name counters, capacities, positions and ownership explicitly.

Do not add redundant type prefixes such as m_iHiddenSize, cryptic aliases such as ll, or names such as AbstractInferenceExecutionManager. Use std::size_t for byte sizes and an explicit fixed-width type for protocol counters.

Keep the plan's lowercase top-level directories, including reference, include, src/cpu, src/cuda, src/runtime, bench, tests and configs. Do not reorganize the project to imitate a graphics engine's directory tree.

## Comments

- Begin a nontrivial component with a short purpose comment when its responsibility is not obvious.
- Explain tensor layouts, units, state transitions, numerical choices and synchronization requirements.
- Introduce meaningful computation phases with brief comments.
- Explain why a barrier or memory order is necessary and which producer/consumer it connects.
- Keep routine code readable through names; do not narrate every assignment.
- Avoid decorative banners, large ASCII headers, stale commented-out implementations and placeholder documentation.
- Do not add author lines from the reference repositories to this project's files.

Prefer a comment such as "Publish after every block has finished writing the score" over "Set flag to one".

## Design and ownership

Use small host classes for resources with lifetimes and plain structs for configurations, kernel arguments and device-visible state. Prefer direct functions for numerical operations.

Runtime, model state, timers and benchmark state must have explicit ownership. crochet's use of singletons is not a requirement here: do not create global GPU objects, global mutable model state or singleton managers for every subsystem.

Use RAII for CUDA buffers, streams, events and graph handles. Make unique resource owners noncopyable; add move support only when required. Shut down and join the persistent kernel before freeing anything it can access. Destructors must not throw.

Raw pointers are appropriate for non-owning tensor views and device interfaces. Make ownership and element counts visible at the interface. Keep byte counts separate from element counts.

Use standard containers and existing approved libraries where they simplify code. Prefer constexpr to dimension macros. Use templates for dtype or fixed kernel shapes when useful; avoid generic factories, plugin registries, virtual interfaces and metaprogramming without a concrete need.

Do not create empty wrappers or duplicate the whole runtime into versioned experiment directories. Share primitives across backends; keep experiment IDs and differences in LAB_NOTEBOOK.md.

## CUDA implementation

- Write computation stages explicitly and keep tensor layout/index calculations easy to inspect.
- Put host orchestration in host code and device-callable primitives in .cuh/.cu files.
- Ordinary host cuBLAS calls cannot be called from inside the persistent kernel.
- Check CUDA/cuBLAS API errors and launch errors. Report asynchronous failures at synchronization boundaries.
- A small checked-call macro/helper is allowed; do not hide allocation, synchronization or control flow in elaborate macros.
- Allocate weights, state, scratch and communication buffers before the measured path.
- Keep logging, file I/O, allocation and per-event configuration parsing out of that path.
- Keep FP32 accumulation/reduction behavior and cast boundaries consistent with the reference. Do not enable fast-math or change precision silently.
- Use block-local synchronization for the one-block variant and validated cooperative synchronization for the multi-block variant.
- Compute legal launch sizes from actual kernel occupancy and shared memory. An all-SM grid is an experiment, not a default.
- Use the documented aligned atomic_ref load/store protocol and memory scope for the mailbox. volatile alone is not synchronization.
- Use sequence IDs for completion. A score value is never a completion sentinel.
- Keep bounded profiling execution and RUN/RESET/STOP handling available.

Preserve per-layer KV ring buffers and device-held absolute positions. Test valid length, eviction, wraparound and reset. Do not substitute independently recomputed windows for stateful local attention, or blindly apply a one-row causal mask to a longer KV cache.

## Build and dependencies

Keep the approved stack: C++20, CUDA, CMake, Ninja, cuBLAS/cuBLASLt, Eigen3, nlohmann/json, GoogleTest, Python/PyTorch, pytest, NumPy and Matplotlib. Earlier bans on CMake or JSON libraries from another brief do not apply to this project.

Use target-based CMake with explicit source lists, include paths, compile features and linked dependencies. Compile CUDA for architecture 120 for the final Blackwell target; architecture 86 remains available for the previous Ampere target. Do not replace the build system or write a new JSON parser merely to imitate a reference repository.

Intended build commands once the native project is bootstrapped:

    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
    cmake --build build
    ctest --test-dir build --output-on-failure

Use the installed driver-compatible toolkit and record exact versions. Do not claim that these commands have passed until they have actually run.

## Python reference and analysis

Use normal Python conventions: four-space indentation, snake_case functions/variables, PascalCase classes, explicit imports and pathlib for paths.

Keep the reference easy to audit. Use separate full-sequence masked and streaming implementations, fixed seeds and explicit dtype/cast rules. Export documented tensors and metadata.

Put reusable reference, export and analysis logic in Python modules. Notebooks may explain experiments but must not be the only place an essential algorithm or benchmark exists.

## Validation and performance evidence

Read the current milestone and neighboring code before editing. Make the smallest complete change that satisfies the task. Avoid unrelated refactors.

Validate numerical primitives before fusion, then complete layer/cache states and outputs. Add regression checks for changed indexing, state, ordering or numerical behavior. Do not weaken tolerances to conceal unexplained errors.

Compare conventional, graph, persistent and CPU paths using the same model, input trace, reset semantics and numerical mode. Preserve the host-service, device-only and scheduled-arrival timing boundaries in the plan.

Record hypotheses, before/after measurements, actual compiler/profiler settings, raw results, failures and explanations. Keep default-clock results and permitted noise-control comparisons. Do not silently drop overloaded events, add separate stage p99 values, or call cached model traffic measured DRAM bandwidth.

Run relevant CPU tests and supported GPU checks. If hardware or profiler permissions prevent a check, identify exactly what was not run; do not fabricate results. State measured improvements only for the tested configuration.

## Inspected style sources

Inspected on 26 October 2026:

- [crochet Timer.hpp](https://github.com/AayushBade14/crochet/blob/master/CrochetEngine/Core/Timer/Timer.hpp) and [Timer.cpp](https://github.com/AayushBade14/crochet/blob/master/CrochetEngine/Core/Timer/Timer.cpp): component files, namespaces, indentation and methods.
- [crochet WindowManager.hpp](https://github.com/AayushBade14/crochet/blob/master/CrochetEngine/Platform/WindowManager/WindowManager.hpp) and [Shader.cpp](https://github.com/AayushBade14/crochet/blob/master/CrochetEngine/Graphics/ShaderManager/Shader.cpp): member names and direct method implementation.
- [CUDA_Softmax shared-memory version](https://github.com/AayushBade14/CUDA_Softmax/blob/main/02_shared_memory_version/softmax_shared.cu): explicit stages, kernel naming and h_/d_ buffer names.
- [TinyRaster Framebuffer.h](https://github.com/AayushBade14/TinyRaster/blob/main/Framebuffer.h): component responsibilities and explicit resource lifetime.
- [TinyRaster CMakeLists.txt](https://github.com/AayushBade14/TinyRaster/blob/main/CMakeLists.txt): CMake is present in the inspected repositories; it is not inherently in conflict with their coding style.

The examples and project rules above are written for this runtime. Observed formatting conventions do not require copying the reference implementations' algorithms, global-state patterns or measurement methods.
