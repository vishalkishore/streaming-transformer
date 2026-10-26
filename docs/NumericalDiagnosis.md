# Main mixed numerical investigation

The initial Blackwell checks used `atol=2e-3, rtol=2e-2` for mixed tensors and `atol=1e-5, rtol=1e-4` for FP32. The observations below preserve those original diagnostics. The subsequent candidate end-to-end policy, rejected intermediate rules and approval gates are documented in [NumericalPolicy.md](NumericalPolicy.md); FP32 and local primitive thresholds remain unchanged.

The initial native SM120 matrix reproduced the handoff result: 27/36 conditions passed (all Small conditions and Main FP32), while all nine Main mixed conditions failed on raw hidden residual coordinates. Scores and complete KV snapshots passed. These are correctness diagnostics, not performance measurements.

`stream-diagnose` executes the actual Eigen CPU implementation with optional operation snapshots. `scripts.diagnose_native` compares those snapshots against streaming FP32 and FP64 references using identical weights, cast boundaries, inputs and resets. It records operands, pre-cast linear/norm results, post-cast projections, RoPE outputs, attention, the attention residual, SwiGLU activation, final hidden states and the score. Snapshot work is excluded from timing runs.

The first isolated input-projection cast disagreement is event 0, coordinate 88:

| Quantity | Value |
|---|---:|
| FP64 dot with identical inputs/weights | -0.02931978205688779 |
| Adjacent FP16 values' midpoint | -0.02931976318359375 |
| Distance to midpoint | 1.887329403871263e-8 |
| Native FP32 pre-cast | -0.02931980788707733 |
| Reference FP32 pre-cast | -0.029319748282432556 |
| Native FP16 result | -0.029327392578125 |
| Reference FP16 result | -0.0293121337890625 |

The two FP32 reductions fall on opposite sides of a rounding midpoint. Subsequent FP16 casts can propagate this discrete difference through the stateful model. The FP32 versus FP64 reference pair also fails two raw layer-1 hidden coordinates on the Main trace. This establishes that strict elementwise agreement is sensitive to legal reduction differences; it does not by itself prove every native error is harmless.

The diagnostic additionally recomputes each of the 16 native linear operations in FP64 using that operation's captured native operands. This removes upstream state divergence from the comparison. For n inputs, it checks the conservative FP32 dot-product bound `gamma(2n) * sum(abs(x_i * w_i))`, where `gamma(k) = k*u/(1-k*u)` and `u=2^-24`. In the recorded 777-event Main mixed trace there were zero violations; the largest observed error/bound fraction was about 0.0289. This bound is a local arithmetic diagnostic, not a replacement end-to-end tolerance.

An early trace regression compared two independently allocated Eigen states for bit identity and failed. Different buffer alignments can change Eigen's reduction partition. The corrected regression replays the same Small/Main inputs and resets with the same CPU state allocation, once without snapshots and once with snapshots. Scores, every layer's hidden states and final complete caches must match exactly. The failure log is retained as `results/blackwell-ctest-trace-alignment-failure.log`.

Evidence is in `results/blackwell-mixed-diagnosis/{report.json,native/manifest.json,native/stages.bin}`. The manifest records hashes of weights, trace, snapshots, source tree and executable. The report preserves failing coordinates, maximum errors, relative L2 and absolute-error quantiles for all captured stages.

Subsequent work added actual CUDA operation snapshots and common-operand checks for Main/Large: `results/completion-gpu-diagnosis` and `results/completion-large-gpu-diagnosis`. The 16 Main and 30 Large linears met the FP32 dot bound; isolated normalization, activation and residual checks passed. Dedicated CPU/CUDA RoPE checks passed absolute positions through 1,000,000. End-to-end candidate-policy and held-out validation remain separate gates; no million-command or final performance result follows from operation checks alone.
