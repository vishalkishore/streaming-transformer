# Numerical acceptance policy

The candidate policy is `mixed-depth-rms-v3`. It changes **acceptance metrics**, not weights, computation, FP32 accumulation, FP16 cast boundaries, cache semantics or TF32 settings. Approval requires the independent arithmetic/held-out checks and complete native validation to pass. Original scalar and short-head diagnostics remain in every correctness report.

FP32 retains `atol=1e-5, rtol=1e-4` for every score, hidden coordinate and cache entry. The local primitive/common-operand tests retain their original FP32/mixed thresholds.

For an end-to-end mixed hidden vector or KV vector in zero-based layer `l`, the componentwise budget is:

```
abs(actual - reference) <= 0.002 * (l + 1) * max(1, RMS(reference_vector))
                           + 0.02 * abs(reference)
```

Each layer's complete hidden vector, complete K cache and complete V cache must also have relative L2 error at most `0.002`. For the coordinate budget, a KV vector is one `[head, slot, head_dimension]` entry; the additional energy gate covers a complete layer cache. Per-head relative errors are still reported as diagnostics. NaN/Inf always fails.

Scores retain the scalar relative coefficient `0.02`, with an absolute budget `0.002 * layers`. The aggregate score sequence must additionally satisfy:

```
RMS(score_error) / max(1, RMS(reference_scores)) <= 0.002
```

Raw maximum absolute and relative L2 errors are recorded, including when a low-energy reference makes relative L2 large. A unit RMS floor gives the aggregate score check the same absolute-error interpretation as the scalar check near zero. It prevents a nearly zero, untrained score head from turning a small absolute error into a misleading rejection.

## Evidence and rejected provisional metrics

The initial raw mixed threshold failed sparse near-zero residual coordinates on Main. Actual CPU and CUDA pre/post-cast traces show FP32 reductions on opposite sides of FP16 rounding midpoints. All 16 Main and 30 Large common-operand CUDA linears met conservative FP32 dot-product bounds. Their isolated normalization, activation and residual checks passed. The FP32/FP64 streaming pair also produces raw Large residual failures: the difference is not unique to native CUDA.

The provisional `mixed-depth-rms-v2` assumed independent error accumulation through its square-root depth factor and applied a `0.002` relative gate to each short head/slot vector. `results/acceptance-v2` disproves acceptance under that rule: all nine Large mixed conditions failed. A few low-energy head vectors exceeded that relative bound, and a residual coordinate missed its scaled absolute bound. Historical copies of the same cache entry caused repeated snapshot failures; these counts must not be presented as independent errors.

The candidate replaces the cancellation assumption with a linear depth budget for potentially correlated errors. Its independent norm gate applies to the layer tensors that own the state, while the maximum component bound and raw short-head statistics remain visible. This is an explicit tensor policy, not a claim of bit-identical arithmetic or a new analytical stability theorem.

A held-out Main checkpoint (seed 17, constant inputs) also disproved a pure relative-L2 score gate: the FP32/FP64 pair differed by at most about `0.000327`, passed all scalar absolute/relative checks, but had score relative L2 around `0.00752` because the scores were small. That rejected candidate record is retained in `results/numerical-policy-validation.*`. The unit-floor aggregate metric above is checked separately in `results/numerical-policy-confirmation.*`.

The candidate is tested with weight seeds 42, 17 and 314, random and constant inputs, multiple cache revolutions and reset, for Small/Main/Large. Wrong-slot permutations, broad drift, non-finite values and FP32 errors have explicit rejection tests. Do not adjust this policy again to admit a held-out failure: inspect the algorithm or retain the failure as an open correctness gate.

Reproduce with:

```bash
uv run --no-sync python -m scripts.verify_numerical_policy --output results/numerical-policy-confirmation.json
uv run --no-sync python scripts/validate.py --binary build-blackwell/stream-bench --sizes small main large --output results/acceptance-final --stress 1000000
```

The stress runner only proceeds when the fixture matrix passes. A provisional policy, a local operation check, or a passing short trace alone does not establish the million-command gate.
