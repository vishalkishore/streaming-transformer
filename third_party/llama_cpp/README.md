# llama.cpp pair-load adaptation

Source: [ggml-cuda/mmvf.cu](https://github.com/ggml-org/llama.cpp/blob/de5122d879d00d2895c36be7e0cee94105aeeabd/ggml/src/ggml-cuda/mmvf.cu),
commit `de5122d879d00d2895c36be7e0cee94105aeeabd`.
MIT license and copyright are preserved in LICENSE and the adapted header.

PairDot.cuh adapts only the float2 and half2 weight-load loops with FP32
accumulation. It removes ggml tensor/dispatch, fusion and multi-column interfaces;
uses our row ownership; retains explicit float multiply then add under
`--fmad=false`; and omits every FP16-accumulating path. The caller uses our warp
reduction and output cast. Pair assignment changes the summation order.

The runtime selects this experiment with `--matvec vector`; scalar remains the
default. Odd widths or insufficient pointer alignment use the scalar fallback.
No ggml build or library installation is required. Local correctness and timing
results are recorded in LAB_NOTEBOOK.md.
