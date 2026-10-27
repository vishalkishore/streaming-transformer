# NVIDIA FasterTransformer excerpts

Source: https://github.com/NVIDIA/FasterTransformer/blob/df4a7534860137e060e18d2ebf019906120ea204/src/fastertransformer/kernels/reduce_kernel_utils.cuh

Pinned upstream commit: `df4a7534860137e060e18d2ebf019906120ea204`.
Upstream source SHA256: `103806b48e55fe5a74958ea086936fb001b47a59d7a357ace7abc94d4d3e24e9`.
License: Apache-2.0, copyright 2020–2023 NVIDIA CORPORATION; see LICENSE.
Upstream has no root NOTICE file at this revision.

Adapted `warpReduceSum` and `warpReduceMax` only. Changes: FP32 specialization,
project namespace/names/formatting, built-in float addition instead of the type
wrapper, and fmaxf for the FP32 maximum. No upstream dependencies are required
beyond CUDA. Full 32-lane participation is required; ragged tensors use neutral
values in inactive tensor lanes, not partially participating warps.

The XOR tree produces the result in every lane. RMS normalization and softmax
use it to eliminate the subsequent lane-zero broadcast. Row-dot reductions
retain the previous down-shuffle tree. No precision, cast or cache policy is
changed. Numerical and performance evidence is recorded in LAB_NOTEBOOK.md.
