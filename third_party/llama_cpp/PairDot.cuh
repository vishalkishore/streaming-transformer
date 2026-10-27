// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: MIT
// Adapted FP32-accumulating pair-load loops; see README.md and LICENSE.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace Stream::Cuda {
  // Caller guarantees even row strides and aligned float2/half2 addresses.
  // Each lane accumulates adjacent pairs in FP32, with no half arithmetic.
  __device__ inline float pair_dot(const void* weight, const float* input, int row, int cols, bool mixed){
    int lane = threadIdx.x % 32;
    const float2* x = reinterpret_cast<const float2*>(input);
    float sum = 0;
    if(mixed){
      const __half2* w = reinterpret_cast<const __half2*>(weight) + row * (cols / 2);
      for(int pair = lane; pair < cols / 2; pair += 32){
        float2 a = __half22float2(w[pair]);
        float2 b = x[pair];
        sum += a.x * b.x;
        sum += a.y * b.y;
      }
    }else{
      const float2* w = reinterpret_cast<const float2*>(weight) + row * (cols / 2);
      for(int pair = lane; pair < cols / 2; pair += 32){
        float2 a = w[pair];
        float2 b = x[pair];
        sum += a.x * b.x;
        sum += a.y * b.y;
      }
    }
    return sum;
  }
}
