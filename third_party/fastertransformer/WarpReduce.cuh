/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Adapted from NVIDIA FasterTransformer; see README.md for revision and changes.
#pragma once

#include <cuda_runtime.h>

namespace Stream::Cuda {
  // All 32 lanes must participate with the same mask. Every lane receives the result.
  __device__ inline float warp_all_sum(float value){
    #pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1){
      value += __shfl_xor_sync(0xffffffff, value, mask, 32);
    }
    return value;
  }

  __device__ inline float warp_all_max(float value){
    #pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1){
      value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, mask, 32));
    }
    return value;
  }
}
