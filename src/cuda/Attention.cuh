#pragma once

#include "../../third_party/fastertransformer/WarpReduce.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace Stream::Cuda {
  // Single-chunk attention for the plan's short windows, with already rotated Q/K.
  // Original implementation informed by FlashInfer's vectorized decode structure;
  // retain expf and explicit normalized FP32 probabilities instead of online exp2.
  struct AttentionArgs{
    const float* query;
    const void* keys;
    const void* values;
    float* output;
    int heads;
    int dim;
    int window;
    int layer;
    std::uint64_t position;
    bool mixed;
  };

  __device__ inline float2 load_attention_pair(const void* cache, int index, bool mixed){
    if(mixed){
      return __half22float2(reinterpret_cast<const __half2*>(cache)[index / 2]);
    }
    return reinterpret_cast<const float2*>(cache)[index / 2];
  }

  // Cooperatively stage up to 128 chronological rows. Padding avoids the
  // bank conflicts of a 32-float stride when threads read different rows.
  __device__ inline void stage_attention_tile(const AttentionArgs& a, const void* cache,
                                               float tile[128][33], int head, int firstSlot,
                                               int start, int count){
    int shift = a.dim == 32 ? 4 : 3;
    int pairsPerRow = a.dim / 2;
    for(int pair = threadIdx.x; pair < count * pairsPerRow; pair += blockDim.x){
      int row = pair >> shift;
      int col = 2 * (pair & (pairsPerRow - 1));
      int slot = firstSlot + start + row;
      if(slot >= a.window){ slot -= a.window; }
      int base = ((a.layer * a.heads + head) * a.window + slot) * a.dim;
      float2 value = load_attention_pair(cache, base + col, a.mixed);
      tile[row][col] = value.x;
      tile[row][col + 1] = value.y;
    }
  }

  // Requires head dimension 16 or 32, window <=512 and aligned cache pairs.
  // All 128 threads participate; each block owns whole heads, extra blocks idle.
  __device__ inline void fused_attention(const AttentionArgs& a){
    __shared__ float probabilities[512];
    __shared__ float partial[4][32];
    __shared__ float tile[128][33];
    __shared__ float query[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int valid = int(a.position + 1 < a.window ? a.position + 1 : a.window);
    int firstSlot = (a.position + 1 - valid) % a.window;
    // valid <= window, so chronological indexing needs at most one subtraction.
    for(int head = blockIdx.x; head < a.heads; head += gridDim.x){
      if(threadIdx.x < a.dim){ query[threadIdx.x] = a.query[head * a.dim + threadIdx.x]; }
      for(int start = 0; start < valid; start += 128){
        int count = min(128, valid - start);
        stage_attention_tile(a, a.keys, tile, head, firstSlot, start, count);
        // Publishes both the query and K tile before each thread computes a row.
        __syncthreads();
        if(threadIdx.x < count){
          float dot = 0;
          for(int i = 0; i < a.dim; i++){ dot += query[i] * tile[threadIdx.x][i]; }
          probabilities[start + threadIdx.x] = dot / sqrtf(float(a.dim));
        }
        // No loader may overwrite K until every score consumer has finished.
        __syncthreads();
      }
      if(warp == 0){
        float maximum = -INFINITY;
        for(int j = lane; j < valid; j += 32){ maximum = fmaxf(maximum, probabilities[j]); }
        maximum = warp_all_max(maximum);
        float total = 0;
        for(int j = lane; j < valid; j += 32){
          probabilities[j] = expf(probabilities[j] - maximum);
          total += probabilities[j];
        }
        total = warp_all_sum(total);
        for(int j = lane; j < valid; j += 32){ probabilities[j] /= total; }
      }
      // Normalized probabilities are shared by all four V-reduction warps.
      __syncthreads();
      float sum = 0;
      for(int start = 0; start < valid; start += 128){
        int count = min(128, valid - start);
        stage_attention_tile(a, a.values, tile, head, firstSlot, start, count);
        __syncthreads();
        if(lane < a.dim){
          for(int j = warp; j < count; j += 4){ sum += probabilities[start + j] * tile[j][lane]; }
        }
        // Reuse the shared tile only after every warp has consumed its V rows.
        __syncthreads();
      }
      if(lane < a.dim){ partial[warp][lane] = sum; }
      __syncthreads();
      if(warp == 0 && lane < a.dim){
        float total = partial[0][lane];
        for(int w = 1; w < 4; w++){ total += partial[w][lane]; }
        a.output[head * a.dim + lane] = a.mixed ? __half2float(__float2half_rn(total)) : total;
      }
      // Do not reuse shared scratch for the next head until its consumer is done.
      __syncthreads();
    }
  }
}
