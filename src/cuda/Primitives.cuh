#pragma once

#include "Model.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#include <cooperative_groups.h>

namespace Stream::Cuda {
  namespace Cg = cooperative_groups;
  enum Command : int{ RUN = 1, RESET = 2, STOP = 3, PING = 4 };
  // Aligned naturally atomic load/store words. No host/device RMW operations.
  struct alignas(64) Mailbox{
    alignas(64) std::uint64_t request;
    alignas(64) std::uint64_t completion;
    alignas(64) std::uint64_t exited;
    alignas(64) int command;
    float score;
    std::uint64_t checksum;
  };
  using SystemAtomic = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>;

  struct Layer{
    const void* norm1;
    const void* norm2;
    const void* q;
    const void* k;
    const void* v;
    const void* out;
    const void* gate;
    const void* up;
    const void* down;
  };
  struct Args{
    ModelConfig c;
    bool mixed;
    const void* inputWeight;
    const void* finalNorm;
    const void* scoreWeight;
    Layer layers[16];
    void* keys;
    void* values;
    float* input;
    float* score;
    float* x;
    float* norm;
    float* q;
    float* k;
    float* v;
    float* attention;
    float* temp;
    float* gate;
    float* up;
    float* logits;
    float* hidden;
    std::uint64_t* position;
    Mailbox* mailbox;
    const float* mappedInput;
    float* mappedOutput;
    std::uint64_t maxEvents;
    std::uint64_t initialSequence;
    std::uint64_t idleCycles;
    bool backoff;
  };

  __device__ inline int global_thread(){ return blockIdx.x * blockDim.x + threadIdx.x; }
  __device__ inline int total_threads(){ return blockDim.x * gridDim.x; }
  __device__ inline float load_value(const void* data, int index, bool mixed){
    return mixed ? __half2float(static_cast<const __half*>(data)[index]) : static_cast<const float*>(data)[index];
  }
  __device__ inline void store_value(void* data, int index, float value, bool mixed){
    if(mixed){
      static_cast<__half*>(data)[index] = __float2half_rn(value);
    }else{
      static_cast<float*>(data)[index] = value;
    }
  }
  __device__ inline float cast_value(float value, bool mixed){
    return mixed ? __half2float(__float2half_rn(value)) : value;
  }
  __device__ inline float warp_sum(float value){
    for(int offset = 16; offset > 0; offset /= 2){
      value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
  }

  // Each warp owns an output row. Dot products always accumulate in FP32.
  __device__ inline void matvec(const Args& a, const void* weight, const float* input,
                               float* output, int rows, int cols, bool final = false, float* preCast = nullptr){
    int lane = threadIdx.x % 32;
    for(int row = global_thread() / 32; row < rows; row += total_threads() / 32){
      float sum = 0;
      for(int i = lane; i < cols; i += 32){
        sum += load_value(weight, row * cols + i, a.mixed) * input[i];
      }
      sum = warp_sum(sum);
      if(lane == 0){
        if(preCast){ preCast[row] = sum; }
        output[row] = final ? sum : cast_value(sum, a.mixed);
      }
    }
  }

  __device__ inline void rms_norm(const Args& a, const float* input, const void* weight, float* output, float* preCast = nullptr){
    // Redundant per-warp reduction avoids a grid barrier for this short vector.
    float sum = 0;
    for(int i = threadIdx.x % 32; i < a.c.width; i += 32){
      sum += input[i] * input[i];
    }
    sum = warp_sum(sum);
    float scale = 1.0f / sqrtf(__shfl_sync(0xffffffff, sum, 0) / a.c.width + a.c.epsilon);
    for(int i = global_thread(); i < a.c.width; i += total_threads()){
      float value = input[i] * scale * load_value(weight, i, a.mixed);
      if(preCast){ preCast[i] = value; }
      output[i] = cast_value(value, a.mixed);
    }
  }

  constexpr int LAYER_STAGES = 18;
  __device__ inline void layer_stage(const Args& a, int l, int stage){
    int width = a.c.width, dim = width / a.c.heads;
    std::uint64_t t = *a.position;
    int valid = int(t + 1 < a.c.window ? t + 1 : a.c.window);
    const Layer& w = a.layers[l];
    switch(stage){
      case 0: rms_norm(a, a.x, w.norm1, a.norm); break;
      case 1: matvec(a, w.q, a.norm, a.q, width, width); break;
      case 2: matvec(a, w.k, a.norm, a.k, width, width); break;
      case 3: matvec(a, w.v, a.norm, a.v, width, width); break;
      case 4:
        for(int pair = global_thread(); pair < width / 2; pair += total_threads()){
          int i = pair * 2;
          float angle = float(t) * powf(10000.0f, -float(i % dim) / dim);
          float co = cosf(angle), si = sinf(angle);
          float q0 = a.q[i], q1 = a.q[i + 1], k0 = a.k[i], k1 = a.k[i + 1];
          a.q[i] = cast_value(q0 * co - q1 * si, a.mixed);
          a.q[i + 1] = cast_value(q0 * si + q1 * co, a.mixed);
          a.k[i] = cast_value(k0 * co - k1 * si, a.mixed);
          a.k[i + 1] = cast_value(k0 * si + k1 * co, a.mixed);
        }
        break;
      case 5:
        for(int i = global_thread(); i < width; i += total_threads()){
          int index = ((l * a.c.heads + i / dim) * a.c.window + t % a.c.window) * dim + i % dim;
          store_value(a.keys, index, a.k[i], a.mixed);
          store_value(a.values, index, a.v[i], a.mixed);
        }
        break;
      case 6:
        for(int item = global_thread() / 32; item < a.c.heads * valid; item += total_threads() / 32){
          int head = item / valid, j = item % valid;
          int slot = (t + 1 - valid + j) % a.c.window;
          int base = ((l * a.c.heads + head) * a.c.window + slot) * dim;
          float dot = 0;
          for(int i = threadIdx.x % 32; i < dim; i += 32){
            dot += a.q[head * dim + i] * load_value(a.keys, base + i, a.mixed);
          }
          dot = warp_sum(dot);
          if(threadIdx.x % 32 == 0){
            a.logits[head * a.c.window + j] = dot / sqrtf(float(dim));
          }
        }
        break;
      case 7:
        for(int h = global_thread(); h < a.c.heads; h += total_threads()){
          float* logits = a.logits + h * a.c.window;
          float maximum = -INFINITY;
          for(int j = 0; j < valid; j++){
            maximum = fmaxf(maximum, logits[j]);
          }
          float total = 0;
          for(int j = 0; j < valid; j++){
            logits[j] = expf(logits[j] - maximum);
            total += logits[j];
          }
          for(int j = 0; j < valid; j++){
            logits[j] /= total;
          }
        }
        break;
      case 8:
        for(int i = global_thread(); i < width; i += total_threads()){
          int h = i / dim;
          float sum = 0;
          for(int j = 0; j < valid; j++){
            int slot = (t + 1 - valid + j) % a.c.window;
            int base = ((l * a.c.heads + h) * a.c.window + slot) * dim;
            sum += a.logits[h * a.c.window + j] * load_value(a.values, base + i % dim, a.mixed);
          }
          a.attention[i] = cast_value(sum, a.mixed);
        }
        break;
      case 9: matvec(a, w.out, a.attention, a.temp, width, width); break;
      case 10:
      case 17:
        for(int i = global_thread(); i < width; i += total_threads()){
          a.x[i] = cast_value(a.x[i] + a.temp[i], a.mixed);
          if(stage == 17){
            a.hidden[l * width + i] = a.x[i];
          }
        }
        break;
      case 11: rms_norm(a, a.x, w.norm2, a.norm); break;
      case 12: matvec(a, w.gate, a.norm, a.gate, a.c.ffn, width); break;
      case 13: matvec(a, w.up, a.norm, a.up, a.c.ffn, width); break;
      case 14:
        for(int i = global_thread(); i < a.c.ffn; i += total_threads()){
          float gate = a.gate[i];
          a.gate[i] = cast_value((gate / (1.0f + expf(-gate))) * a.up[i], a.mixed);
        }
        break;
      case 15: matvec(a, w.down, a.gate, a.temp, width, a.c.ffn); break;
      case 16: break; // Reserved instrumentation boundary, omitted in ordinary launches.
    }
  }

  template<bool Multi> __device__ inline void phase_barrier(){
    if constexpr(Multi){
      Cg::this_grid().sync();
    }else{
      __syncthreads();
    }
  }

  template<bool Multi> __device__ void forward(const Args& a){
    matvec(a, a.inputWeight, a.input, a.x, a.c.width, a.c.features);
    phase_barrier<Multi>();
    for(int l = 0; l < a.c.layers; l++){
      for(int stage = 0; stage < LAYER_STAGES; stage++){
        if(stage != 16){
          layer_stage(a, l, stage);
          // All output owners finish before a later phase consumes their values.
          phase_barrier<Multi>();
        }
      }
    }
    rms_norm(a, a.x, a.finalNorm, a.norm);
    phase_barrier<Multi>();
    matvec(a, a.scoreWeight, a.norm, a.score, 1, a.c.width, true);
    phase_barrier<Multi>();
    if(global_thread() == 0){
      (*a.position)++;
    }
    phase_barrier<Multi>();
  }

  __device__ inline void reset_state(const Args& a){
    int count = a.c.layers * a.c.width * a.c.window;
    for(int i = global_thread(); i < count; i += total_threads()){
      store_value(a.keys, i, 0, a.mixed);
      store_value(a.values, i, 0, a.mixed);
    }
    for(int i = global_thread(); i < a.c.layers * a.c.width; i += total_threads()){
      a.hidden[i] = 0;
    }
    if(global_thread() == 0){
      *a.position = 0;
    }
  }
}
