#include "Runtime.hpp"
#include "Provenance.hpp"
#include "Resources.cuh"
#include "Primitives.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <sys/resource.h>
#include <unistd.h>

namespace Stream::Cuda {
  __global__ void stage_kernel(Args a, int layer, int stage){ layer_stage(a, layer, stage); }
  __global__ void attention_kernel(Args a, int layer){
    fused_attention({a.q, a.keys, a.values, a.attention, a.c.heads,
                     a.c.width / a.c.heads, a.c.window, layer, *a.position, a.mixed});
  }
  __global__ void input_kernel(Args a){ matvec(a, a.inputWeight, a.input, a.x, a.c.width, a.c.features); }
  __global__ void norm_kernel(Args a){ rms_norm(a, a.x, a.finalNorm, a.norm); }
  __global__ void score_kernel(Args a){ matvec(a, a.scoreWeight, a.norm, a.score, 1, a.c.width, true); }
  __global__ void advance_kernel(Args a){ (*a.position)++; }
  __global__ void reset_kernel(Args a){ reset_state(a); }
  __global__ void empty_kernel(){}
  __global__ void diagnostic_linear(Args a, const void* weights, const float* input,
                                    float* output, int rows, int cols, bool final, float* preCast){
    matvec(a, weights, input, output, rows, cols, final, preCast);
  }
  __global__ void diagnostic_norm(Args a, const float* input, const void* weights, float* output, float* preCast){
    rms_norm(a, input, weights, output, preCast);
  }
  __global__ void cast_kernel(float* data, __half* halfData, int count, bool round){
    for(int i = global_thread(); i < count; i += total_threads()){
      halfData[i] = __float2half_rn(data[i]);
      if(round){
        data[i] = __half2float(halfData[i]);
      }
    }
  }
  __global__ void stage_input(Args a){
    for(int i = global_thread(); i < a.c.features; i += total_threads()){
      a.input[i] = a.mappedInput[i];
    }
  }
  __global__ void stage_output(Args a){ *a.mappedOutput = *a.score; }

  template<bool Multi> __global__ void fused_kernel(Args a){ forward<Multi>(a); }

  // Coordinator fields are device scratch. Every block follows identical barriers.
  struct Control{
    std::uint64_t sequence;
    int command;
    bool exit;
  };
  template<bool Multi> __global__ void persistent_kernel(Args a, Control* control){
    std::uint64_t previous = a.initialSequence, processed = 0;
    while(true){
      if(global_thread() == 0){
        auto start = clock64();
        std::uint64_t sequence;
        while((sequence = SystemAtomic(a.mailbox->request).load(cuda::memory_order_acquire)) == previous){
          if(clock64() - start >= a.idleCycles){
            control->exit = true;
            break;
          }
          if(a.backoff){
            __nanosleep(128);
          }
        }
        if(!control->exit){
          control->sequence = sequence;
          control->command = a.mailbox->command;
        }
      }
      phase_barrier<Multi>();
      if(control->exit){
        break;
      }
      int command = control->command;
      if(command == RUN){
        for(int i = global_thread(); i < a.c.features; i += total_threads()){
          a.input[i] = a.mappedInput[i];
        }
        phase_barrier<Multi>();
        forward<Multi>(a);
      }else if(command == RESET){
        reset_state(a);
        phase_barrier<Multi>();
      }else if(command == PING){
        if(global_thread() == 0){
          std::uint64_t checksum = 0;
          for(int i = 0; i < a.c.features; i++){
            checksum += __float_as_uint(a.mappedInput[i]);
          }
          a.mailbox->checksum = checksum;
        }
      }
      phase_barrier<Multi>();
      if(global_thread() == 0){
        if(command == RUN){
          *a.mappedOutput = *a.score;
        }
        // Grid writes are visible before this release publishes completion to the CPU.
        SystemAtomic(a.mailbox->completion).store(control->sequence, cuda::memory_order_release);
      }
      phase_barrier<Multi>();
      previous = control->sequence;
      if(command == RUN){
        processed++;
      }
      if(command == STOP || (a.maxEvents && processed >= a.maxEvents)){
        break;
      }
    }
    phase_barrier<Multi>();
    if(global_thread() == 0){
      SystemAtomic(a.mailbox->exited).store(1, cuda::memory_order_release);
    }
  }

  template<bool Multi> __global__ void trace_kernel(Args a, const float* inputs, std::size_t count){
    for(std::size_t event = 0; event < count; event++){
      for(int i = global_thread(); i < a.c.features; i += total_threads()){
        a.input[i] = inputs[event * a.c.features + i];
      }
      phase_barrier<Multi>();
      forward<Multi>(a);
    }
  }

  Json deviceProbe(){
    int device, driver, runtime;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, device));
    auto attribute = [&](cudaDeviceAttr attr){
      int value;
      CUDA_CHECK(cudaDeviceGetAttribute(&value, attr, device));
      return value;
    };
    char bus[32];
    CUDA_CHECK(cudaDeviceGetPCIBusId(bus, sizeof(bus), device));
    return {{"gpu", p.name}, {"device", device}, {"compute_major", p.major}, {"compute_minor", p.minor},
      {"sm_count", p.multiProcessorCount}, {"registers_per_sm", p.regsPerMultiprocessor},
      {"shared_bytes_per_sm", p.sharedMemPerMultiprocessor}, {"shared_bytes_per_block", p.sharedMemPerBlock},
      {"global_bytes", p.totalGlobalMem}, {"l2_bytes", p.l2CacheSize}, {"pci_bus", bus},
      {"clock_khz", attribute(cudaDevAttrClockRate)}, {"memory_clock_khz", attribute(cudaDevAttrMemoryClockRate)},
      {"driver_version", driver}, {"runtime_version", runtime}, {"nvcc", __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 + __CUDACC_VER_BUILD__},
      {"build_cuda_architectures", STREAM_CUDA_ARCHITECTURES},
      {"kernel_timeout", attribute(cudaDevAttrKernelExecTimeout)},
      {"cooperative_launch", attribute(cudaDevAttrCooperativeLaunch)},
      {"can_map_host", attribute(cudaDevAttrCanMapHostMemory)},
      {"unified_addressing", attribute(cudaDevAttrUnifiedAddressing)},
      {"host_native_atomic_supported", attribute(cudaDevAttrHostNativeAtomicSupported)},
      {"cpu_logical_count", std::thread::hardware_concurrency()}, {"os", "Linux"}};
  }
}

namespace Stream {
  using namespace Cuda;

  class Runtime::Impl{
    public:
      Impl(const Model& model, RuntimeOptions options): mModel(model), mOptions(options),
          mScratch(9 * model.config.width + 2 * model.config.ffn + model.config.heads * model.config.window +
                   model.config.layers * model.config.width + model.config.features + 1),
          mKeys(model.config.cacheElements() * (model.mixed ? 2 : 4)),
          mValues(model.config.cacheElements() * (model.mixed ? 2 : 4)), mPosition(1),
          mControl(1), mMailbox(1), mHostInput(model.config.features), mHostOutput(1),
          mHalfScratch(std::max({model.config.width, model.config.ffn, model.config.features})),
          mFloatInputWeight(model.get("input").data.size()){
        const auto& c = model.config;
        if(options.blocks <= 0 || options.idleSeconds <= 0 || options.idleSeconds > 3600){
          throw std::runtime_error("Invalid blocks or idle budget");
        }
        if(options.initialSequence >= std::numeric_limits<std::uint64_t>::max() - 1){
          throw std::runtime_error("Initial sequence must leave room for RUN and STOP");
        }
        mProbe = deviceProbe();
        int major = mProbe["compute_major"], minor = mProbe["compute_minor"];
        if(!((major == 8 && minor == 6) || (major == 12 && minor == 0))){
          throw std::runtime_error("Supported targets are SM86 and SM120; rebuild for the selected GPU");
        }
        if(mProbe["kernel_timeout"] == 1 && options.backend == Backend::PERSISTENT){
          throw std::runtime_error("Interactive persistence refused with active watchdog; use finite device profiling");
        }
        if(!mProbe["can_map_host"].get<int>() || !mProbe["unified_addressing"].get<int>()){
          throw std::runtime_error("Mapped communication requires mapped host memory and UVA");
        }
        // x86-64 naturally aligned 64-bit loads/stores to CUDA mapped pinned memory:
        // use system-scope load/store only, never host/device RMW. See docs/transport.md.
        static_assert(sizeof(void*) == 8 && alignof(Mailbox) >= 64);
        mArgs = {};
        mArgs.c = c;
        mArgs.mixed = model.mixed;
        mArgs.vectorMatvec = options.vectorMatvec;
        mArgs.fusedAttention = options.fusedAttention;
        if(options.fusedAttention && ((c.width / c.heads != 16 && c.width / c.heads != 32) || c.window > 512)){
          throw std::runtime_error("Fused attention requires head dimension 16 or 32 and window <=512");
        }
        mArgs.keys = mKeys.get();
        mArgs.values = mValues.get();
        mArgs.position = mPosition.get();
        mArgs.mailbox = mMailbox.dData;
        mArgs.mappedInput = mHostInput.dData;
        mArgs.mappedOutput = mHostOutput.dData;
        mArgs.maxEvents = options.maxEvents;
        mArgs.initialSequence = options.initialSequence;
        mArgs.idleCycles = static_cast<std::uint64_t>(options.idleSeconds * mProbe["clock_khz"].get<double>() * 1000);
        mArgs.backoff = options.backoff;
        float* cursor = mScratch.get();
        auto take = [&](int count){ auto pointer = cursor; cursor += count; return pointer; };
        mArgs.x = take(c.width);
        mArgs.norm = take(c.width);
        mArgs.q = take(c.width);
        mArgs.k = take(c.width);
        mArgs.v = take(c.width);
        mArgs.attention = take(c.width);
        mArgs.temp = take(c.width);
        mArgs.gate = take(c.ffn);
        mArgs.up = take(c.ffn);
        mArgs.logits = take(c.heads * c.window);
        mArgs.hidden = take(c.layers * c.width);
        mArgs.input = take(c.features);
        mArgs.score = take(1);
        auto upload = [&](const std::string& name) -> const void* {
          const auto& data = model.get(name).data;
          auto buffer = std::make_unique<Buffer<char>>(data.size() * (model.mixed ? 2 : 4));
          if(model.mixed){
            std::vector<__half> half(data.size());
            for(std::size_t i = 0; i < data.size(); i++){
              half[i] = __float2half(data[i]);
            }
            CUDA_CHECK(cudaMemcpy(buffer->get(), half.data(), half.size() * 2, cudaMemcpyHostToDevice));
          }else{
            CUDA_CHECK(cudaMemcpy(buffer->get(), data.data(), data.size() * 4, cudaMemcpyHostToDevice));
          }
          auto result = buffer->get();
          mWeights.push_back(std::move(buffer));
          return result;
        };
        mArgs.inputWeight = upload("input");
        mArgs.finalNorm = upload("final_norm");
        mArgs.scoreWeight = upload("score");
        for(int l = 0; l < c.layers; l++){
          auto p = "layers." + std::to_string(l) + ".";
          mArgs.layers[l] = {upload(p + "norm1"), upload(p + "norm2"), upload(p + "q"),
            upload(p + "k"), upload(p + "v"), upload(p + "out"), upload(p + "gate"), upload(p + "up"), upload(p + "down")};
        }
        CUDA_CHECK(cudaMemcpy(mFloatInputWeight.get(), model.get("input").data.data(),
                              model.get("input").data.size() * 4, cudaMemcpyHostToDevice));
        BLAS_CHECK(cublasSetStream(mBlas.value, mStream.value));
        BLAS_CHECK(cublasSetMathMode(mBlas.value, CUBLAS_PEDANTIC_MATH));
        BLAS_CHECK(cublasSetAtomicsMode(mBlas.value, CUBLAS_ATOMICS_NOT_ALLOWED));
        // cuBLAS workspace is explicitly owned and allocated before capture/timing.
        mBlasWorkspace = std::make_unique<Buffer<char>>(4 * 1024 * 1024);
        BLAS_CHECK(cublasSetWorkspace(mBlas.value, mBlasWorkspace->get(), mBlasWorkspace->count()));
        CUDA_CHECK(cudaMemsetAsync(mScratch.get(), 0, mScratch.count() * 4, mStream.value));
        resetDevice();
        CUDA_CHECK(cudaStreamSynchronize(mStream.value));
        std::memset(mMailbox.hData, 0, sizeof(Mailbox));
        mSequence = options.initialSequence;
        mMailbox.hData->request = mSequence;
        mMailbox.hData->completion = mSequence;
        std::fill_n(mHostInput.hData, c.features, 0);
        *mHostOutput.hData = 0;
        if(options.backend == Backend::GRAPH || options.backend == Backend::CUBLAS_GRAPH){
          // Warm the library outside capture, then restore the initial state.
          launchSequence();
          resetDevice();
          CUDA_CHECK(cudaStreamSynchronize(mStream.value));
          CUDA_CHECK(cudaStreamBeginCapture(mStream.value, cudaStreamCaptureModeThreadLocal));
          if(options.mapped){ stage_input<<<1, 128, 0, mStream.value>>>(mArgs); }
          launchSequence();
          if(options.mapped){ stage_output<<<1, 1, 0, mStream.value>>>(mArgs); }
          CUDA_CHECK(cudaStreamEndCapture(mStream.value, &mGraph.value));
          CUDA_CHECK(cudaGraphInstantiate(&mGraph.exec, mGraph.value, 0));
        }
        mLegalBlocks = legalBlocks();
        if((options.backend == Backend::FUSED || options.backend == Backend::PERSISTENT) && options.blocks > mLegalBlocks){
          throw std::runtime_error("Requested cooperative grid exceeds actual occupancy limit " + std::to_string(mLegalBlocks));
        }
        // Querying a cold kernel can synchronize the CUDA context under lazy
        // loading. Cache metadata before starting a worker that waits for the CPU.
        CUDA_CHECK(cudaFuncGetAttributes(&mAttentionAttributes, attention_kernel));
        if(options.backend == Backend::PERSISTENT){ start(); }
      }

      ~Impl(){
        try{ stop(); }catch(...){
          // A failed kernel must be joined before member buffers can be destroyed.
          cudaStreamSynchronize(mStream.value);
        }
      }

      void resetDevice(){
        reset_kernel<<<mOptions.blocks, 128, 0, mStream.value>>>(mArgs);
        CUDA_CHECK(cudaGetLastError());
      }

      int legalBlocks(){
        int blocks;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, persistent_kernel<true>, 128, 0));
        int fusedBlocks;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fusedBlocks, fused_kernel<true>, 128, 0));
        int limit = std::min(blocks, fusedBlocks) * mProbe["sm_count"].get<int>();
        if(mProbe["cooperative_launch"] != 1){ return 1; }
        return limit;
      }

      void start(){
        CUDA_CHECK(cudaMemsetAsync(mControl.get(), 0, sizeof(Control), mStream.value));
        if(mOptions.blocks == 1){
          persistent_kernel<false><<<1, 128, 0, mStream.value>>>(mArgs, mControl.get());
        }else{
          auto control = mControl.get();
          void* parameters[] = {&mArgs, &control};
          CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(persistent_kernel<true>),
              mOptions.blocks, 128, parameters, 0, mStream.value));
        }
        CUDA_CHECK(cudaGetLastError());
        mRunning = true;
      }

      void blasLinear(const void* weights, const float* input, float* output, int rows, int cols, bool final = false, bool inputProjection = false){
        const float alpha = 1, beta = 0;
        if(!mModel.mixed || inputProjection){
          auto data = inputProjection ? mFloatInputWeight.get() : static_cast<const float*>(weights);
          BLAS_CHECK(cublasSgemv(mBlas.value, CUBLAS_OP_T, cols, rows, &alpha, data, cols, input, 1, &beta, output, 1));
        }else{
          cast_kernel<<<1, 128, 0, mStream.value>>>(const_cast<float*>(input), mHalfScratch.get(), cols, false);
          BLAS_CHECK(cublasGemmEx(mBlas.value, CUBLAS_OP_T, CUBLAS_OP_N, rows, 1, cols,
              &alpha, weights, CUDA_R_16F, cols, mHalfScratch.get(), CUDA_R_16F, cols,
              &beta, output, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
        }
        if(mModel.mixed && !final){
          cast_kernel<<<1, 128, 0, mStream.value>>>(output, mHalfScratch.get(), rows, true);
        }
      }

      void launchSequence(){
        bool blas = mOptions.backend == Backend::CUBLAS || mOptions.backend == Backend::CUBLAS_GRAPH;
        int blocks = mOptions.blocks;
        const auto& c = mArgs.c;
        if(blas){
          blasLinear(mArgs.inputWeight, mArgs.input, mArgs.x, c.width, c.features, false, true);
        }else{
          input_kernel<<<blocks, 128, 0, mStream.value>>>(mArgs);
        }
        for(int l = 0; l < c.layers; l++){
          const auto& w = mArgs.layers[l];
          for(int s = 0; s < LAYER_STAGES; s++){
            if(s == 16 || (mArgs.fusedAttention && (s == 7 || s == 8))){ continue; }
            if(mArgs.fusedAttention && s == 6){
              attention_kernel<<<c.heads, 128, 0, mStream.value>>>(mArgs, l);
              CUDA_CHECK(cudaGetLastError());
              continue;
            }
            if(blas && (s == 1 || s == 2 || s == 3 || s == 9 || s == 12 || s == 13 || s == 15)){
              if(s == 1){ blasLinear(w.q, mArgs.norm, mArgs.q, c.width, c.width); }
              if(s == 2){ blasLinear(w.k, mArgs.norm, mArgs.k, c.width, c.width); }
              if(s == 3){ blasLinear(w.v, mArgs.norm, mArgs.v, c.width, c.width); }
              if(s == 9){ blasLinear(w.out, mArgs.attention, mArgs.temp, c.width, c.width); }
              if(s == 12){ blasLinear(w.gate, mArgs.norm, mArgs.gate, c.ffn, c.width); }
              if(s == 13){ blasLinear(w.up, mArgs.norm, mArgs.up, c.ffn, c.width); }
              if(s == 15){ blasLinear(w.down, mArgs.gate, mArgs.temp, c.width, c.ffn); }
            }else{
              stage_kernel<<<blocks, 128, 0, mStream.value>>>(mArgs, l, s);
            }
          }
        }
        norm_kernel<<<blocks, 128, 0, mStream.value>>>(mArgs);
        if(blas){
          blasLinear(mArgs.scoreWeight, mArgs.norm, mArgs.score, 1, c.width, true);
        }else{
          score_kernel<<<blocks, 128, 0, mStream.value>>>(mArgs);
        }
        advance_kernel<<<1, 1, 0, mStream.value>>>(mArgs);
        CUDA_CHECK(cudaGetLastError());
      }

      void launchFused(){
        if(mOptions.blocks == 1){
          fused_kernel<false><<<1, 128, 0, mStream.value>>>(mArgs);
        }else{
          void* parameters[] = {&mArgs};
          CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(fused_kernel<true>),
              mOptions.blocks, 128, parameters, 0, mStream.value));
        }
        CUDA_CHECK(cudaGetLastError());
      }

      void command(int command){
        if(!mRunning){ throw std::runtime_error("Persistent runtime is stopped"); }
        // Reserve the final ID for STOP so exhaustion cannot strand a live worker.
        auto maximum = std::numeric_limits<std::uint64_t>::max();
        if(mSequence == maximum || (mSequence == maximum - 1 && command != STOP)){
          throw std::runtime_error("Sequence exhausted; create a new runtime before wraparound");
        }
        if(SystemAtomic(mMailbox.hData->exited).load(cuda::memory_order_acquire)){
          CUDA_CHECK(cudaStreamSynchronize(mStream.value));
          mRunning = false;
          throw std::runtime_error("Persistent worker exited: finite-event or idle budget reached");
        }
        mMailbox.hData->command = command;
        // Input and command are fully written before publishing the new request.
        SystemAtomic(mMailbox.hData->request).store(++mSequence, cuda::memory_order_release);
        auto start = std::chrono::steady_clock::now();
        std::uint64_t polls = 0;
        while(SystemAtomic(mMailbox.hData->completion).load(cuda::memory_order_acquire) != mSequence){
          if(mOptions.backoff){ std::this_thread::yield(); }
          if((++polls & 0xffff) == 0){
            auto status = cudaStreamQuery(mStream.value);
            if(status != cudaErrorNotReady){
              CUDA_CHECK(status);
              throw std::runtime_error("Persistent worker exited before completion");
            }
            if(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > mOptions.idleSeconds + 30){
              throw std::runtime_error("Persistent response timeout");
            }
          }
        }
      }

      float step(const float* input){
        std::memcpy(mHostInput.hData, input, mArgs.c.features * 4);
        if(mOptions.backend == Backend::PERSISTENT){
          command(RUN);
        }else{
          if(!mOptions.mapped){
            CUDA_CHECK(cudaMemcpyAsync(mArgs.input, mHostInput.hData, mArgs.c.features * 4, cudaMemcpyHostToDevice, mStream.value));
          }else if(mOptions.backend != Backend::GRAPH && mOptions.backend != Backend::CUBLAS_GRAPH){
            stage_input<<<1, 128, 0, mStream.value>>>(mArgs);
          }
          if(mOptions.backend == Backend::GRAPH || mOptions.backend == Backend::CUBLAS_GRAPH){
            CUDA_CHECK(cudaGraphLaunch(mGraph.exec, mStream.value));
          }else if(mOptions.backend == Backend::FUSED){
            launchFused();
          }else{
            launchSequence();
          }
          if(!mOptions.mapped){
            CUDA_CHECK(cudaMemcpyAsync(mHostOutput.hData, mArgs.score, 4, cudaMemcpyDeviceToHost, mStream.value));
          }else if(mOptions.backend != Backend::GRAPH && mOptions.backend != Backend::CUBLAS_GRAPH){
            stage_output<<<1, 1, 0, mStream.value>>>(mArgs);
          }
          CUDA_CHECK(cudaStreamSynchronize(mStream.value));
        }
        return *mHostOutput.hData;
      }

      void reset(){
        if(mOptions.backend == Backend::PERSISTENT){
          command(RESET);
        }else{
          resetDevice();
          CUDA_CHECK(cudaStreamSynchronize(mStream.value));
        }
      }

      void stop(){
        if(mRunning){
          if(!SystemAtomic(mMailbox.hData->exited).load(cuda::memory_order_acquire)){
            command(STOP);
          }
          CUDA_CHECK(cudaStreamSynchronize(mStream.value));
          mRunning = false;
        }
      }

      std::map<std::string, std::vector<float>> traceStep(const float* input){
        if(mOptions.backend != Backend::CUDA){ throw std::runtime_error("traceStep requires conventional custom CUDA"); }
        std::map<std::string, std::vector<float>> trace;
        const auto& c = mArgs.c;
        Buffer<float> preCast(std::max(c.width, c.ffn));
        auto capture = [&](const std::string& name, const float* pointer, int count){
          auto& values = trace[name];
          values.resize(count);
          CUDA_CHECK(cudaMemcpyAsync(values.data(), pointer, count * sizeof(float), cudaMemcpyDeviceToHost, mStream.value));
          CUDA_CHECK(cudaStreamSynchronize(mStream.value));
        };
        auto linear = [&](const std::string& name, const void* weights, const float* operand,
                          float* output, int rows, int cols, bool final = false){
          capture(name + ".operand", operand, cols);
          diagnostic_linear<<<mOptions.blocks, 128, 0, mStream.value>>>(mArgs, weights, operand, output, rows, cols, final, preCast.get());
          CUDA_CHECK(cudaGetLastError());
          capture(name + ".pre_cast", preCast.get(), rows);
          capture(name + ".linear", output, rows);
        };
        auto norm = [&](const std::string& name, const void* weights){
          capture(name + ".operand", mArgs.x, c.width);
          diagnostic_norm<<<mOptions.blocks, 128, 0, mStream.value>>>(mArgs, mArgs.x, weights, mArgs.norm, preCast.get());
          CUDA_CHECK(cudaGetLastError());
          capture(name + ".pre_cast", preCast.get(), c.width);
          capture(name + ".norm", mArgs.norm, c.width);
        };
        CUDA_CHECK(cudaMemcpyAsync(mArgs.input, input, c.features * sizeof(float), cudaMemcpyHostToDevice, mStream.value));
        linear("input", mArgs.inputWeight, mArgs.input, mArgs.x, c.width, c.features);
        capture("input", mArgs.x, c.width);
        for(int l = 0; l < c.layers; l++){
          const auto& w = mArgs.layers[l];
          auto p = "layers." + std::to_string(l) + ".";
          for(int s = 0; s < LAYER_STAGES; s++){
            if(s == 16 || (mArgs.fusedAttention && (s == 7 || s == 8))){ continue; }
            if(mArgs.fusedAttention && s == 6){
              attention_kernel<<<c.heads, 128, 0, mStream.value>>>(mArgs, l);
              CUDA_CHECK(cudaGetLastError());
              capture(p + "attention", mArgs.attention, c.width);
              continue;
            }
            if(s == 0){ norm(p + "norm1", w.norm1); }
            else if(s == 1){ linear(p + "q", w.q, mArgs.norm, mArgs.q, c.width, c.width); }
            else if(s == 2){ linear(p + "k", w.k, mArgs.norm, mArgs.k, c.width, c.width); }
            else if(s == 3){ linear(p + "v", w.v, mArgs.norm, mArgs.v, c.width, c.width); }
            else if(s == 9){ linear(p + "out", w.out, mArgs.attention, mArgs.temp, c.width, c.width); }
            else if(s == 11){ norm(p + "norm2", w.norm2); }
            else if(s == 12){ linear(p + "gate", w.gate, mArgs.norm, mArgs.gate, c.ffn, c.width); }
            else if(s == 13){ linear(p + "up", w.up, mArgs.norm, mArgs.up, c.ffn, c.width); }
            else if(s == 15){ linear(p + "down", w.down, mArgs.gate, mArgs.temp, c.width, c.ffn); }
            else{
              stage_kernel<<<mOptions.blocks, 128, 0, mStream.value>>>(mArgs, l, s);
              CUDA_CHECK(cudaGetLastError());
            }
            if(s == 4){
              capture(p + "q", mArgs.q, c.width);
              capture(p + "k", mArgs.k, c.width);
              capture(p + "v", mArgs.v, c.width);
            }
            if(s == 8){ capture(p + "attention", mArgs.attention, c.width); }
            if(s == 10){ capture(p + "attention_residual", mArgs.x, c.width); }
            if(s == 14){ capture(p + "activation", mArgs.gate, c.ffn); }
            if(s == 17){ capture(p + "hidden", mArgs.x, c.width); }
          }
        }
        norm("final_norm", mArgs.finalNorm);
        linear("score", mArgs.scoreWeight, mArgs.norm, mArgs.score, 1, c.width, true);
        advance_kernel<<<1, 1, 0, mStream.value>>>(mArgs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(mStream.value));
        return trace;
      }

      Snapshot snapshot(){
        // One outstanding command; caller invokes only after its completion.
        Snapshot s;
        CUDA_CHECK(cudaMemcpy(&s.position, mArgs.position, 8, cudaMemcpyDeviceToHost));
        auto download = [&](const void* data){
          std::vector<float> result(mArgs.c.cacheElements());
          if(mArgs.mixed){
            std::vector<__half> half(result.size());
            CUDA_CHECK(cudaMemcpy(half.data(), data, half.size() * 2, cudaMemcpyDeviceToHost));
            for(std::size_t i = 0; i < result.size(); i++){ result[i] = __half2float(half[i]); }
          }else{
            CUDA_CHECK(cudaMemcpy(result.data(), data, result.size() * 4, cudaMemcpyDeviceToHost));
          }
          return result;
        };
        s.keys = download(mArgs.keys);
        s.values = download(mArgs.values);
        s.hidden.resize(mArgs.c.layers * mArgs.c.width);
        CUDA_CHECK(cudaMemcpy(s.hidden.data(), mArgs.hidden, s.hidden.size() * 4, cudaMemcpyDeviceToHost));
        return s;
      }

      Json info() const{
        cudaFuncAttributes attributes;
        const void* kernel = reinterpret_cast<const void*>(stage_kernel);
        std::string resourceKernel = "stage_kernel (custom support stages; cuBLAS internals require profiling)";
        if(mOptions.backend == Backend::PERSISTENT){
          kernel = mOptions.blocks == 1 ? reinterpret_cast<const void*>(persistent_kernel<false>) : reinterpret_cast<const void*>(persistent_kernel<true>);
          resourceKernel = mOptions.blocks == 1 ? "persistent_kernel<false>" : "persistent_kernel<true>";
        }else if(mOptions.backend == Backend::FUSED){
          kernel = mOptions.blocks == 1 ? reinterpret_cast<const void*>(fused_kernel<false>) : reinterpret_cast<const void*>(fused_kernel<true>);
          resourceKernel = mOptions.blocks == 1 ? "fused_kernel<false>" : "fused_kernel<true>";
        }
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
        return {{"environment", mProbe}, {"blocks", mOptions.blocks}, {"legal_cooperative_blocks", mLegalBlocks},
          {"resource_kernel", resourceKernel},
          {"attention_implementation", mOptions.fusedAttention ? "fused-tiled-block-per-head" : "staged"},
          {"standalone_fused_attention_resources", {{"blocks", mArgs.c.heads}, {"threads", 128},
            {"registers_per_thread", mAttentionAttributes.numRegs}, {"local_bytes", mAttentionAttributes.localSizeBytes},
            {"shared_bytes", mAttentionAttributes.sharedSizeBytes}}},
          {"matvec_implementation", mOptions.vectorMatvec ? "vector-pair-with-scalar-fallback" : "scalar"},
          {"threads", 128}, {"registers_per_thread", attributes.numRegs}, {"local_bytes", attributes.localSizeBytes},
          {"shared_bytes", attributes.sharedSizeBytes}, {"transport", mOptions.backend == Backend::PERSISTENT ? "mapped-system-load-store" : mOptions.mapped ? "mapped-stream-sync" : "pinned-copy-stream-sync"},
          {"poll_policy", mOptions.backoff ? "bounded-device-backoff-host-yield" : "tight"},
          {"workspace_storage", "FP32 with explicit FP16 rounding in mixed mode"},
          {"weights_and_kv_storage", mArgs.mixed ? "FP16" : "FP32"}};
      }

      double deviceTrace(const float* inputs, std::size_t count){
        if(mOptions.backend == Backend::PERSISTENT){ stop(); }
        Buffer<float> trace(count * mArgs.c.features);
        CUDA_CHECK(cudaMemcpyAsync(trace.get(), inputs, trace.count() * 4, cudaMemcpyHostToDevice, mStream.value));
        resetDevice();
        Event start, end;
        CUDA_CHECK(cudaEventRecord(start.value, mStream.value));
        if(mOptions.backend == Backend::PERSISTENT || mOptions.backend == Backend::FUSED){
          if(mOptions.blocks == 1){
            trace_kernel<false><<<1, 128, 0, mStream.value>>>(mArgs, trace.get(), count);
          }else{
            auto pointer = trace.get();
            void* parameters[] = {&mArgs, &pointer, &count};
            CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(trace_kernel<true>),
                mOptions.blocks, 128, parameters, 0, mStream.value));
          }
        }else{
          for(std::size_t i = 0; i < count; i++){
            CUDA_CHECK(cudaMemcpyAsync(mArgs.input, trace.get() + i * mArgs.c.features,
                                      mArgs.c.features * 4, cudaMemcpyDeviceToDevice, mStream.value));
            // Compute graph excludes host I/O; stage through a distinct capture if mapped.
            if((mOptions.backend == Backend::GRAPH || mOptions.backend == Backend::CUBLAS_GRAPH) && !mOptions.mapped){
              CUDA_CHECK(cudaGraphLaunch(mGraph.exec, mStream.value));
            }else{
              launchSequence();
            }
          }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(end.value, mStream.value));
        CUDA_CHECK(cudaEventSynchronize(end.value));
        float milliseconds;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.value, end.value));
        return milliseconds * 1000.0 / count;
      }

      const Model& mModel;
      RuntimeOptions mOptions;
      StreamHandle mStream;
      Blas mBlas;
      Graph mGraph;
      Buffer<float> mScratch;
      Buffer<char> mKeys, mValues;
      Buffer<std::uint64_t> mPosition;
      Buffer<Control> mControl;
      Pinned<Mailbox> mMailbox;
      Pinned<float> mHostInput, mHostOutput;
      Buffer<__half> mHalfScratch;
      Buffer<float> mFloatInputWeight;
      std::unique_ptr<Buffer<char>> mBlasWorkspace;
      std::vector<std::unique_ptr<Buffer<char>>> mWeights;
      Args mArgs{};
      cudaFuncAttributes mAttentionAttributes{};
      Json mProbe;
      int mLegalBlocks = 1;
      bool mRunning = false;
      std::uint64_t mSequence = 0;
  };

  Runtime::Runtime(const Model& model, RuntimeOptions options): mImpl(std::make_unique<Impl>(model, options)){}
  Runtime::~Runtime() = default;
  float Runtime::step(const float* input){ return mImpl->step(input); }
  void Runtime::reset(){ mImpl->reset(); }
  void Runtime::stop(){ mImpl->stop(); }
  Snapshot Runtime::snapshot(){ return mImpl->snapshot(); }
  std::map<std::string, std::vector<float>> Runtime::traceStep(const float* input){ return mImpl->traceStep(input); }
  Json Runtime::info() const{ return mImpl->info(); }
  double Runtime::deviceTrace(const float* inputs, std::size_t count){ return mImpl->deviceTrace(inputs, count); }
  Json probe(){ return deviceProbe(); }
}

namespace Stream::Cuda {
  __global__ void echo_kernel(const float* input, float* output, int count){
    for(int i = threadIdx.x; i < count; i += blockDim.x){ output[i] = input[i]; }
  }
  __global__ void ping_kernel(Mailbox* mailbox, float* input, std::size_t count, bool backoff,
                              std::uint64_t idleCycles, bool gpuInitiated, std::uint64_t* cycles, int* errors){
    if(threadIdx.x || blockIdx.x){ return; }
    for(std::uint64_t seq = 1; seq <= count; seq++){
      auto start = clock64();
      if(gpuInitiated){
        mailbox->checksum = seq * 2654435761ULL;
        SystemAtomic(mailbox->request).store(seq, cuda::memory_order_release);
        while(SystemAtomic(mailbox->completion).load(cuda::memory_order_acquire) != seq){
          if(clock64() - start > idleCycles){
            SystemAtomic(mailbox->exited).store(1, cuda::memory_order_release);
            return;
          }
          if(backoff){ __nanosleep(128); }
        }
        if(mailbox->checksum != (seq * 2654435761ULL ^ 0xabcdefULL)){ (*errors)++; }
      }else{
        while(SystemAtomic(mailbox->request).load(cuda::memory_order_acquire) != seq){
          if(clock64() - start > idleCycles){
            SystemAtomic(mailbox->exited).store(1, cuda::memory_order_release);
            return;
          }
          if(backoff){ __nanosleep(128); }
        }
        std::uint64_t checksum = 0;
        for(int i = 0; i < 32; i++){ checksum += __float_as_uint(input[i]); }
        mailbox->checksum = checksum;
        SystemAtomic(mailbox->completion).store(seq, cuda::memory_order_release);
      }
      cycles[seq - 1] = clock64() - start;
    }
    SystemAtomic(mailbox->exited).store(1, cuda::memory_order_release);
  }
}

namespace Stream {
  Json transportBench(const std::string& mode, std::size_t count, bool backoff, const std::string& output){
    auto environment = probe();
    if((mode == "ping-pong" || mode == "gpu-ping") && environment["kernel_timeout"] == 1){
      throw std::runtime_error("Host-fed ping-pong refused with active watchdog");
    }
    StreamHandle stream;
    Graph graph;
    Pinned<Mailbox> mailbox(1);
    Pinned<float> input(32), result(32);
    Buffer<float> d_input(32), d_output(32);
    Buffer<std::uint64_t> cycles(count);
    Buffer<int> errors(1);
    CUDA_CHECK(cudaMemset(errors.get(), 0, sizeof(int)));
    std::memset(mailbox.hData, 0, sizeof(Mailbox));
    bool interactive = mode == "ping-pong" || mode == "gpu-ping";
    if(mode == "empty-graph"){
      CUDA_CHECK(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
      empty_kernel<<<1, 1, 0, stream.value>>>();
      CUDA_CHECK(cudaStreamEndCapture(stream.value, &graph.value));
      CUDA_CHECK(cudaGraphInstantiate(&graph.exec, graph.value, 0));
    }
    std::uint64_t budget = environment["clock_khz"].get<std::uint64_t>() * 1000ULL * 30;
    if(interactive){
      ping_kernel<<<1, 1, 0, stream.value>>>(mailbox.dData, input.dData, count, backoff, budget,
                                          mode == "gpu-ping", cycles.get(), errors.get());
      CUDA_CHECK(cudaGetLastError());
    }
    std::vector<double> samples(count);
    std::uint64_t failed = 0;
    rusage before{}, after{};
    getrusage(RUSAGE_SELF, &before);
    auto begin = std::chrono::steady_clock::now();
    auto wait = [&](std::uint64_t& word, std::uint64_t seq){
      auto start = std::chrono::steady_clock::now();
      std::size_t polls = 0;
      while(SystemAtomic(word).load(cuda::memory_order_acquire) != seq){
        if(backoff){ std::this_thread::yield(); }
        if((++polls & 0xffff) == 0){
          if(SystemAtomic(mailbox.hData->exited).load(cuda::memory_order_acquire)){
            throw std::runtime_error("Ping-pong kernel exited before request/acknowledgement");
          }
          if(std::chrono::steady_clock::now() - start > std::chrono::seconds(60)){
            throw std::runtime_error("Ping-pong host timeout");
          }
        }
      }
    };
    for(std::uint64_t seq = 1; seq <= count; seq++){
      std::uint64_t checksum = 0;
      for(int i = 0; i < 32; i++){
        input.hData[i] = float(seq % 1024 + i);
        std::uint32_t bits;
        std::memcpy(&bits, input.hData + i, 4);
        checksum += bits;
      }
      auto start = std::chrono::steady_clock::now();
      if(mode == "gpu-ping"){
        wait(mailbox.hData->request, seq);
        if(mailbox.hData->checksum != seq * 2654435761ULL){ failed++; }
        mailbox.hData->checksum ^= 0xabcdefULL;
        SystemAtomic(mailbox.hData->completion).store(seq, cuda::memory_order_release);
      }else if(mode == "ping-pong"){
        SystemAtomic(mailbox.hData->request).store(seq, cuda::memory_order_release);
        wait(mailbox.hData->completion, seq);
        if(mailbox.hData->checksum != checksum){ failed++; }
      }else{
        if(mode == "empty"){
          empty_kernel<<<1, 1, 0, stream.value>>>();
        }else if(mode == "empty-graph"){
          CUDA_CHECK(cudaGraphLaunch(graph.exec, stream.value));
        }else{
          CUDA_CHECK(cudaMemcpyAsync(d_input.get(), input.hData, 128, cudaMemcpyHostToDevice, stream.value));
          echo_kernel<<<1, 32, 0, stream.value>>>(d_input.get(), d_output.get(), 32);
          CUDA_CHECK(cudaMemcpyAsync(result.hData, d_output.get(), 128, cudaMemcpyDeviceToHost, stream.value));
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream.value));
        if(mode == "copy" && std::memcmp(result.hData, input.hData, 128)){ failed++; }
      }
      samples[seq - 1] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.value));
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    getrusage(RUSAGE_SELF, &after);
    auto seconds = [](const rusage& usage){
      return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6 + usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6;
    };
    std::vector<std::uint64_t> deviceCycles(count, 0);
    if(interactive){
      CUDA_CHECK(cudaMemcpy(deviceCycles.data(), cycles.get(), count * 8, cudaMemcpyDeviceToHost));
      int deviceErrors;
      CUDA_CHECK(cudaMemcpy(&deviceErrors, errors.get(), 4, cudaMemcpyDeviceToHost));
      failed += deviceErrors;
    }
    std::ofstream csv(output + ".csv");
    csv << "sequence,host_observation_us,device_round_trip_or_wait_cycles\n";
    csv.precision(17);
    for(std::size_t i = 0; i < count; i++){ csv << i + 1 << ',' << samples[i] << ',' << deviceCycles[i] << '\n'; }
    std::sort(samples.begin(), samples.end());
    Json report{{"environment", environment}, {"mode", mode}, {"events", count}, {"failed", failed}, {"provenance", provenance()},
      {"poll_policy", backoff ? "device-nanosleep-128-host-yield" : "tight"}, {"wall_seconds", wall},
      {"cpu_seconds", seconds(after) - seconds(before)},
      {"p50_us", samples[count / 2]}, {"p99_us", samples[std::min(count - 1, std::size_t(count * .99))]},
      {"worst_us", samples.back()},
      {"boundary", mode == "gpu-ping" ? "host wait+ack service observation; GPU full RTT stored separately in device cycles" : "host submission through corresponding host-visible completion"}};
    if(failed){ throw std::runtime_error("Transport checksum failure"); }
    return report;
  }
}
