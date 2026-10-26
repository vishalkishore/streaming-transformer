#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace Stream::Cuda {
  inline void check(cudaError_t status, const char* expression){
    if(status != cudaSuccess){
      throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
    }
  }
  inline void checkBlas(cublasStatus_t status, const char* expression){
    if(status != CUBLAS_STATUS_SUCCESS){
      throw std::runtime_error(std::string(expression) + ": cuBLAS " + std::to_string(int(status)));
    }
  }
  #define CUDA_CHECK(expr) ::Stream::Cuda::check((expr), #expr)
  #define BLAS_CHECK(expr) ::Stream::Cuda::checkBlas((expr), #expr)

  template<class T> class Buffer{
    public:
      explicit Buffer(std::size_t count): mCount(count){
        CUDA_CHECK(cudaMalloc(&mData, count * sizeof(T)));
      }
      ~Buffer(){ cudaFree(mData); }
      Buffer(const Buffer&) = delete;
      Buffer& operator=(const Buffer&) = delete;
      T* get() const{ return mData; }
      std::size_t count() const{ return mCount; }
    private:
      T* mData = nullptr;
      std::size_t mCount;
  };

  class StreamHandle{
    public:
      StreamHandle(){ CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
      ~StreamHandle(){ cudaStreamDestroy(value); }
      StreamHandle(const StreamHandle&) = delete;
      cudaStream_t value = nullptr;
  };
  class Event{
    public:
      Event(){ CUDA_CHECK(cudaEventCreate(&value)); }
      ~Event(){ cudaEventDestroy(value); }
      Event(const Event&) = delete;
      cudaEvent_t value = nullptr;
  };
  class Blas{
    public:
      Blas(){ BLAS_CHECK(cublasCreate(&value)); }
      ~Blas(){ cublasDestroy(value); }
      Blas(const Blas&) = delete;
      cublasHandle_t value = nullptr;
  };
  class Graph{
    public:
      ~Graph(){
        if(exec){ cudaGraphExecDestroy(exec); }
        if(value){ cudaGraphDestroy(value); }
      }
      cudaGraph_t value = nullptr;
      cudaGraphExec_t exec = nullptr;
  };
  template<class T> class Pinned{
    public:
      explicit Pinned(std::size_t count){
        CUDA_CHECK(cudaHostAlloc(&hData, count * sizeof(T), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&dData, hData, 0));
      }
      ~Pinned(){ cudaFreeHost(hData); }
      Pinned(const Pinned&) = delete;
      T* hData = nullptr;
      T* dData = nullptr;
  };
}
