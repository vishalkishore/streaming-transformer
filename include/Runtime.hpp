#pragma once

#include "Model.hpp"
#include <memory>

namespace Stream {
  enum class Backend{ CUDA, GRAPH, CUBLAS, CUBLAS_GRAPH, FUSED, PERSISTENT };
  struct RuntimeOptions{
    Backend backend = Backend::GRAPH;
    int blocks = 1;
    bool backoff = false;
    bool mapped = false;
    bool vectorMatvec = false;
    bool fusedAttention = false;
    std::uint64_t maxEvents = 0;
    // Allows bounded protocol recovery tests near exhaustion; STOP retains one sequence ID.
    std::uint64_t initialSequence = 0;
    double idleSeconds = 30;
  };
  struct Snapshot{
    std::uint64_t position;
    std::vector<float> keys, values, hidden;
  };

  // Owns all device resources and joins its persistent worker before freeing them.
  class Runtime{
    public:
      Runtime(const Model& model, RuntimeOptions options);
      ~Runtime();
      Runtime(const Runtime&) = delete;
      Runtime& operator=(const Runtime&) = delete;
      float step(const float* input);
      void reset();
      void stop();
      Snapshot snapshot();
      // Synchronizing operation snapshots for diagnosis; supported by conventional CUDA only.
      std::map<std::string, std::vector<float>> traceStep(const float* input);
      Json info() const;
      double deviceTrace(const float* inputs, std::size_t count);
    private:
      class Impl;
      std::unique_ptr<Impl> mImpl;
  };

  Json probe();
  Json transportBench(const std::string& mode, std::size_t count, bool backoff, const std::string& output);
}
