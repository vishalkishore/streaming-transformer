#pragma once

#include "Model.hpp"
#include <cstdint>
#include <map>
#include <vector>

namespace Stream {
  using CpuTrace = std::map<std::string, std::vector<float>>;
  // Preallocated Eigen baseline; no per-event tensor allocation.
  class Cpu{
    public:
      explicit Cpu(const Model& model, int threads = 1);
      // Optional operation snapshots are for numerical diagnosis, outside timing runs.
      float step(const float* input, CpuTrace* trace = nullptr);
      void reset();
      std::uint64_t getPosition() const;
      const std::vector<float>& getKeys() const;
      const std::vector<float>& getValues() const;
      const std::vector<float>& getHidden() const;
    private:
      void linear(const float* input, const std::string& name, float* output, bool final = false, CpuTrace* trace = nullptr);
      void norm(const float* input, const std::string& name, float* output, CpuTrace* trace = nullptr);
      void cast(float* data, int size);
      const Model& mModel;
      ModelConfig mConfig;
      std::uint64_t mPosition = 0;
      struct LayerNames{
        std::string norm1, norm2, q, k, v, out, gate, up, down;
      };
      std::vector<LayerNames> mNames;
      std::vector<float> mKeys, mValues, mHidden;
      std::vector<float> mX, mNorm, mQ, mK, mV, mAttention, mTemp, mGate, mUp, mLogits;
  };
}
