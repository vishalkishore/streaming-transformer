#include "Cpu.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/Core>

namespace Stream {
  Cpu::Cpu(const Model& model, int threads): mModel(model), mConfig(model.config),
      mKeys(mConfig.cacheElements()), mValues(mConfig.cacheElements()),
      mHidden(mConfig.layers * mConfig.width), mX(mConfig.width), mNorm(mConfig.width),
      mQ(mConfig.width), mK(mConfig.width), mV(mConfig.width), mAttention(mConfig.width),
      mTemp(mConfig.width), mGate(mConfig.ffn), mUp(mConfig.ffn), mLogits(mConfig.window){
    Eigen::setNbThreads(threads);
    for(int l = 0; l < mConfig.layers; l++){
      auto p = "layers." + std::to_string(l) + ".";
      mNames.push_back({p + "norm1", p + "norm2", p + "q", p + "k", p + "v",
                       p + "out", p + "gate", p + "up", p + "down"});
    }
  }

  void Cpu::cast(float* data, int size){
    if(mModel.mixed){
      for(int i = 0; i < size; i++){
        data[i] = float(Eigen::half(data[i]));
      }
    }
  }

  void Cpu::linear(const float* input, const std::string& name, float* output, bool final, CpuTrace* trace){
    const auto& tensor = mModel.get(name);
    int rows = tensor.shape[0], cols = tensor.shape[1];
    if(trace){ (*trace)[name + ".operand"] = {input, input + cols}; }
    using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<Eigen::VectorXf>(output, rows).noalias() =
        Eigen::Map<const Matrix>(tensor.data.data(), rows, cols) * Eigen::Map<const Eigen::VectorXf>(input, cols);
    if(trace){ (*trace)[name + ".pre_cast"] = {output, output + rows}; }
    if(!final){
      cast(output, rows);
    }
    if(trace){ (*trace)[name + ".linear"] = {output, output + rows}; }
  }

  void Cpu::norm(const float* input, const std::string& name, float* output, CpuTrace* trace){
    if(trace){ (*trace)[name + ".operand"] = {input, input + mConfig.width}; }
    float sum = 0;
    for(int i = 0; i < mConfig.width; i++){
      sum += input[i] * input[i];
    }
    float scale = 1.0f / std::sqrt(sum / mConfig.width + mConfig.epsilon);
    const auto& weight = mModel.get(name).data;
    for(int i = 0; i < mConfig.width; i++){
      output[i] = input[i] * scale * weight[i];
    }
    if(trace){ (*trace)[name + ".pre_cast"] = {output, output + mConfig.width}; }
    cast(output, mConfig.width);
    if(trace){ (*trace)[name + ".norm"] = {output, output + mConfig.width}; }
  }

  float Cpu::step(const float* input, CpuTrace* trace){
    const auto& c = mConfig;
    int d = c.width / c.heads;
    int slot = mPosition % c.window;
    int valid = std::min<std::uint64_t>(mPosition + 1, c.window);
    if(trace){ trace->clear(); }
    linear(input, "input", mX.data(), false, trace);
    if(trace){ (*trace)["input"] = mX; }
    for(int l = 0; l < c.layers; l++){
      const auto& n = mNames[l];
      norm(mX.data(), n.norm1, mNorm.data(), trace);
      linear(mNorm.data(), n.q, mQ.data(), false, trace);
      linear(mNorm.data(), n.k, mK.data(), false, trace);
      linear(mNorm.data(), n.v, mV.data(), false, trace);
      for(int h = 0; h < c.heads; h++){
        for(int i = 0; i < d; i += 2){
          float angle = float(mPosition) * std::pow(10000.0f, -float(i) / d);
          float co = std::cos(angle), si = std::sin(angle);
          for(float* vector : {mQ.data(), mK.data()}){
            float a = vector[h * d + i], b = vector[h * d + i + 1];
            vector[h * d + i] = a * co - b * si;
            vector[h * d + i + 1] = a * si + b * co;
          }
        }
      }
      cast(mQ.data(), c.width);
      cast(mK.data(), c.width);
      if(trace){
        (*trace)[n.q] = mQ;
        (*trace)[n.k] = mK;
        (*trace)[n.v] = mV;
      }
      for(int h = 0; h < c.heads; h++){
        std::size_t base = (std::size_t(l) * c.heads + h) * c.window * d;
        for(int i = 0; i < d; i++){
          mKeys[base + slot * d + i] = mK[h * d + i];
          mValues[base + slot * d + i] = mV[h * d + i];
        }
        // Physical ring order is sufficient: attention is invariant to a joint
        // permutation of keys and values. The unfilled cache is a valid prefix.
        using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const Matrix> keys(mKeys.data() + base, valid, d);
        Eigen::Map<const Matrix> values(mValues.data() + base, valid, d);
        Eigen::Map<Eigen::VectorXf> logits(mLogits.data(), valid);
        logits.noalias() = keys * Eigen::Map<const Eigen::VectorXf>(mQ.data() + h * d, d);
        logits /= std::sqrt(float(d));
        float maximum = logits.maxCoeff();
        logits = (logits.array() - maximum).exp();
        float total = logits.sum();
        logits /= total;
        Eigen::Map<Eigen::VectorXf>(mAttention.data() + h * d, d).noalias() = values.transpose() * logits;
      }
      cast(mAttention.data(), c.width);
      auto prefix = trace ? "layers." + std::to_string(l) + "." : std::string();
      if(trace){ (*trace)[prefix + "attention"] = mAttention; }
      linear(mAttention.data(), n.out, mTemp.data(), false, trace);
      for(int i = 0; i < c.width; i++){
        mX[i] += mTemp[i];
      }
      cast(mX.data(), c.width);
      if(trace){ (*trace)[prefix + "attention_residual"] = mX; }
      norm(mX.data(), n.norm2, mNorm.data(), trace);
      linear(mNorm.data(), n.gate, mGate.data(), false, trace);
      linear(mNorm.data(), n.up, mUp.data(), false, trace);
      for(int i = 0; i < c.ffn; i++){
        mGate[i] = (mGate[i] / (1.0f + std::exp(-mGate[i]))) * mUp[i];
      }
      cast(mGate.data(), c.ffn);
      if(trace){ (*trace)[prefix + "activation"] = mGate; }
      linear(mGate.data(), n.down, mTemp.data(), false, trace);
      for(int i = 0; i < c.width; i++){
        mX[i] += mTemp[i];
      }
      cast(mX.data(), c.width);
      if(trace){ (*trace)[prefix + "hidden"] = mX; }
      std::copy(mX.begin(), mX.end(), mHidden.begin() + l * c.width);
    }
    norm(mX.data(), "final_norm", mNorm.data(), trace);
    float score;
    linear(mNorm.data(), "score", &score, true, trace);
    mPosition++;
    return score;
  }

  void Cpu::reset(){
    mPosition = 0;
    std::fill(mKeys.begin(), mKeys.end(), 0);
    std::fill(mValues.begin(), mValues.end(), 0);
    std::fill(mHidden.begin(), mHidden.end(), 0);
  }

  std::uint64_t Cpu::getPosition() const{ return mPosition; }
  const std::vector<float>& Cpu::getKeys() const{ return mKeys; }
  const std::vector<float>& Cpu::getValues() const{ return mValues; }
  const std::vector<float>& Cpu::getHidden() const{ return mHidden; }
}
