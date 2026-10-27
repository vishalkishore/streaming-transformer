#include "Primitives.cuh"
#include "Resources.cuh"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {
  using namespace Stream;
  using namespace Stream::Cuda;
  __global__ void primitive_kernel(Args a, int op){
    if(op == -1){
      matvec(a, a.inputWeight, a.input, a.x, 8, 4);
    }else if(op == -2){
      rms_norm(a, a.x, a.finalNorm, a.norm);
    }else{
      layer_stage(a, 0, op);
    }
  }

  TEST(Primitives, AbsolutePositions){
    int devices = 0;
    if(cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0){ GTEST_SKIP() << "No CUDA GPU"; }
    Args a{};
    a.c = {1, 128, 4, 256, 2, 32, 1e-5f};
    Buffer<float> data(256);
    Buffer<std::uint64_t> position(1);
    a.position = position.get();
    a.q = data.get();
    a.k = data.get() + 128;
    std::vector<float> input(256), output(256);
    for(int i = 0; i < 256; i++){ input[i] = std::sin(float(i)); }
    for(std::uint64_t t : {0ULL, 255ULL, 10007ULL, 1000000ULL}){
      CUDA_CHECK(cudaMemcpy(position.get(), &t, sizeof(t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(data.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
      primitive_kernel<<<1, 128>>>(a, 4);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(output.data(), data.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost));
      for(int i = 0; i < 256; i += 2){
        float angle = float(t) * std::pow(10000.0f, -float(i % 32) / 32);
        float co = std::cos(angle), si = std::sin(angle);
        float expected0 = input[i] * co - input[i + 1] * si;
        float expected1 = input[i] * si + input[i + 1] * co;
        ASSERT_NEAR(output[i], expected0, 1e-5 + 1e-4 * std::abs(expected0)) << "position " << t << " pair " << i;
        ASSERT_NEAR(output[i + 1], expected1, 1e-5 + 1e-4 * std::abs(expected1)) << "position " << t << " pair " << i;
      }
    }
  }

  TEST(Primitives, WarpSoftmaxLengthsAndOffsets){
    int devices = 0;
    if(cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0){ GTEST_SKIP() << "No CUDA GPU"; }
    Args a{};
    a.c = {1, 128, 4, 256, 512, 32, 1e-5f};
    Buffer<float> logits(a.c.heads * a.c.window);
    Buffer<std::uint64_t> position(1);
    a.position = position.get();
    a.logits = logits.get();
    std::vector<float> input(logits.count()), output(logits.count());
    for(int valid : {1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 512}){
      std::uint64_t t = valid - 1;
      CUDA_CHECK(cudaMemcpy(position.get(), &t, sizeof(t), cudaMemcpyHostToDevice));
      for(float offset : {-10000.0f, 0.0f, 10000.0f}){
        std::fill(input.begin(), input.end(), 123456.0f);
        for(int head = 0; head < a.c.heads; head++){
          for(int j = 0; j < valid; j++){
            input[head * a.c.window + j] = offset + std::sin(float(j * 3 + head)) * 7;
          }
        }
        for(int blocks : {1, 4, 110}){
          CUDA_CHECK(cudaMemcpy(logits.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
          primitive_kernel<<<blocks, 128>>>(a, 7);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaMemcpy(output.data(), logits.get(), output.size() * sizeof(float), cudaMemcpyDeviceToHost));
          for(int head = 0; head < a.c.heads; head++){
            const float* source = input.data() + head * a.c.window;
            double maximum = *std::max_element(source, source + valid), total = 0, actualTotal = 0;
            for(int j = 0; j < valid; j++){ total += std::exp(double(source[j]) - maximum); }
            for(int j = 0; j < valid; j++){
              double expected = std::exp(double(source[j]) - maximum) / total;
              float actual = output[head * a.c.window + j];
              ASSERT_NEAR(actual, expected, 1e-6) << "valid=" << valid << " blocks=" << blocks;
              actualTotal += actual;
            }
            ASSERT_NEAR(actualTotal, 1.0, 1e-6);
            for(int j = valid; j < a.c.window; j++){
              ASSERT_EQ(output[head * a.c.window + j], source[j]);
            }
          }
        }
      }
    }
  }

  TEST(Primitives, ParallelValuesWrapAndLengths){
    int devices = 0;
    if(cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0){ GTEST_SKIP() << "No CUDA GPU"; }
    Args a{};
    a.c = {1, 128, 4, 256, 512, 32, 1e-5f};
    Buffer<float> logits(4 * 512), attention(128);
    Buffer<std::uint64_t> position(1);
    a.position = position.get();
    a.logits = logits.get();
    a.attention = attention.get();
    std::vector<float> probabilities(4 * 512), cache(128 * 512), output(128);
    for(bool mixed : {false, true}){
      a.mixed = mixed;
      Buffer<char> values(cache.size() * (mixed ? 2 : 4));
      a.values = values.get();
      for(std::size_t i = 0; i < cache.size(); i++){
        float value = std::sin(float(i) * .13f);
        cache[i] = mixed ? roundHalf(value) : value;
      }
      if(mixed){
        std::vector<__half> half(cache.size());
        for(std::size_t i = 0; i < cache.size(); i++){ half[i] = __float2half(cache[i]); }
        CUDA_CHECK(cudaMemcpy(values.get(), half.data(), half.size() * 2, cudaMemcpyHostToDevice));
      }else{
        CUDA_CHECK(cudaMemcpy(values.get(), cache.data(), cache.size() * 4, cudaMemcpyHostToDevice));
      }
      for(std::uint64_t t : {0ULL, 30ULL, 31ULL, 32ULL, 254ULL, 255ULL, 256ULL, 510ULL, 511ULL, 512ULL, 1027ULL, 10007ULL}){
        int valid = int(std::min<std::uint64_t>(t + 1, 512));
        CUDA_CHECK(cudaMemcpy(position.get(), &t, sizeof(t), cudaMemcpyHostToDevice));
        std::fill(probabilities.begin(), probabilities.end(), 0);
        for(int head = 0; head < 4; head++){
          for(int j = 0; j < valid; j++){
            probabilities[head * 512 + j] = float(j + 1) / (valid * (valid + 1) / 2.0f);
          }
        }
        CUDA_CHECK(cudaMemcpy(logits.get(), probabilities.data(), probabilities.size() * 4, cudaMemcpyHostToDevice));
        for(int blocks : {1, 4, 110}){
          primitive_kernel<<<blocks, 128>>>(a, 8);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaMemcpy(output.data(), attention.get(), output.size() * 4, cudaMemcpyDeviceToHost));
          for(int i = 0; i < 128; i++){
            double expected = 0;
            for(int j = 0; j < valid; j++){
              int slot = (t + 1 - valid + j) % 512;
              expected += double(probabilities[(i / 32) * 512 + j]) * cache[((i / 32) * 512 + slot) * 32 + i % 32];
            }
            float reference = mixed ? roundHalf(expected) : float(expected);
            ASSERT_NEAR(output[i], reference, (mixed ? .002 : 1e-5) + (mixed ? .02 : 1e-4) * std::abs(reference))
              << "position=" << t << " blocks=" << blocks << " mixed=" << mixed;
          }
        }
      }
    }
  }

  TEST(Primitives, Fp32AndMixed){
    int devices = 0;
    if(cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0){ GTEST_SKIP() << "No CUDA GPU"; }
    for(bool mixed : {false, true}){
      Args a{};
      a.c = {1, 8, 2, 8, 3, 4, 1e-5f};
      a.mixed = mixed;
      Buffer<float> data(160);
      Buffer<char> weights(64 * (mixed ? 2 : 4)), keys(24 * (mixed ? 2 : 4)), values(24 * (mixed ? 2 : 4));
      Buffer<std::uint64_t> position(1);
      std::uint64_t t = 3;
      CUDA_CHECK(cudaMemcpy(position.get(), &t, 8, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(data.get(), 0, data.count() * 4));
      CUDA_CHECK(cudaMemset(keys.get(), 0, keys.count()));
      CUDA_CHECK(cudaMemset(values.get(), 0, values.count()));
      a.input = data.get(); a.x = data.get() + 8; a.norm = data.get() + 16;
      a.q = data.get() + 24; a.k = data.get() + 32; a.v = data.get() + 40;
      a.gate = data.get() + 48; a.up = data.get() + 56; a.temp = data.get() + 64;
      a.attention = data.get() + 72; a.logits = data.get() + 80; a.hidden = data.get() + 96;
      a.position = position.get(); a.keys = keys.get(); a.values = values.get();
      a.inputWeight = weights.get(); a.finalNorm = weights.get();
      auto cast = [&](float x){ return mixed ? roundHalf(x) : x; };
      auto upload = [&](void* target, const std::vector<float>& source, bool typed){
        if(typed && mixed){
          std::vector<__half> half(source.size());
          for(std::size_t i = 0; i < source.size(); i++){ half[i] = __float2half(source[i]); }
          CUDA_CHECK(cudaMemcpy(target, half.data(), half.size() * 2, cudaMemcpyHostToDevice));
        }else{
          CUDA_CHECK(cudaMemcpy(target, source.data(), source.size() * 4, cudaMemcpyHostToDevice));
        }
      };
      auto read = [&](float* pointer, int count){
        std::vector<float> output(count);
        CUDA_CHECK(cudaMemcpy(output.data(), pointer, count * 4, cudaMemcpyDeviceToHost));
        return output;
      };
      auto run = [&](int op){
        primitive_kernel<<<1, 128>>>(a, op);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
      };
      std::vector<float> w(64), input{.25f, -.5f, .75f, 1};
      for(int i = 0; i < 64; i++){ w[i] = cast(float(i % 11 - 5) * .125f); }
      upload(weights.get(), w, true);
      upload(a.input, input, false);
      run(-1);
      auto x = read(a.x, 8);
      for(int row = 0; row < 8; row++){
        double expected = 0;
        for(int i = 0; i < 4; i++){ expected += double(w[row * 4 + i]) * input[i]; }
        EXPECT_NEAR(x[row], cast(expected), 1e-6);
      }
      std::vector<float> ones(64, 1);
      upload(weights.get(), ones, true);
      run(-2);
      auto norm = read(a.norm, 8);
      double sum = 0;
      for(float value : x){ sum += double(value) * value; }
      for(int i = 0; i < 8; i++){
        EXPECT_NEAR(norm[i], cast(x[i] / std::sqrt(sum / 8 + 1e-5)), mixed ? 1e-3 : 1e-6);
      }
      std::vector<float> q{1, 2, 3, 4, -1, -2, -3, -4};
      upload(a.q, q, false); upload(a.k, q, false); upload(a.v, q, false);
      run(4);
      auto rotated = read(a.q, 8);
      for(int i = 0; i < 8; i += 2){
        double angle = 3 * std::pow(10000., -double(i % 4) / 4);
        EXPECT_NEAR(rotated[i], cast(q[i] * std::cos(angle) - q[i + 1] * std::sin(angle)), mixed ? 1e-3 : 1e-6);
        EXPECT_NEAR(rotated[i + 1], cast(q[i] * std::sin(angle) + q[i + 1] * std::cos(angle)), mixed ? 1e-3 : 1e-6);
      }
      run(5); run(6); run(7); run(8);
      auto attention = read(a.attention, 8);
      // t=3 overwrites slot 0; the two other slots contain zeros.
      for(int h = 0; h < 2; h++){
        double dot = 0;
        for(int i = 0; i < 4; i++){ dot += double(rotated[h * 4 + i]) * rotated[h * 4 + i]; }
        double probability = 1 / (1 + 2 * std::exp(-dot / 2));
        for(int i = 0; i < 4; i++){
          EXPECT_NEAR(attention[h * 4 + i], cast(probability * q[h * 4 + i]), mixed ? 1e-3 : 1e-6);
        }
      }
      upload(a.gate, q, false); upload(a.up, q, false);
      run(14);
      auto gate = read(a.gate, 8);
      for(int i = 0; i < 8; i++){
        EXPECT_NEAR(gate[i], cast((q[i] / (1 + std::exp(-q[i]))) * q[i]), mixed ? 1e-3 : 2e-6);
      }
      upload(a.temp, q, false);
      run(10);
      auto residual = read(a.x, 8);
      for(int i = 0; i < 8; i++){ EXPECT_NEAR(residual[i], cast(x[i] + q[i]), 1e-6); }
    }
  }
}
