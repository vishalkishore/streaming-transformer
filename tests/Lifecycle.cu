#include "Runtime.hpp"
#include "Cpu.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <chrono>
#include <cstring>
#include <limits>
#include <thread>

namespace {
  bool hasGpu(){
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
  }

  TEST(Persistent, ResetRepeatedScoresAndRestart){
    if(!hasGpu()){ GTEST_SKIP() << "No CUDA GPU"; }
    for(const std::string mode : {"fp32", "mixed"}){
      Stream::Model model(std::filesystem::path("fixtures") / ("small-" + mode));
      for(int blocks : {1, 4}){
        Stream::RuntimeOptions options;
        options.backend = Stream::Backend::PERSISTENT;
        options.blocks = blocks;
        // Each reconstruction starts an independent state and mailbox sequence.
        for(int restart = 0; restart < 2; restart++){
          Stream::Runtime runtime(model, options);
          Stream::Cpu cpu(model);
          std::vector<float> event(model.config.features, 0);
          for(int i = 0; i < 3 * model.config.window + 9; i++){
            if(i == 1 || i == model.config.window || i == 2 * model.config.window + 3){
              runtime.reset();
              runtime.reset();
              cpu.reset();
              EXPECT_EQ(runtime.snapshot().position, 0u);
            }
            // Repeated zero scores still advance and complete distinct sequence IDs.
            EXPECT_FLOAT_EQ(runtime.step(event.data()), cpu.step(event.data()));
            EXPECT_EQ(runtime.snapshot().position, cpu.getPosition());
          }
          runtime.stop();
          runtime.stop();
          EXPECT_THROW(runtime.step(event.data()), std::runtime_error);
          EXPECT_THROW(runtime.reset(), std::runtime_error);
        }
      }
    }
  }

  TEST(Persistent, FiniteAndIdleExit){
    if(!hasGpu()){ GTEST_SKIP() << "No CUDA GPU"; }
    Stream::Model model("fixtures/small-fp32");
    std::vector<float> event(model.config.features, 0);
    for(int blocks : {1, 4}){
      Stream::RuntimeOptions options;
      options.backend = Stream::Backend::PERSISTENT;
      options.blocks = blocks;
      options.maxEvents = 2;
      Stream::Runtime runtime(model, options);
      EXPECT_FLOAT_EQ(runtime.step(event.data()), 0);
      runtime.reset();
      EXPECT_FLOAT_EQ(runtime.step(event.data()), 0);
      EXPECT_THROW(runtime.step(event.data()), std::runtime_error);
      runtime.stop();
    }
    Stream::RuntimeOptions idle;
    idle.backend = Stream::Backend::PERSISTENT;
    idle.idleSeconds = .001;
    Stream::Runtime runtime(model, idle);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_THROW(runtime.step(event.data()), std::runtime_error);
    runtime.stop();
  }

  TEST(Persistent, SequenceExhaustionStillAllowsStop){
    if(!hasGpu()){ GTEST_SKIP() << "No CUDA GPU"; }
    Stream::Model model("fixtures/small-fp32");
    std::vector<float> event(model.config.features, 0);
    for(int blocks : {1, 4}){
      Stream::RuntimeOptions options;
      options.backend = Stream::Backend::PERSISTENT;
      options.blocks = blocks;
      options.initialSequence = std::numeric_limits<std::uint64_t>::max() - 2;
      Stream::Runtime runtime(model, options);
      EXPECT_FLOAT_EQ(runtime.step(event.data()), 0);
      EXPECT_THROW(runtime.step(event.data()), std::runtime_error);
      EXPECT_THROW(runtime.reset(), std::runtime_error);
      EXPECT_NO_THROW(runtime.stop());
      EXPECT_NO_THROW(runtime.stop());
    }
  }

  TEST(Attention, RejectUnsupportedShapes){
    if(!hasGpu()){ GTEST_SKIP() << "No CUDA GPU"; }
    for(bool largeWindow : {false, true}){
      Stream::Model model("fixtures/small-fp32");
      if(largeWindow){
        model.config.window = 513;
      }else{
        model.config.heads = 1;
      }
      Stream::RuntimeOptions options;
      options.fusedAttention = true;
      EXPECT_THROW({ Stream::Runtime runtime(model, options); }, std::runtime_error);
    }
  }

  TEST(Diagnostic, CudaSnapshotsPreserveExecution){
    if(!hasGpu()){ GTEST_SKIP() << "No CUDA GPU"; }
    for(bool fused : {false, true}){
      Stream::Model model("fixtures/main-mixed");
      Stream::RuntimeOptions options;
      options.backend = Stream::Backend::CUDA;
      options.blocks = 4;
      options.fusedAttention = fused;
      Stream::Runtime runtime(model, options);
      auto bytes = Stream::readBytes("fixtures/main-mixed/trace.bin");
      std::vector<float> event(model.config.features), scores;
      for(int i = 0; i < 12; i++){
        std::memcpy(event.data(), bytes.data() + i * event.size() * sizeof(float), event.size() * sizeof(float));
        scores.push_back(runtime.step(event.data()));
      }
      auto state = runtime.snapshot();
      runtime.reset();
      for(int i = 0; i < 12; i++){
        std::memcpy(event.data(), bytes.data() + i * event.size() * sizeof(float), event.size() * sizeof(float));
        auto trace = runtime.traceStep(event.data());
        ASSERT_EQ(trace.at("layers.0.attention").size(), model.config.width);
        ASSERT_EQ(scores[i], trace.at("score.linear")[0]);
      }
      auto traced = runtime.snapshot();
      EXPECT_EQ(state.position, traced.position);
      EXPECT_EQ(state.keys, traced.keys);
      EXPECT_EQ(state.values, traced.values);
      EXPECT_EQ(state.hidden, traced.hidden);
    }
  }
}
