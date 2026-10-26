#include "Cpu.hpp"
#include "Runtime.hpp"
#include "Numerical.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <cstring>

TEST(Config, RejectInvalid){
  auto c = Stream::ModelConfig::fromJson({{"layers", 1}, {"width", 64}, {"heads", 4}, {"ffn", 128},
                                        {"window", 64}, {"features", 32}, {"epsilon", 1e-5}});
  EXPECT_EQ(c.cacheElements(), 4096u);
  c.heads = 3;
  EXPECT_THROW(c.validate(), std::runtime_error);
}

TEST(Format, HashAndHalf){
  EXPECT_EQ(Stream::sha256("abc", 3), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_FLOAT_EQ(Stream::roundHalf(1.0001f), 1.0f);
  EXPECT_TRUE(std::isinf(Stream::roundHalf(1e10f)));
}

TEST(Numerical, ResidualPolicyRetainsCancellationAndRejectsBroadDrift){
  std::vector<float> reference(128, 2), cancellation = reference, biased = reference;
  reference[0] = .003f;
  cancellation[0] = .006f;
  double bound = .002 * Stream::residualScale(reference) + .02 * reference[0];
  EXPECT_GT(.003, .002 + .02 * reference[0]);
  EXPECT_LT(.003, bound);
  EXPECT_LT(Stream::relativeL2(cancellation, reference), .002);
  for(std::size_t i = 0; i < reference.size(); i++){ biased[i] = reference[i] + .02f; }
  EXPECT_GT(Stream::relativeL2(biased, reference), .002);
}

TEST(Cpu, DiagnosticTracePreservesStateAndCastBoundaries){
  for(const std::string size : {"small", "main"}){
    for(const std::string mode : {"fp32", "mixed"}){
      SCOPED_TRACE(size + "-" + mode);
      auto path = std::filesystem::path("fixtures") / (size + "-" + mode);
      ASSERT_TRUE(std::filesystem::exists(path / "manifest.json")) << "Regenerate fixtures with python -m scripts.fixtures";
      Stream::Model model(path);
      Stream::Cpu cpu(model);
      std::vector<float> event(model.config.features), scores;
      std::vector<std::vector<float>> hidden;
      auto bytes = Stream::readBytes(path / "trace.bin");
      int count = 3 * model.config.window + 9;
      for(int i = 0; i < count; i++){
        if(i == 2 * model.config.window + 3){ cpu.reset(); }
        std::memcpy(event.data(), bytes.data() + i * event.size() * sizeof(float), event.size() * sizeof(float));
        scores.push_back(cpu.step(event.data()));
        hidden.push_back(cpu.getHidden());
      }
      auto keys = cpu.getKeys(), values = cpu.getValues();
      // Reuse allocations: Eigen may select different reduction partitions for
      // differently aligned buffers. Only snapshot instrumentation varies here.
      cpu.reset();
      Stream::CpuTrace trace;
      for(int i = 0; i < count; i++){
        if(i == 2 * model.config.window + 3){ cpu.reset(); }
        std::memcpy(event.data(), bytes.data() + i * event.size() * sizeof(float), event.size() * sizeof(float));
        ASSERT_EQ(scores[i], cpu.step(event.data(), &trace)) << "event " << i;
        ASSERT_EQ(hidden[i], cpu.getHidden()) << "event " << i;
        for(int j = 0; j < model.config.width; j++){
          float preCast = trace.at("input.pre_cast")[j];
          ASSERT_EQ(trace.at("input.linear")[j], model.mixed ? Stream::roundHalf(preCast) : preCast);
          ASSERT_EQ(trace.at("layers.0.hidden")[j], cpu.getHidden()[j]);
        }
      }
      ASSERT_EQ(keys, cpu.getKeys());
      ASSERT_EQ(values, cpu.getValues());
    }
  }
}
