#include "Cpu.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

// Built separately with Eigen allocation assertions enabled, even in Release.
TEST(CpuAllocation, StreamingStepUsesPreallocatedEigenStorage){
  for(const std::string size : {"small", "main", "large"}){
    for(const std::string mode : {"fp32", "mixed"}){
      Stream::Model model(std::filesystem::path("fixtures") / (size + "-" + mode));
      Stream::Cpu cpu(model);
      auto bytes = Stream::readBytes(std::filesystem::path("fixtures") / (size + "-" + mode) / "trace.bin");
      std::vector<float> events(bytes.size() / sizeof(float));
      std::memcpy(events.data(), bytes.data(), bytes.size());
      std::size_t count = events.size() / model.config.features;
      for(std::size_t i = 0; i < count; i++){
        if(i == std::size_t(2 * model.config.window + 3)){ cpu.reset(); }
        Eigen::internal::set_is_malloc_allowed(false);
        float score = cpu.step(events.data() + i * model.config.features);
        Eigen::internal::set_is_malloc_allowed(true);
        ASSERT_TRUE(std::isfinite(score));
      }
    }
  }
}
