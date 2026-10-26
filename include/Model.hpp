#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace Stream {
  using Json = nlohmann::json;
  struct ModelConfig{
    int layers, width, heads, ffn, window, features;
    float epsilon;
    static ModelConfig fromJson(const Json& json);
    void validate() const;
    std::size_t cacheElements() const;
  };

  struct Tensor{
    std::vector<int> shape;
    std::vector<float> data;
  };

  // Loader validates hashes and exact shape contracts before exposing tensor views.
  class Model{
    public:
      explicit Model(const std::filesystem::path& path);
      const Tensor& get(const std::string& name) const;
      ModelConfig config;
      bool mixed;
      Json manifest;
    private:
      std::map<std::string, Tensor> mTensors;
  };

  std::string sha256(const void* data, std::size_t bytes);
  std::vector<char> readBytes(const std::filesystem::path& path);
  float roundHalf(float value);
}
