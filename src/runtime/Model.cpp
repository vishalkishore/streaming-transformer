#include "Model.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <Eigen/Core>
#include <openssl/sha.h>

namespace Stream {
  ModelConfig ModelConfig::fromJson(const Json& j){
    ModelConfig c{j.at("layers"), j.at("width"), j.at("heads"), j.at("ffn"),
                  j.at("window"), j.at("features"), j.at("epsilon")};
    c.validate();
    return c;
  }

  void ModelConfig::validate() const{
    if(layers < 1 || layers > 16 || width < 1 || width > 1024 || heads < 1 ||
       width % heads || (width / heads) % 2 || ffn < 1 || ffn > 4096 ||
       window < 1 || window > 4096 || features < 1 || features > 1024 ||
       !std::isfinite(epsilon) || epsilon <= 0){
      throw std::runtime_error("Invalid or unsupported model dimensions");
    }
  }

  std::size_t ModelConfig::cacheElements() const{
    return std::size_t(layers) * width * window;
  }

  std::vector<char> readBytes(const std::filesystem::path& path){
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file){
      throw std::runtime_error("Cannot read " + path.string());
    }
    auto length = file.tellg();
    std::vector<char> bytes(static_cast<std::size_t>(length));
    file.seekg(0);
    if(!file.read(bytes.data(), bytes.size())){
      throw std::runtime_error("Short read " + path.string());
    }
    return bytes;
  }

  std::string sha256(const void* data, std::size_t bytes){
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(static_cast<const unsigned char*>(data), bytes, hash);
    std::ostringstream out;
    for(auto byte : hash){
      out << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    }
    return out.str();
  }

  float roundHalf(float value){
    return float(Eigen::half(value));
  }

  Model::Model(const std::filesystem::path& path){
    std::ifstream file(path / "manifest.json");
    file >> manifest;
    if(manifest.at("version") != 1 || manifest.at("byte_order") != "little" ||
       manifest.at("orientation") != "out_in" || std::endian::native != std::endian::little){
      throw std::runtime_error("Unsupported tensor format");
    }
    config = ModelConfig::fromJson(manifest.at("config"));
    auto mode = manifest.at("mode").get<std::string>();
    if(mode != "fp32" && mode != "mixed"){
      throw std::runtime_error("Unknown numerical mode");
    }
    mixed = mode == "mixed";
    auto bytes = readBytes(path / "tensors.bin");
    if(sha256(bytes.data(), bytes.size()) != manifest.at("sha256").get<std::string>()){
      throw std::runtime_error("Checkpoint hash mismatch");
    }
    std::size_t end = 0;
    for(const auto& item : manifest.at("tensors")){
      Tensor tensor;
      tensor.shape = item.at("shape").get<std::vector<int>>();
      std::size_t count = 1;
      for(int dim : tensor.shape){
        if(dim <= 0 || count > bytes.size() / std::size_t(dim)){
          throw std::runtime_error("Invalid tensor dimensions");
        }
        count *= dim;
      }
      auto dtype = item.at("dtype").get<std::string>();
      std::size_t size = dtype == "float32" ? 4 : dtype == "float16" ? 2 : 0;
      std::size_t offset = item.at("offset"), nbytes = item.at("nbytes");
      if(!size || offset != end || nbytes != count * size || offset > bytes.size() || nbytes > bytes.size() - offset){
        throw std::runtime_error("Invalid tensor bounds");
      }
      if(sha256(bytes.data() + offset, nbytes) != item.at("sha256").get<std::string>()){
        throw std::runtime_error("Tensor hash mismatch");
      }
      tensor.data.resize(count);
      if(size == 4){
        std::memcpy(tensor.data.data(), bytes.data() + offset, nbytes);
      }else{
        for(std::size_t i = 0; i < count; i++){
          Eigen::half value;
          std::memcpy(&value, bytes.data() + offset + 2 * i, 2);
          tensor.data[i] = float(value);
        }
      }
      auto name = item.at("name").get<std::string>();
      if(!mTensors.emplace(name, std::move(tensor)).second){
        throw std::runtime_error("Duplicate tensor");
      }
      end += nbytes;
    }
    if(end != bytes.size()){
      throw std::runtime_error("Trailing tensor bytes");
    }
    auto check = [&](const std::string& name, std::vector<int> shape){
      if(get(name).shape != shape){
        throw std::runtime_error("Shape mismatch: " + name);
      }
    };
    check("input", {config.width, config.features});
    check("final_norm", {config.width});
    check("score", {1, config.width});
    for(int l = 0; l < config.layers; l++){
      auto p = "layers." + std::to_string(l) + ".";
      for(auto name : {"norm1", "norm2"}){
        check(p + name, {config.width});
      }
      for(auto name : {"q", "k", "v", "out"}){
        check(p + name, {config.width, config.width});
      }
      for(auto name : {"gate", "up"}){
        check(p + name, {config.ffn, config.width});
      }
      check(p + "down", {config.width, config.ffn});
    }
  }

  const Tensor& Model::get(const std::string& name) const{
    return mTensors.at(name);
  }
}
