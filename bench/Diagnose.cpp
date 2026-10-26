#include "Cpu.hpp"
#include "Provenance.hpp"
#include "Runtime.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>

// Dump operation/cast boundaries from the actual CPU implementation, never a timed run.
int main(int argc, char** argv){
  using namespace Stream;
  try{
    if(argc != 3 && argc != 4){ throw std::runtime_error("Usage: stream-diagnose FIXTURE OUTPUT_DIRECTORY [cpu|cuda]"); }
    std::filesystem::path fixture = argv[1], output = argv[2];
    Model model(fixture);
    Cpu cpu(model);
    std::unique_ptr<Runtime> gpu;
    std::string backend = argc == 4 ? argv[3] : "cpu";
    if(backend == "cuda"){
      RuntimeOptions options;
      options.backend = Backend::CUDA;
      options.blocks = 4;
      gpu = std::make_unique<Runtime>(model, options);
    }else if(backend != "cpu"){ throw std::runtime_error("Unknown diagnostic backend"); }
    auto bytes = readBytes(fixture / "trace.bin");
    if(bytes.empty() || bytes.size() % (model.config.features * sizeof(float))){
      throw std::runtime_error("Invalid trace size");
    }
    std::vector<float> events(bytes.size() / sizeof(float));
    std::memcpy(events.data(), bytes.data(), bytes.size());
    std::set<std::size_t> resets;
    Json resetList;
    std::ifstream(fixture / "resets.json") >> resetList;
    for(auto index : resetList){ resets.insert(index.get<std::size_t>()); }
    std::filesystem::create_directories(output);
    std::ofstream data(output / "stages.bin", std::ios::binary);
    data.exceptions(std::ios::badbit | std::ios::failbit);
    Json layout = Json::array();
    std::size_t count = events.size() / model.config.features, eventElements = 0;
    for(std::size_t i = 0; i < count; i++){
      if(resets.contains(i)){ if(gpu){ gpu->reset(); }else{ cpu.reset(); } }
      CpuTrace stages;
      if(gpu){ stages = gpu->traceStep(events.data() + i * model.config.features); }
      else{ cpu.step(events.data() + i * model.config.features, &stages); }
      for(const auto& [name, values] : stages){
        if(i == 0){
          layout.push_back({{"name", name}, {"offset_elements", eventElements}, {"elements", values.size()}});
          eventElements += values.size();
        }
        data.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
      }
    }
    data.close();
    Json manifest = {{"version", 1}, {"dtype", "float32"}, {"layout", "event_major"},
      {"events", count}, {"event_elements", eventElements}, {"stages", layout}, {"backend", backend},
      {"checkpoint_sha256", model.manifest.at("sha256")}, {"compiler", __VERSION__},
      {"trace_sha256", sha256(bytes.data(), bytes.size())}, {"resets", resetList}, {"provenance", provenance()}};
    auto stageBytes = readBytes(output / "stages.bin");
    manifest["sha256"] = sha256(stageBytes.data(), stageBytes.size());
    std::ofstream(output / "manifest.json") << manifest.dump(2) << '\n';
    return 0;
  }catch(const std::exception& error){
    std::cerr << error.what() << '\n';
    return 1;
  }
}
