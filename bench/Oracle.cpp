#include "Cpu.hpp"
#include "Model.hpp"
#include "Provenance.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

// Generate one reusable CPU trace; this executable has no CUDA dependency.
int main(int argc, char** argv){
  using namespace Stream;
  try{
    std::map<std::string, std::string> args;
    for(int i = 1; i < argc; i += 2){
      if(i + 1 >= argc || !args.emplace(argv[i], argv[i + 1]).second){
        throw std::runtime_error("Expected unique option/value pairs");
      }
    }
    const std::set<std::string> allowed{"--weights", "--trace", "--events", "--reset-every", "--snapshot-every", "--output"};
    for(const auto& [key, value] : args){
      if(!allowed.contains(key)){ throw std::runtime_error("Unknown oracle option: " + key); }
    }
    auto option = [&](const std::string& key, const std::string& fallback){
      return args.contains(key) ? args.at(key) : fallback;
    };
    std::filesystem::path weights = args.at("--weights"), output = args.at("--output");
    std::size_t count = std::stoull(option("--events", "1000000"));
    std::size_t resetEvery = std::stoull(option("--reset-every", "10007"));
    std::size_t snapshotEvery = std::stoull(option("--snapshot-every", "4096"));
    if(!count || count > 100000000 || !snapshotEvery){ throw std::runtime_error("Invalid oracle event/snapshot count"); }
    Model model(weights);
    auto traceBytes = readBytes(option("--trace", (weights / "trace.bin").string()));
    std::size_t eventBytes = model.config.features * sizeof(float);
    if(traceBytes.empty() || traceBytes.size() % eventBytes){ throw std::runtime_error("Invalid oracle trace"); }
    std::vector<float> trace(traceBytes.size() / sizeof(float));
    std::memcpy(trace.data(), traceBytes.data(), traceBytes.size());
    Cpu cpu(model, 1);
    std::vector<float> scores(count), keys, values, hidden;
    std::vector<std::size_t> stateEvents;
    std::size_t snapshots = (count - 1) / snapshotEvery + 2;
    keys.reserve(snapshots * model.config.cacheElements());
    values.reserve(snapshots * model.config.cacheElements());
    hidden.reserve(snapshots * model.config.layers * model.config.width);
    for(std::size_t i = 0; i < count; i++){
      if(resetEvery && i && i % resetEvery == 0){ cpu.reset(); }
      scores[i] = cpu.step(trace.data() + (i % (trace.size() / model.config.features)) * model.config.features);
      if(!std::isfinite(scores[i])){ throw std::runtime_error("Nonfinite CPU oracle score"); }
      if(i % snapshotEvery == 0 || i + 1 == count){
        stateEvents.push_back(i);
        const auto& k = cpu.getKeys();
        const auto& v = cpu.getValues();
        const auto& h = cpu.getHidden();
        keys.insert(keys.end(), k.begin(), k.end());
        values.insert(values.end(), v.begin(), v.end());
        hidden.insert(hidden.end(), h.begin(), h.end());
      }
      if((i + 1) % 10000 == 0){ std::cerr << "CPU oracle " << i + 1 << '/' << count << '\n'; }
    }
    Json manifest{{"format", "stream-sparse-oracle-v1"}, {"events", count}, {"reset_every", resetEvery},
      {"snapshot_every", snapshotEvery}, {"state_events", stateEvents}, {"config", model.manifest.at("config")},
      {"provenance", provenance()},
      {"mode", model.mixed ? "mixed" : "fp32"}, {"checkpoint_sha256", model.manifest.at("sha256")},
      {"trace_sha256", sha256(traceBytes.data(), traceBytes.size())}, {"tensors", Json::array()}};
    std::filesystem::create_directories(output);
    std::ofstream data(output / "tensors.bin", std::ios::binary);
    std::size_t offset = 0;
    auto write = [&](const std::string& name, const std::vector<float>& tensor){
      std::size_t bytes = tensor.size() * sizeof(float);
      data.write(reinterpret_cast<const char*>(tensor.data()), bytes);
      if(!data){ throw std::runtime_error("Failed writing CPU oracle"); }
      manifest["tensors"].push_back({{"name", name}, {"dtype", "float32"}, {"offset", offset}, {"nbytes", bytes}});
      offset += bytes;
    };
    write("scores", scores);
    write("keys", keys);
    write("values", values);
    write("hidden", hidden);
    data.close();
    auto bytes = readBytes(output / "tensors.bin");
    manifest["sha256"] = sha256(bytes.data(), bytes.size());
    std::ofstream(output / "manifest.json") << manifest.dump(2) << '\n';
    std::cout << "CPU oracle complete: " << count << " scores, " << stateEvents.size() << " state snapshots\n";
  }catch(const std::exception& error){
    std::cerr << error.what() << '\n';
    return 1;
  }
}
