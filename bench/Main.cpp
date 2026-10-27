#include "Cpu.hpp"
#include "Runtime.hpp"
#include "Provenance.hpp"
#include "Numerical.hpp"
#include <Eigen/Core>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <sched.h>
#include <sys/resource.h>

namespace {
  using namespace Stream;
  using Clock = std::chrono::steady_clock;
  double now(){ return std::chrono::duration<double, std::nano>(Clock::now().time_since_epoch()).count(); }
  double cpuSeconds(){
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6 + usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6;
  }
  void pin(int core){
    if(core >= 0){
      if(core >= CPU_SETSIZE){ throw std::runtime_error("Invalid CPU core"); }
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(core, &set);
      if(sched_setaffinity(0, sizeof(set), &set)){ throw std::runtime_error("Cannot set CPU affinity"); }
    }
  }
  Json loadJson(const std::filesystem::path& path){
    std::ifstream file(path);
    Json json;
    file >> json;
    return json;
  }
  struct Sample{
    double intended = 0, enqueue = 0, submit = 0, complete = 0;
    float score = 0;
    bool dropped = false;
  };
  Json distribution(std::vector<double> values){
    if(values.empty()){ return Json::object(); }
    std::sort(values.begin(), values.end());
    auto percentile = [&](double q){ return values[std::size_t(std::ceil(q * values.size())) - 1]; };
    return {{"p50_us", percentile(.5)}, {"p95_us", percentile(.95)}, {"p99_us", percentile(.99)},
            {"p999_us_diagnostic", percentile(.999)}, {"worst_us", values.back()}};
  }
  struct Errors{
    double maxAbs = 0, squaredError = 0, squaredReference = 0;
    std::uint64_t count = 0, failed = 0, nonfinite = 0;
    Json examples = Json::array();
    void check(float actual, float expected, bool mixed, double absoluteScale = 1){
      count++;
      if(!std::isfinite(actual) || !std::isfinite(expected)){
        nonfinite++;
        failed++;
        return;
      }
      double diff = std::abs(double(actual) - expected);
      maxAbs = std::max(maxAbs, diff);
      squaredError += diff * diff;
      squaredReference += double(expected) * expected;
      if(diff > (mixed ? 2e-3 * absoluteScale : 1e-5) + (mixed ? 2e-2 : 1e-4) * std::abs(expected)){
        failed++;
        if(examples.size() < 16){ examples.push_back({{"index", count - 1}, {"actual", actual}, {"expected", expected}, {"difference", diff}}); }
      }
    }
    Json json() const{
      double rmsError = std::sqrt(squaredError / std::max<std::uint64_t>(1, count));
      double rmsReference = std::sqrt(squaredReference / std::max<std::uint64_t>(1, count));
      return {{"elements", count}, {"max_abs", maxAbs}, {"relative_l2", std::sqrt(squaredError / std::max(1e-30, squaredReference))},
              {"normalized_rmse", rmsError / std::max(1.0, rmsReference)},
              {"failed", failed}, {"nonfinite", nonfinite}, {"examples", examples}};
    }
  };
  class Expected{
    public:
      explicit Expected(const std::filesystem::path& path): mManifest(loadJson(path / "manifest.json")), mData(readBytes(path / "tensors.bin")){
        for(const auto& tensor : mManifest.at("tensors")){
          std::size_t offset = tensor.at("offset"), bytes = tensor.at("nbytes");
          if(tensor.at("dtype") != "float32" || offset > mData.size() || bytes > mData.size() - offset){ throw std::runtime_error("Invalid expected tensor"); }
          mOffsets.emplace(tensor.at("name").get<std::string>(), std::make_pair(offset, bytes / 4));
        }
        if(mManifest.contains("state_events")){
          mStateEvents = mManifest.at("state_events").get<std::vector<std::size_t>>();
          if(mStateEvents.empty() || mStateEvents.front() != 0 || !std::is_sorted(mStateEvents.begin(), mStateEvents.end()) ||
             std::adjacent_find(mStateEvents.begin(), mStateEvents.end()) != mStateEvents.end()){
            throw std::runtime_error("Invalid sparse oracle state indices");
          }
        }
        if(sha256(mData.data(), mData.size()) != mManifest.at("sha256").get<std::string>()){
          throw std::runtime_error("Expected fixture hash mismatch");
        }
      }
      float get(const std::string& name, std::size_t index) const{
        auto [offset, count] = mOffsets.at(name);
        if(index >= count){ throw std::runtime_error("Expected index out of bounds"); }
        float result;
        std::memcpy(&result, mData.data() + offset + 4 * index, 4);
        return result;
      }
      void copy(const std::string& name, std::size_t index, std::span<float> output) const{
        auto [offset, count] = mOffsets.at(name);
        if(index > count || output.size() > count - index){ throw std::runtime_error("Expected range out of bounds"); }
        std::memcpy(output.data(), mData.data() + offset + sizeof(float) * index, output.size_bytes());
      }
      bool hasState(std::size_t event) const{
        return mStateEvents.empty() || std::binary_search(mStateEvents.begin(), mStateEvents.end(), event);
      }
      std::size_t stateRow(std::size_t event) const{
        if(mStateEvents.empty()){ return event; }
        auto found = std::lower_bound(mStateEvents.begin(), mStateEvents.end(), event);
        if(found == mStateEvents.end() || *found != event){ throw std::runtime_error("Missing sparse oracle snapshot"); }
        return found - mStateEvents.begin();
      }
      void validateSparse(const Model& model, std::size_t count, int resetEvery, const std::string& traceHash) const{
        if(mManifest.at("format") != "stream-sparse-oracle-v1" || mManifest.at("events") != count ||
           mManifest.at("reset_every") != resetEvery || mManifest.at("config") != model.manifest.at("config") ||
           mManifest.at("mode") != model.manifest.at("mode") || mManifest.at("checkpoint_sha256") != model.manifest.at("sha256") ||
           mManifest.at("trace_sha256") != traceHash || mStateEvents.empty() || mStateEvents.back() != count - 1){
          throw std::runtime_error("Sparse CPU oracle differs from requested GPU replay");
        }
      }
    private:
      Json mManifest;
      std::vector<char> mData;
      std::vector<std::size_t> mStateEvents;
      std::map<std::string, std::pair<std::size_t, std::size_t>> mOffsets;
  };
}

int main(int argc, char** argv){
  using namespace Stream;
  try{
    std::map<std::string, std::string> args;
    const std::set<std::string> allowed{"config", "backend", "weights", "trace", "blocks", "max-events", "events", "warmup", "mode", "output", "transport", "backoff", "idle-seconds", "threads", "rate", "bursty", "queue", "deadline-us", "worker-core", "producer-core", "reset-every", "clock-regime", "expected", "idle-us"};
    for(int i = 1; i < argc; i += 2){
      std::string key = argv[i];
      if(key == "--help"){
        std::cout << "stream-bench --config configs/main.json --weights fixtures/main-fp32 --trace fixtures/main-fp32/trace.bin\n"
                     "  --backend cpu|cuda|cublas|graph|cublas-graph|fused|persistent\n"
                     "  --mode correctness|service|scheduled|device|ping-pong|empty|empty-graph|copy|gpu-ping\n"
                     "  --events N --warmup N --blocks N --transport copy|mapped --backoff 0|1\n"
                     "  --rate events_per_second --bursty 0|1 --queue N --output prefix\n";
        return 0;
      }
      if(key.rfind("--", 0) || !allowed.contains(key.substr(2)) || i + 1 >= argc){
        throw std::runtime_error("Unknown option or missing value: " + key);
      }
      if(!args.emplace(key.substr(2), argv[i + 1]).second){ throw std::runtime_error("Duplicate option"); }
    }
    auto arg = [&](std::string key, std::string fallback){ return args.contains(key) ? args.at(key) : fallback; };
    auto integer = [&](std::string key, int fallback){ return std::stoi(arg(key, std::to_string(fallback))); };
    auto mode = arg("mode", "service"), backend = arg("backend", "graph"), output = arg("output", "results/run");
    std::filesystem::create_directories(std::filesystem::path(output).parent_path());
    std::size_t count = std::stoull(arg("events", "1000"));
    if(!count || count > 100000000){ throw std::runtime_error("Invalid event count"); }
    pin(integer("worker-core", -1));
    if(mode == "ping-pong" || mode == "empty" || mode == "empty-graph" || mode == "copy" || mode == "gpu-ping"){
      auto result = transportBench(mode, count, integer("backoff", 0), output);
      std::ofstream(output + ".json") << result.dump(2) << '\n';
      std::cout << result.dump(2) << '\n';
      return 0;
    }
    if(mode != "correctness" && mode != "service" && mode != "scheduled" && mode != "device"){
      throw std::runtime_error("Unknown measurement mode");
    }
    auto startup = now();
    std::filesystem::path weights = arg("weights", "fixtures/small-fp32");
    Model model(weights);
    auto configPath = arg("config", "configs/small.json");
    auto configBytes = readBytes(configPath);
    if(Json::parse(configBytes) != model.manifest.at("config")){
      throw std::runtime_error("Configuration differs from checkpoint");
    }
    auto tracePath = arg("trace", (weights / "trace.bin").string());
    auto traceBytes = readBytes(tracePath);
    if(traceBytes.empty() || traceBytes.size() % (model.config.features * 4)){
      throw std::runtime_error("Invalid trace byte count");
    }
    std::vector<float> trace(traceBytes.size() / 4);
    std::memcpy(trace.data(), traceBytes.data(), traceBytes.size());
    std::size_t traceCount = trace.size() / model.config.features;
    auto input = [&](std::size_t i){ return trace.data() + (i % traceCount) * model.config.features; };
    bool cpuOracle = mode == "correctness" && arg("expected", "fixture") == "cpu";
    bool storedOracle = mode == "correctness" && arg("expected", "fixture") != "cpu" && arg("expected", "fixture") != "fixture";
    if(backend == "cpu" && cpuOracle){ throw std::runtime_error("CPU correctness requires an independent fixture"); }
    std::unique_ptr<Cpu> cpu;
    if(backend == "cpu" || cpuOracle){ cpu = std::make_unique<Cpu>(model, integer("threads", 1)); }
    std::unique_ptr<Runtime> gpu;
    Json metadata;
    RuntimeOptions options;
    std::map<std::string, Backend> backends{{"cuda", Backend::CUDA}, {"cublas", Backend::CUBLAS},
      {"graph", Backend::GRAPH}, {"cublas-graph", Backend::CUBLAS_GRAPH}, {"fused", Backend::FUSED}, {"persistent", Backend::PERSISTENT}};
    if(backend != "cpu"){
      options.backend = backends.at(backend);
      options.blocks = integer("blocks", 1);
      options.maxEvents = std::stoull(arg("max-events", "0"));
      options.idleSeconds = std::stod(arg("idle-seconds", "30"));
      options.backoff = integer("backoff", 0);
      auto transport = arg("transport", "copy");
      if(transport != "copy" && transport != "mapped"){ throw std::runtime_error("Unknown transport"); }
      options.mapped = transport == "mapped";
      gpu = std::make_unique<Runtime>(model, options);
      metadata = gpu->info();
    }
    auto reset = [&](){ if(gpu){ gpu->reset(); }else{ cpu->reset(); } };
    auto step = [&](const float* event){ return gpu ? gpu->step(event) : cpu->step(event); };
    metadata["startup_ms"] = (now() - startup) / 1e6;
    metadata["backend"] = backend;
    metadata["mode"] = mode;
    metadata["config"] = model.manifest.at("config");
    metadata["precision"] = model.manifest.at("mode");
    metadata["checkpoint_sha256"] = model.manifest.at("sha256");
    metadata["config_sha256"] = sha256(configBytes.data(), configBytes.size());
    metadata["trace_sha256"] = sha256(traceBytes.data(), traceBytes.size());
    metadata["clock_regime"] = arg("clock-regime", "default-uncontrolled-shared-GPU");
    metadata["compiler"] = __VERSION__;
    metadata["cpu_threads_requested"] = integer("threads", 1);
    metadata["cpu_threads_actual"] = cpu ? Eigen::nbThreads() : 0;
    metadata["cpu_oracle_active"] = cpuOracle;
    metadata["acceptance_policy"] = "mixed-depth-rms-v3: atol=0.002*(layer+1)*max(1,vector RMS), rtol=0.02; complete per-layer state relative L2<=0.002; raw coordinate/head errors retained; score atol=0.002*layers and aggregate normalized RMSE<=0.002 (unit RMS floor)";
    metadata["worker_core"] = integer("worker-core", -1);
    metadata["producer_core"] = integer("producer-core", -1);
    metadata["provenance"] = provenance();
    if(mode == "device"){
      if(!gpu){ throw std::runtime_error("Device mode requires a GPU backend"); }
      std::vector<float> inputs(count * model.config.features);
      for(std::size_t i = 0; i < count; i++){
        std::copy_n(input(i), model.config.features, inputs.data() + i * model.config.features);
      }
      metadata["device_us_per_event"] = gpu->deviceTrace(inputs.data(), count);
      metadata["device_boundary"] = "preloaded trace; fused loop or native sequence including D2D input staging";
      metadata["events"] = count;
      std::ofstream(output + ".json") << metadata.dump(2) << '\n';
      std::cout << metadata.dump(2) << '\n';
      return 0;
    }
    int warmup = integer("warmup", mode == "correctness" ? 0 : 100);
    if(warmup < 0){ throw std::runtime_error("Negative warmup"); }
    if(options.maxEvents && options.maxEvents < count + warmup){
      throw std::runtime_error("max-events must cover warmup plus measured RUN commands");
    }
    for(int i = 0; i < warmup; i++){ step(input(i)); }
    reset();
    if(cpu){ cpu->reset(); }
    std::set<std::size_t> resetIndices;
    int resetEvery = integer("reset-every", 0);
    if(mode == "correctness" && !cpuOracle && !storedOracle && std::filesystem::exists(weights / "resets.json")){
      for(auto index : loadJson(weights / "resets.json")){ resetIndices.insert(index.get<std::size_t>()); }
    }
    std::unique_ptr<Expected> expected;
    if(mode == "correctness" && arg("expected", "fixture") != "cpu"){
      std::filesystem::path expectedPath = storedOracle ? std::filesystem::path(arg("expected", "")) : weights / "expected";
      expected = std::make_unique<Expected>(expectedPath);
      if(storedOracle){ expected->validateSparse(model, count, resetEvery, metadata["trace_sha256"]); }
      else if(count > traceCount){ throw std::runtime_error("Fixture correctness cannot exceed trace; use a stored CPU oracle for stress"); }
      auto expectedManifest = readBytes(expectedPath / "manifest.json");
      metadata["expected_manifest_sha256"] = sha256(expectedManifest.data(), expectedManifest.size());
    }
    metadata["expected_source"] = arg("expected", "fixture");
    Errors scores, scoresRaw, keys, keysRaw, values, valuesRaw, hidden, hiddenRaw, hiddenRelative, cacheRelative, cacheHeadRelative;
    std::vector<Sample> samples(count);
    double startCpu = cpuSeconds(), startWall = now();
    std::size_t dropped = 0;
    double idleUs = std::stod(arg("idle-us", "0"));
    if(mode == "scheduled"){
      double rate = std::stod(arg("rate", "1000"));
      std::size_t capacity = std::stoull(arg("queue", "4096"));
      if(rate <= 0 || !std::isfinite(rate) || !capacity){ throw std::runtime_error("Invalid rate or queue capacity"); }
      bool bursty = integer("bursty", 0);
      std::vector<std::size_t> queue(capacity);
      std::vector<double> schedule(count);
      for(std::size_t i = 0; i < count; i++){
        schedule[i] = (bursty ? (i / 16) * 16 : i) * 1e9 / rate;
      }
      std::atomic<std::size_t> head = 0, tail = 0;
      std::atomic<bool> done = false;
      double epoch = now() + 1e8;
      int producerCore = integer("producer-core", -1);
      std::exception_ptr producerError;
      std::thread producer([&](){
        try{
          pin(producerCore);
          for(std::size_t i = 0; i < count; i++){
            samples[i].intended = epoch + schedule[i];
            while(now() < samples[i].intended){ std::this_thread::yield(); }
            samples[i].enqueue = now();
            auto write = tail.load(std::memory_order_relaxed);
            if(write - head.load(std::memory_order_acquire) == capacity){
              samples[i].dropped = true;
              dropped++;
            }else{
              queue[write % capacity] = i;
              tail.store(write + 1, std::memory_order_release);
            }
          }
        }catch(...){ producerError = std::current_exception(); }
        done.store(true, std::memory_order_release);
      });
      try{
        while(true){
          auto read = head.load(std::memory_order_relaxed);
          if(read == tail.load(std::memory_order_acquire)){
            if(done.load(std::memory_order_acquire)){
              // Re-read tail after acquiring done so the final queued item cannot be lost.
              if(read == tail.load(std::memory_order_acquire)){ break; }
            }
            std::this_thread::yield();
            continue;
          }
          auto i = queue[read % capacity];
          head.store(read + 1, std::memory_order_release);
          samples[i].submit = now();
          samples[i].score = step(input(i));
          samples[i].complete = now();
        }
      }catch(...){
        producer.join();
        throw;
      }
      producer.join();
      if(producerError){ std::rethrow_exception(producerError); }
      metadata["rate_per_second"] = rate;
      metadata["schedule_sha256"] = sha256(schedule.data(), schedule.size() * sizeof(double));
      metadata["bursty"] = bursty;
      metadata["queue_capacity"] = capacity;
    }else{
      for(std::size_t i = 0; i < count; i++){
        if((resetEvery > 0 && i && i % resetEvery == 0) || resetIndices.contains(i)){
          reset();
          if(gpu && cpu){ cpu->reset(); }
        }
        if(idleUs > 0){ std::this_thread::sleep_for(std::chrono::duration<double, std::micro>(idleUs)); }
        auto& s = samples[i];
        s.intended = s.enqueue = s.submit = now();
        s.score = step(input(i));
        s.complete = now();
        if(mode == "correctness"){
          float oracle = expected ? expected->get("scores", i) : gpu ? cpu->step(input(i)) : s.score;
          scoresRaw.check(s.score, oracle, model.mixed);
          scores.check(s.score, oracle, model.mixed, model.config.layers);
          bool fullCheck = expected ? expected->hasState(i) : i % 4096 == 0 || i + 1 == count;
          auto state = fullCheck ? (gpu ? gpu->snapshot() : Snapshot{cpu->getPosition(), cpu->getKeys(), cpu->getValues(), cpu->getHidden()}) : Snapshot{};
          std::uint64_t expectedPosition = i + 1;
          auto lastReset = resetIndices.upper_bound(i);
          if(lastReset != resetIndices.begin()){ expectedPosition -= *std::prev(lastReset); }
          if(resetEvery > 0){ expectedPosition = i % resetEvery + 1; }
          if(fullCheck && state.position != expectedPosition){ throw std::runtime_error("Position drift"); }
          // Full state on fixture runs. CPU stress checks all scores and periodic cache snapshots.
          if(fullCheck){
            std::size_t stateRow = expected ? expected->stateRow(i) : i;
            auto check = [&](const std::vector<float>& actual, const std::vector<float>& native,
                             const std::string& name, Errors& errors, Errors& raw){
              int dim = model.config.width / model.config.heads;
              std::vector<float> referenceData(actual.size());
              if(expected){ expected->copy(name, stateRow * actual.size(), referenceData); }
              else{ referenceData = native; }
              for(std::size_t base = 0; base < actual.size(); base += dim){
                int layer = base / (model.config.width * model.config.window);
                std::span<const float> reference(referenceData.data() + base, dim);
                double scale = model.mixed ? mixedStateScale(reference, layer) : 1;
                for(int j = 0; j < dim; j++){
                  raw.check(actual[base + j], reference[j], model.mixed);
                  errors.check(actual[base + j], reference[j], model.mixed, scale);
                }
                if(model.mixed){ cacheHeadRelative.check(relativeL2(std::span<const float>(actual.data() + base, dim), reference), 0, true); }
              }
              if(model.mixed){
                std::size_t layerElements = model.config.width * model.config.window;
                for(int layer = 0; layer < model.config.layers; layer++){
                  std::span<const float> actualLayer(actual.data() + layer * layerElements, layerElements);
                  std::span<const float> referenceLayer(referenceData.data() + layer * layerElements, layerElements);
                  cacheRelative.check(relativeL2(actualLayer, referenceLayer), 0, true);
                }
              }
            };
            const std::vector<float> unused;
            check(state.keys, cpu ? cpu->getKeys() : unused, "keys", keys, keysRaw);
            check(state.values, cpu ? cpu->getValues() : unused, "values", values, valuesRaw);
            std::vector<float> referenceHidden(state.hidden.size());
            if(expected){ expected->copy("hidden", stateRow * referenceHidden.size(), referenceHidden); }
            else{ referenceHidden = cpu->getHidden(); }
            for(int layer = 0; layer < model.config.layers; layer++){
              std::span<const float> actual(state.hidden.data() + layer * model.config.width, model.config.width);
              std::span<const float> reference(referenceHidden.data() + layer * model.config.width, model.config.width);
              double scale = model.mixed ? mixedStateScale(reference, layer) : 1;
              for(int j = 0; j < model.config.width; j++){
                hiddenRaw.check(actual[j], reference[j], model.mixed);
                hidden.check(actual[j], reference[j], model.mixed, scale);
              }
              if(model.mixed){ hiddenRelative.check(relativeL2(actual, reference), 0, true); }
            }
          }
        }
        if(mode == "correctness" && (i + 1) % 10000 == 0){
          std::cerr << "Checked " << i + 1 << '/' << count << " RUN scores; CPU oracle active=" << cpuOracle << '\n';
        }
      }
    }
    double elapsed = (now() - startWall) / 1e9;
    metadata["cpu_seconds"] = cpuSeconds() - startCpu;
    metadata["wall_seconds"] = elapsed;
    metadata["cpu_core_equivalents"] = metadata["cpu_seconds"].get<double>() / elapsed;
    metadata["events"] = count;
    metadata["completed"] = count - dropped;
    metadata["dropped"] = dropped;
    metadata["warmup"] = warmup;
    metadata["service_rate_per_second"] = (count - dropped) / elapsed;
    std::vector<double> service, arrival, queueing, injection;
    std::size_t misses = 0;
    double deadline = std::stod(arg("deadline-us", "1000"));
    std::ofstream csv(output + ".csv");
    csv << "event,intended_ns,enqueue_ns,submit_ns,complete_ns,score,dropped\n";
    csv.precision(17);
    for(std::size_t i = 0; i < count; i++){
      const auto& s = samples[i];
      csv << i << ',' << s.intended << ',' << s.enqueue << ',' << s.submit << ',' << s.complete << ',' << s.score << ',' << s.dropped << '\n';
      injection.push_back((s.enqueue - s.intended) / 1000);
      if(!s.dropped){
        service.push_back((s.complete - s.submit) / 1000);
        arrival.push_back((s.complete - s.intended) / 1000);
        queueing.push_back((s.submit - s.enqueue) / 1000);
        if(arrival.back() > deadline){ misses++; }
      }
    }
    metadata["service"] = distribution(service);
    metadata["arrival"] = distribution(arrival);
    metadata["queueing"] = distribution(queueing);
    metadata["injection"] = distribution(injection);
    metadata["deadline_us"] = deadline;
    metadata["deadline_misses"] = misses;
    metadata["scores"] = scores.json();
    metadata["scores_raw"] = scoresRaw.json();
    metadata["keys"] = keys.json();
    metadata["values"] = values.json();
    metadata["keys_raw"] = keysRaw.json();
    metadata["values_raw"] = valuesRaw.json();
    metadata["cache_relative_l2_gate"] = cacheRelative.json();
    metadata["cache_head_relative_l2_diagnostic"] = cacheHeadRelative.json();
    metadata["hidden"] = hidden.json();
    metadata["hidden_raw"] = hiddenRaw.json();
    metadata["hidden_relative_l2_gate"] = hiddenRelative.json();
    metadata["timing_clock"] = "std::chrono::steady_clock; all host timestamps share this clock";
    metadata["final_condition_sample_gate"] = count - dropped >= 100000;
    if(gpu){ gpu->stop(); }
    std::ofstream(output + ".json") << metadata.dump(2) << '\n';
    std::cout << metadata.dump(2) << '\n';
    bool scoreNormFailed = model.mixed && scores.json()["normalized_rmse"].get<double>() > .002;
    return scores.failed || scoreNormFailed || keys.failed || values.failed || hidden.failed || hiddenRelative.failed || cacheRelative.failed ? 2 : 0;
  }catch(const std::exception& error){
    std::cerr << error.what() << '\n';
    return 1;
  }
}
