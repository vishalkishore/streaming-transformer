#include "Provenance.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <vector>

namespace Stream {
  namespace {
    std::string gitOutput(const char* command){
      // Commands are fixed literals; no arguments come from the model or CLI.
      auto close = [](FILE* file){ pclose(file); };
      std::unique_ptr<FILE, decltype(close)> pipe(popen(command, "r"), close);
      if(!pipe){ return {}; }
      std::array<char, 4096> buffer;
      std::string output;
      while(auto count = std::fread(buffer.data(), 1, buffer.size(), pipe.get())){
        output.append(buffer.data(), count);
      }
      return output;
    }
  }

  Json provenance(){
    auto commit = gitOutput("git rev-parse --verify HEAD 2>/dev/null");
    if(!commit.empty() && commit.back() == '\n'){ commit.pop_back(); }
    auto status = gitOutput("git status --porcelain=v1 --untracked-files=normal 2>/dev/null");
    Json result = {{"git_commit", commit.empty() ? Json(nullptr) : Json(commit)},
      {"git_status", status}, {"git_dirty", !status.empty()},
      {"source_snapshot_boundary", "current source/config/docs at startup; binary hash identifies the actual executable"}};
    std::vector<std::filesystem::path> paths;
    for(const auto* directory : {"bench", "configs", "deliverables", "include", "reference", "scripts", "src", "tests", "docs", "third_party"}){
      if(!std::filesystem::exists(directory)){ continue; }
      for(const auto& entry : std::filesystem::recursive_directory_iterator(directory)){
        auto path = entry.path();
        if(entry.is_regular_file() && path.extension() != ".pyc"){
          paths.push_back(path);
        }
      }
    }
    for(const auto* name : {"CMakeLists.txt", "pyproject.toml", "uv.lock", "requirements.lock", "README.md", "HANDOFF.md", "AGENTS.md", ".gitignore"}){
      if(std::filesystem::is_regular_file(name)){ paths.emplace_back(name); }
    }
    std::sort(paths.begin(), paths.end());
    std::string snapshot;
    for(const auto& path : paths){
      auto bytes = readBytes(path);
      auto name = path.generic_string();
      snapshot += std::to_string(name.size()) + ":" + name + ":" + sha256(bytes.data(), bytes.size()) + "\n";
    }
    result["source_snapshot_sha256"] = sha256(snapshot.data(), snapshot.size());
    result["source_files"] = paths.size();
    auto executable = readBytes("/proc/self/exe");
    result["executable_sha256"] = sha256(executable.data(), executable.size());
    return result;
  }
}
