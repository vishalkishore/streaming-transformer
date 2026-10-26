#include "Runtime.hpp"

#include <fstream>
#include <iostream>

int main(int argc, char** argv){
  try{
    auto result = Stream::probe();
    std::cout << result.dump(2) << '\n';
    if(argc == 2){
      std::ofstream(argv[1]) << result.dump(2) << '\n';
    }
    return 0;
  }catch(const std::exception& error){
    std::cerr << error.what() << '\n';
    return 1;
  }
}
