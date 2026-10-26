#pragma once

#include <algorithm>
#include <cmath>
#include <span>

namespace Stream {
  // Raw residual coordinates have arbitrary scale and can nearly cancel.
  // Retain raw errors separately; use the vector RMS for their absolute error term.
  inline double residualScale(std::span<const float> reference){
    double sum = 0;
    for(float value : reference){ sum += double(value) * value; }
    return std::max(1.0, std::sqrt(sum / reference.size()));
  }

  inline double relativeL2(std::span<const float> actual, std::span<const float> reference){
    double squaredError = 0, squaredReference = 0;
    for(std::size_t i = 0; i < reference.size(); i++){
      double difference = double(actual[i]) - reference[i];
      squaredError += difference * difference;
      squaredReference += double(reference[i]) * reference[i];
    }
    return std::sqrt(squaredError / std::max(1e-30, squaredReference));
  }

  inline double mixedStateScale(std::span<const float> reference, int layer){
    // Cast errors can be correlated across residual layers. A square-root depth
    // factor assumes cancellation that the FP32/FP64 oracle pair does not guarantee.
    return (layer + 1) * residualScale(reference);
  }
}
