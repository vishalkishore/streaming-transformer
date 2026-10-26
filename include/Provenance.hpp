#pragma once

#include "Model.hpp"

namespace Stream {
  // Capture outside timing: resolved Git state, current source tree and executable identity.
  Json provenance();
}
