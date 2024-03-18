#pragma once

#include <memory>
#include <vector>

// #include "trace.h"
#include "array.h"

namespace ainl::core {

class Array;

void eval(std::vector<Array> &outputs);
} // namespace ainl::core