#pragma once

#include <memory>
#include <vector>

// #include "ailang/Core/Trace.h"
#include "ailang/Core/Array.h"

namespace ainl::core {

class Array;

void eval(std::vector<Array> &outputs);
} // namespace ainl::core