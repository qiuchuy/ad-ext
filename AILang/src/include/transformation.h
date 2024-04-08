#pragma once
#include "array.h"
#include "ir/function.h"

namespace ainl::core {

/*
 * JVP (Jacobian-Vector Product) transformation
 */
template <typename FuncType>
void jvp(const FuncType &func, const std::vector<Array> &inputs) {
  // Algorithm:
  // 1. Push `JVPTrace` onto the trace stack
  // 2. for each input in `inputs`:
  // 2.1 Create a `JVPTracer` wrapper around the input
  // 3. Pop `JVPTrace` from the trace stack
}

} // namespace ainl::core