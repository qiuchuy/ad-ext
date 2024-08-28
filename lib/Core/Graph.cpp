#include "ailang/Core/Graph.h"

#include "ailang/Core/Array.h"
#include "ailang/Core/Trace.h"

namespace ainl::core {

void eval(std::vector<Array> &outputs) {
  //     std::function<void(Array &)> recursion = [&](Array &arr) -> void {
  //         if (arr.evaluated()) {
  //             return;
  //         } else {
  //             for (auto &input : arr.getInputs()) {
  //                 recursion(input);
  //             }
  //             auto trace = getTopTrace();
  //             trace->process(arr.getPrimitive(), arr.getInputs(), arr);
  //         }
  //     };
  //
  //     for (auto &output : outputs) {
  //         recursion(output);
  //     }
  //
}
} // namespace ainl::core