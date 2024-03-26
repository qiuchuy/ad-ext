#include "ffi/ops.h"
#include "ops.h"

namespace ainl::ffi {

void initOps(py::module_ &m) {
  m.def("flatten", &ainl::core::flatten, "Flatten the input Array");
}

}; // namespace ainl::ffi