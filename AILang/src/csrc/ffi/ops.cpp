#include "ffi/ops.h"
#include "array.h"
#include "ops.h"

namespace ainl::ffi {

static inline py::object flatten(const py::object &input) {
  return operatorCallingInterface<ainl::core::FlattenPrimitive>(
      input, ainl::core::flatten);
}

static inline py::object reshape(const py::object &input,
                                 const std::vector<int> &shape) {
  return operatorCallingInterface<ainl::core::ReshapePrimitive>(
      input, [&shape](const ainl::core::Array &input) {
        return ainl::core::reshape(input, shape);
      });
}

static inline py::object slice(const py::object &input,
                               const std::vector<int> &start,
                               const std::vector<int> &end,
                               const std::vector<int> &stride) {
  return operatorCallingInterface<ainl::core::SlicePrimitive>(
      input, [&start, &end, &stride](const ainl::core::Array &input) {
        return ainl::core::slice(input, start, end, stride);
      });
}

void initOps(py::module_ &m) {
  m.def("flatten", &flatten, "Flatten the input Array");
  m.def("reshape", &reshape, "Reshape the input Array");
  m.def("slice", &slice, "Slice the input Array");
}

}; // namespace ainl::ffi