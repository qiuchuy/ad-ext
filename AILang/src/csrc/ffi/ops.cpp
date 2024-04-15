#include "ffi/ops.h"
#include "array.h"
#include "ops.h"
#include "transformation.h"

namespace ainl::ffi {

py::object ArrayOperatorInterface::flatten(const py::object &input) {
  return py::cast(ainl::core::flatten(input.cast<ainl::core::Array>()));
}

py::object ArrayOperatorInterface::reshape(const py::object &input,
                                           const std::vector<int> &shape) {
  return py::cast(ainl::core::reshape(input.cast<ainl::core::Array>(), shape));
}

py::object ArrayOperatorInterface::slice(const py::object &input,
                                         const std::vector<int> &start,
                                         const std::vector<int> &end,
                                         const std::vector<int> &stride) {
  return py::cast(
      ainl::core::slice(input.cast<ainl::core::Array>(), start, end, stride));
}

std::shared_ptr<ArrayOperatorInterface> getArrayOperatorInterface() {
  static std::shared_ptr<ArrayOperatorInterface> instance =
      std::make_shared<ArrayOperatorInterface>();
  return instance;
}

TRACER_OPERATOR_INTERFACE_IMPL(JVPTracer)

static inline std::shared_ptr<TracerOperatorInterface>
getTracerOperatorInterface(const py::object &input) {
  if (py::isinstance<ainl::core::Array>(input)) {
    return getArrayOperatorInterface();
  } else if (py::isinstance<ainl::core::JVPTracer>(input)) {
    return getJVPTracerOperatorInterface();
  } else {
    throw std::runtime_error("Unsupported input tracer type, please implement "
                             "the operator interface for this type of tracer.");
  }
}

void initOps(py::module_ &m) {
  m.def(
      "flatten",
      [](const py::object &input) {
        return getTracerOperatorInterface(input)->flatten(input);
      },
      "Flatten the input");

  m.def(
      "reshape",
      [](const py::object &input, const std::vector<int> &shape) {
        return getTracerOperatorInterface(input)->reshape(input, shape);
      },
      "Reshape the input");

  m.def(
      "slice",
      [](const py::object &input, const std::vector<int> &start,
         const std::vector<int> &end, const std::vector<int> &stride) {
        return getTracerOperatorInterface(input)->slice(input, start, end,
                                                        stride);
      },
      "Slice the input");
}

}; // namespace ainl::ffi