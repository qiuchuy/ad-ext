#include "ffi/pytensor.h"

namespace ainl::ffi {

void initPyTensor(py::module &_m) {
  py::class_<ainl::core::Array>(_m, "array")
      .def(py::init([]() { return ainl::core::makeArrayFromScalar(1); }))
      .def("__repr__", [](const ainl::core::Array &a) {
        return std::string("hello ailang");
      });
}

} // namespace ainl::ffi