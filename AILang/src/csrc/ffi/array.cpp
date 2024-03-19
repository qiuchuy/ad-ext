#include <sstream>

#include "array.h"
#include "ffi/array.h"

namespace ainl::ffi {

void initArray(py::module &_m) {
  py::class_<ainl::core::Array>(_m, "array")
      .def(py::init<>([]() { return ainl::core::Array(1); }))
      .def("__repr__", [](ainl::core::Array &a) {
        std::ostringstream oss;
        oss << a;
        return oss.str();
      });
}

} // namespace ainl::ffi