#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "array.h"
#include "trace.h"

namespace py = pybind11;

namespace ainl::ffi {

template <typename Primitive, typename FuncType>
py::object operatorCallingInterface(const py::object &input, FuncType func) {
  if (py::isinstance<ainl::core::Array>(input)) {
    return py::cast(func(input.cast<ainl::core::Array>()));
  } else {
    return py::cast(ainl::core::Tracer({std::make_shared<ainl::core::Tracer>(
                                           input.cast<ainl::core::Tracer>())},
                                       std::make_shared<Primitive>()));
  }
}

void initOps(py::module_ &m);

} // namespace ainl::ffi