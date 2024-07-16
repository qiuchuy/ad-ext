#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ainl::ffi {

void initTransforms(py::module_ &M);

} // namespace ainl::ffi