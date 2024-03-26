#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ainl::ffi {

void initOps(py::module_ &m);

} // namespace ainl::ffi