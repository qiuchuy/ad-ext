#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "array.h"

namespace py = pybind11;

namespace ainl::ffi {

void initPyTensor(py::module_ &m);

} // namespace ainl::ffi