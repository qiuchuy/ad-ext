#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace ainl::ir {

namespace py = pybind11;

void initAINL(py::module_ &m);
} // namespace ainl::ir
