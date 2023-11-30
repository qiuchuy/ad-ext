#ifndef AINL_SRC_INCLUDE_BINDING_H
#define AINL_SRC_INCLUDE_BINDING_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

void initAINL(py::module_ &m);

#endif // AINL_SRC_INCLUDE_BINDING_H