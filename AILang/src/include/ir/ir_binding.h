

#ifndef AINL_SRC_INCLUDE_IR_BINDING_H
#define AINL_SRC_INCLUDE_IR_BINDING_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void initIR(py::module_ &m);

#endif
