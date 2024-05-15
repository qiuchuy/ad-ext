#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dtype.h"

namespace py = pybind11;

namespace ainl::ffi {

void initArray(py::module_ &m);

#define DEFINE_COMPARE_OPERATOR_ON_SCALAR(TYPE)                                \
  .def("__gt__", [](ainl::core::Tracer &tracer,                                \
                    TYPE scalar) { return tracer > scalar; })                  \
      .def("__eq__", [](ainl::core::Tracer &tracer,                            \
                        TYPE scalar) { return tracer == scalar; })             \
      .def("__lt__",                                                           \
           [](ainl::core::Tracer &tracer, TYPE scalar) {                       \
             return !(tracer > scalar) && !(tracer == scalar);                 \
           })                                                                  \
      .def("__le__",                                                           \
           [](ainl::core::Tracer &tracer, TYPE scalar) {                       \
             return !(tracer > scalar) || tracer == scalar;                    \
           })                                                                  \
      .def("__ge__",                                                           \
           [](ainl::core::Tracer &tracer, TYPE scalar) {                       \
             return tracer > scalar || tracer == scalar;                       \
           })                                                                  \
      .def("__ne__", [](ainl::core::Tracer &tracer, TYPE scalar) {             \
        return !(tracer == scalar);                                            \
      })

} // namespace ainl::ffi