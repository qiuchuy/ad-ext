#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ir/node.h"

namespace py = pybind11;

namespace ainl::ffi {

void initArray(py::module_ &m);

#define DEFINE_COMPARE_OPERATOR_ON_SCALAR(TYPE)                                \
  .def("__eq__",                                                               \
       [](const std::shared_ptr<ainl::core::Tracer> &tracer, TYPE scalar) {    \
         auto array = std::make_shared<ainl::core::Array>(scalar);             \
         return ainl::core::unary<ainl::core::ComparePrimitive>(               \
             {tracer, array}, ainl::ir::CompareOp::CompareType::EQ);           \
       })                                                                      \
      .def(                                                                    \
          "__ne__",                                                            \
          [](const std::shared_ptr<ainl::core::Tracer> &tracer, TYPE scalar) { \
            auto array = std::make_shared<ainl::core::Array>(scalar);          \
            return ainl::core::unary<ainl::core::ComparePrimitive>(            \
                {tracer, array}, ainl::ir::CompareOp::CompareType::NE);        \
          })                                                                   \
      .def(                                                                    \
          "__lt__",                                                            \
          [](const std::shared_ptr<ainl::core::Tracer> &tracer, TYPE scalar) { \
            auto array = std::make_shared<ainl::core::Array>(scalar);          \
            return ainl::core::unary<ainl::core::ComparePrimitive>(            \
                {tracer, array}, ainl::ir::CompareOp::CompareType::LT);        \
          })                                                                   \
      .def(                                                                    \
          "__le__",                                                            \
          [](const std::shared_ptr<ainl::core::Tracer> &tracer, TYPE scalar) { \
            auto array = std::make_shared<ainl::core::Array>(scalar);          \
            return ainl::core::unary<ainl::core::ComparePrimitive>(            \
                {tracer, array}, ainl::ir::CompareOp::CompareType::LE);        \
          })                                                                   \
      .def(                                                                    \
          "__gt__",                                                            \
          [](const std::shared_ptr<ainl::core::Tracer> &tracer, TYPE scalar) { \
            auto array = std::make_shared<ainl::core::Array>(scalar);          \
            return ainl::core::unary<ainl::core::ComparePrimitive>(            \
                {tracer, array}, ainl::ir::CompareOp::CompareType::GT);        \
          })                                                                   \
      .def("__ge__", [](const std::shared_ptr<ainl::core::Tracer> &tracer,     \
                        TYPE scalar) {                                         \
        auto array = std::make_shared<ainl::core::Array>(scalar);              \
        return ainl::core::unary<ainl::core::ComparePrimitive>(                \
            {tracer, array}, ainl::ir::CompareOp::CompareType::GE);            \
      })

} // namespace ainl::ffi