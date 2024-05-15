#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "array.h"
#include "trace.h"
#include "transformation.h"

namespace py = pybind11;

namespace ainl::ffi {

void initOps(py::module_ &m);

template <typename TracerTy, typename PrimTy, typename... Args>
py::object op(const std::vector<TracerTy> &inputs, Args &&... args) {
  std::vector<std::shared_ptr<ainl::core::Tracer>> inputs_;
  for (const auto &input : inputs) {
    inputs_.push_back(std::static_pointer_cast<ainl::core::Tracer>(
        std::make_shared<TracerTy>(input)));
  }
  return py::cast(
      TracerTy(inputs_, std::make_shared<PrimTy>(std::forward<Args>(args)...)));
}

} // namespace ainl::ffi