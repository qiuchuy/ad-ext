#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dtype.h"

namespace py = pybind11;

namespace ainl::ffi {

template <typename iterable, typename T>
std::vector<T> pythonIterableToVector(const iterable &iter,
                                      std::vector<int> &shape) {
  std::vector<T> result;
  int currentShape = 0;
  for (const auto &item : iter) {
    if (py::isinstance<py::iterable>(item)) {
      auto inner = pythonIterableToVector(item, shape);
      result.insert(result.end(), inner.begin(), inner.end());
    } else {
      result.push_back(py::cast<T>(item));
    }
    currentShape += 1;
  }
  shape.push_back(currentShape);
  return result;
}

void initArray(py::module_ &m);

} // namespace ainl::ffi