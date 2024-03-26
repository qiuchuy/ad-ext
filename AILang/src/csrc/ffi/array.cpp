#include <sstream>

#include "array.h"
#include "ffi/array.h"
#include "utils/logger.h"

namespace ainl::ffi {

py::tuple vector2Tuple(const std::vector<int> &vec) {
  py::tuple result(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    result[i] = vec[i];
  }
  return result;
}

template <typename T>
py::list toPyListRec(ainl::core::Array &arr, size_t offset, size_t dim) {
  py::list result;
  if (arr.ndim() == 0) {
    result.append(*(arr.data<T>() + offset / arr.itemsize()));
    return result;
  }
  if (dim == arr.ndim() - 1) {
    for (size_t i = 0; i < arr.shape().at(dim); i++) {
      result.append(*(arr.data<T>() + offset / arr.itemsize() + i));
    }
  } else {
    for (size_t i = 0; i < arr.shape().at(dim); i++) {
      int dimOffset = 1;
      for (size_t i = dim + 1; i < arr.shape().size(); i++) {
        dimOffset *= arr.shape()[i];
      }
      // auto dimOffset =
      //     std::accumulate(arr.shape().begin() + dim, arr.shape().end(), 1,
      //                     std::multiplies<int>());
      // [TODO] why this causes a segfault?
      result.append(toPyListRec<T>(arr, offset + i * dimOffset * arr.itemsize(),
                                   dim + 1));
    }
  }
  return result;
}

py::object toPyList(ainl::core::Array &arr) {
  if (!arr.evaluated()) {
    arr.eval();
  }

  switch (arr.dtype().type) {
  case ainl::core::Dtype::DataType::BoolType:
    return toPyListRec<bool>(arr, 0, 0);
  case ainl::core::Dtype::DataType::Int8Type:
    return toPyListRec<int8_t>(arr, 0, 0);
  case ainl::core::Dtype::DataType::Int16Type:
    return toPyListRec<int16_t>(arr, 0, 0);
  case ainl::core::Dtype::DataType::Int32Type:
    return toPyListRec<int32_t>(arr, 0, 0);
  case ainl::core::Dtype::DataType::Int64Type:
    return toPyListRec<int64_t>(arr, 0, 0);
  case ainl::core::Dtype::DataType::Float32Type:
    return toPyListRec<float>(arr, 0, 0);
  case ainl::core::Dtype::DataType::Float64Type:
    return toPyListRec<double>(arr, 0, 0);
  default:
    // Handle unknown data type
    // Perhaps throw an exception or return an error
    throw std::invalid_argument("Unknown data type");
  }
}

void initArray(py::module &_m) {
  py::class_<ainl::core::Array>(_m, "array", py::buffer_protocol())
      .def(py::init<>([]() { return ainl::core::Array(1.0f); }))
      .def_buffer([](ainl::core::Array &a) -> py::buffer_info {
        return py::buffer_info(
            a.data<void>(), a.itemsize(),
            py::format_descriptor<ainl::core::Dtype>::format(), a.ndim(),
            a.shape(), a.strides());
      })
      .def("__repr__",
           [](ainl::core::Array &a) {
             std::ostringstream oss;
             oss << a;
             return oss.str();
           })
      .def(
          "__iter__",
          [](ainl::core::Array &a) {
            return py::make_iterator(a.begin(), a.end());
          },
          py::keep_alive<0, 1>())
      .def("__len__",
           [](ainl::core::Array &a) {
             assert(a.ndim() >= 1);
             return a.shape().at(0);
           })
      .def_property_readonly("shape",
                             [](ainl::core::Array &a) {
                               if (!a.evaluated()) {
                                 a.eval();
                               }
                               return vector2Tuple(a.shape());
                             })
      .def_property_readonly("strides",
                             [](ainl::core::Array &a) {
                               if (!a.evaluated()) {
                                 a.eval();
                               }
                               return vector2Tuple(a.strides());
                             })
      .def_property_readonly("data_size", &ainl::core::Array::size)
      .def_property_readonly("dtype", &ainl::core::Array::dtype)
      .def_property_readonly("ndim", &ainl::core::Array::ndim)
      .def("eval", &ainl::core::Array::eval)
      .def("tolist", [](ainl::core::Array &a) { return toPyList(a); });

  _m.def("from_numpy", [](py::buffer arr) {
    py::buffer_info buffer = arr.request();
    ainl::core::Dtype dtype = ainl::core::getDtypeFromFormat(buffer.format);
    auto shape = std::vector<int>(buffer.shape.begin(), buffer.shape.end());
    auto stride =
        std::vector<int>(buffer.strides.begin(), buffer.strides.end());
    auto result = ainl::core::Array(ainl::core::allocator::Buffer(buffer.ptr),
                                    dtype, shape, stride);
    return result;
  });
}

} // namespace ainl::ffi