#include <sstream>

#include "array.h"
#include "ffi/array.h"

namespace ainl::ffi {

void initArray(py::module &_m) {
  py::class_<ainl::core::Array>(_m, "array", py::buffer_protocol())
      .def(py::init<>([]() { return ainl::core::Array(10.0f); }))
      .def_buffer([](ainl::core::Array &a) -> py::buffer_info {
        return py::buffer_info(a.data()->ptr(), a.size(),
                               py::format_descriptor<float>::format(), a.ndim(),
                               a.shape(), a.strides());
      })
      .def("__repr__",
           [](ainl::core::Array &a) {
             std::ostringstream oss;
             oss << a;
             return oss.str();
           })
      .def_property_readonly("shape", &ainl::core::Array::shape)
      .def_property_readonly("stride", &ainl::core::Array::strides)
      .def_property_readonly("data_size", &ainl::core::Array::size)
      .def_property_readonly("dtype", &ainl::core::Array::dtype)
      .def_property_readonly("ndim", &ainl::core::Array::ndim)
      .def("eval", &ainl::core::Array::eval);

  _m.def("from_numpy", [](py::buffer arr) {
    py::buffer_info buffer = arr.request();
    ainl::core::Dtype dtype = ainl::core::TypeToDtype<float>();
    auto shape = std::vector<int>(buffer.shape.begin(), buffer.shape.end());
    auto stride =
        std::vector<int>(buffer.strides.begin(), buffer.strides.end());
    auto result = ainl::core::Array(ainl::core::allocator::Buffer(buffer.ptr),
                                    dtype, shape, stride);
    return result;
  });
}

} // namespace ainl::ffi