#include "ailang/Core/Array.h"
#include "ailang/Core/Device.h"
#include "ailang/Core/Ops.h"
#include "ailang/Core/Primitive.h"
#include "ailang/Core/Transformation.h"
#include "ailang/IR/Function.h"
#include "ailang/Transforms/StablehloConversion.h"

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace ainl::core;
using namespace ainl::ir;

namespace {

py::tuple vector_to_tuple(const std::vector<int> &vec) {
  py::tuple result(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    result[i] = vec[i];
  }
  return result;
}

template <typename T>
py::list to_pylist_rec(Array &arr, size_t offset, size_t dim) {
  py::list result;
  if (arr.ndim() == 0) {
    result.append(*(arr.data<T>() + offset / arr.itemsize()));
    return result;
  }
  if (dim == arr.ndim() - 1) {
    for (size_t i = 0; i < arr.shape().at(dim); i++) {
      result.append(*(arr.data<T>() +
                      (offset + i * arr.strides().at(dim)) / arr.itemsize()));
    }
  } else {
    for (size_t i = 0; i < arr.shape().at(dim); i++) {
      result.append(
          to_pylist_rec<T>(arr, offset + i * arr.strides().at(dim), dim + 1));
    }
  }
  return result;
}

py::object to_pylist(Array &arr) {
  if (!arr.evaluated()) {
    arr.eval();
  }

  if (arr.ndim() == 0) {
    switch (arr.dtype().type) {
    case Dtype::DataType::BoolType:
      return py::cast(arr.item<bool>());
    case Dtype::DataType::Int8Type:
      return py::cast(arr.item<int8_t>());
    case Dtype::DataType::Int16Type:
      return py::cast(arr.item<int16_t>());
    case Dtype::DataType::Int32Type:
      return py::cast(arr.item<int32_t>());
    case Dtype::DataType::Int64Type:
      return py::cast(arr.item<int64_t>());
    case Dtype::DataType::Float32Type:
      return py::cast(arr.item<float>());
    case Dtype::DataType::Float64Type:
      return py::cast(arr.item<double>());
    default:
      throw std::invalid_argument("Unknown data type");
    }
  }

  switch (arr.dtype().type) {
  case Dtype::DataType::BoolType:
    return to_pylist_rec<bool>(arr, 0, 0);
  case Dtype::DataType::Int8Type:
    return to_pylist_rec<int8_t>(arr, 0, 0);
  case Dtype::DataType::Int16Type:
    return to_pylist_rec<int16_t>(arr, 0, 0);
  case Dtype::DataType::Int32Type:
    return to_pylist_rec<int32_t>(arr, 0, 0);
  case Dtype::DataType::Int64Type:
    return to_pylist_rec<int64_t>(arr, 0, 0);
  case Dtype::DataType::Float32Type:
    return to_pylist_rec<float>(arr, 0, 0);
  case Dtype::DataType::Float64Type:
    return to_pylist_rec<double>(arr, 0, 0);
  default:
    // Handle unknown data type
    // Perhaps throw an exception or return an error
    throw std::invalid_argument("Unknown data type");
  }
}

auto parse_attr = [](const py::object &obj) -> int {
  if (py::isinstance<py::none>(obj)) {
    return 1;
  } else if (py::isinstance<py::int_>(obj)) {
    return obj.cast<int>();
  } else {
    throw std::invalid_argument("Invalid slice indices");
  }
};
} // anonymous namespace

void init_ailang_core(py::module &m) {
  py::class_<Dtype>(m, "dtype")
      .def(py::init<>())
      .def("__repr__", &Dtype::toString)
      .def("__hash__", &Dtype::hash)
      .def("__eq__", &Dtype::operator==);

  m.attr("bool") = Bool;
  m.attr("i8") = Int8;
  m.attr("i16") = Int16;
  m.attr("i32") = Int32;
  m.attr("i64") = Int64;
  m.attr("f32") = Float32;
  m.attr("f64") = Float64;

  py::class_<Device>(m, "device")
    .def(py::init<>())
    .def("__repr__", &Device::toString)
    .def("__hash__", &Device::hash)
    .def("__eq__", &Device::operator==);

  py::class_<Tracer, std::shared_ptr<Tracer>>(m, "tracer")
      .def("__repr__", &Tracer::toString)
      .def("__bool__",
           [](std::shared_ptr<Tracer> &tracer) {
             auto array = asTracer<Array>(tracer->aval());
             return array->item<bool>();
           })
      .def("__len__",
           [](Tracer &tracer) {
             throw std::runtime_error(
                 "Cannot get the length of an abstract Tracer.");
           })
      .def("__eq__",
           [](const std::shared_ptr<Tracer> &tracer,
              const std::shared_ptr<Tracer> &other) {
             return single<ComparePrimitive>({tracer, other},
                                            CompareOp::CompareType::EQ);
           })
      .def("__ne__",
           [](const std::shared_ptr<Tracer> &tracer,
              const std::shared_ptr<Tracer> &other) {
             return single<ComparePrimitive>({tracer, other},
                                            CompareOp::CompareType::NE);
           })
      .def("__lt__",
           [](const std::shared_ptr<Tracer> &tracer,
              const std::shared_ptr<Tracer> &other) {
             return single<ComparePrimitive>({tracer, other},
                                            CompareOp::CompareType::LT);
           })
      .def("__le__",
           [](const std::shared_ptr<Tracer> &tracer,
              const std::shared_ptr<Tracer> &other) {
             return single<ComparePrimitive>({tracer, other},
                                            CompareOp::CompareType::LE);
           })
      .def("__gt__",
           [](const std::shared_ptr<Tracer> &tracer,
              const std::shared_ptr<Tracer> &other) {
             return single<ComparePrimitive>({tracer, other},
                                            CompareOp::CompareType::GT);
           })
      .def("__ge__", [](const std::shared_ptr<Tracer> &tracer,
                        const std::shared_ptr<Tracer> &other) {
        return single<ComparePrimitive>({tracer, other},
                                       CompareOp::CompareType::GE);
      });

  py::class_<JVPTracer, Tracer, std::shared_ptr<JVPTracer>>(m, "jvptracer");
  py::class_<JITTracer, Tracer, std::shared_ptr<JITTracer>>(m, "jittracer");
  py::class_<Array, Tracer, std::shared_ptr<Array>>(m, "array",
                                                    py::buffer_protocol())
      .def(py::init<>([]() { return Array(0.0f); }))
      .def_buffer([](Array &a) -> py::buffer_info {
        return py::buffer_info(a.data<void>(), a.itemsize(),
                               py::format_descriptor<Dtype>::format(), a.ndim(),
                               a.shape(), a.strides());
      })
      .def("__repr__",
           [](Array &a) {
             if (!a.evaluated()) {
               a.eval();
             }
             std::ostringstream oss;
             oss << a;
             return oss.str();
           })
      .def(
          "__iter__",
          [](Array &a) {
            if (!a.evaluated()) {
              a.eval();
            }
            return py::make_iterator(a.begin(), a.end());
          },
          py::keep_alive<0, 1>())
      .def("__len__",
           [](Array &a) {
             assert(a.ndim() >= 1);
             return a.shape().at(0);
           })
      .def("__bool__", [](Array &a) { return a.item<bool>(); })
      .def("__getitem__",
           [](Array &a, const py::object &object) {
             if (!a.evaluated()) {
               a.eval();
             }
             std::vector<int> start;
             std::vector<int> end;
             std::vector<int> stride;
             if (py::isinstance<py::slice>(object)) {
               auto slice_ = object.cast<py::slice>();
               auto start_ = parse_attr(getattr(slice_, "start"));
               auto stop_ = parse_attr((getattr(slice_, "stop")));
               auto step_ = parse_attr(getattr(slice_, "step"));
               start.push_back(start_);
               end.push_back(stop_);
               stride.push_back(step_);
               for (size_t i = 1; i < a.ndim(); i++) {
                 start.push_back(0);
                 end.push_back(a.shape().at(i));
                 stride.push_back(1);
               }
               return slice(a, start, end, stride);
             } else if (py::isinstance<py::int_>(object)) {
               auto index = object.cast<int>();
               auto iter = a.begin();
               for (size_t i = 0; i < index; i++) {
                 iter++;
               }
               return *iter;
             } else if (py::isinstance<py::tuple>(object)) {
               auto sliceList = object.cast<py::list>();
               size_t i = 0;
               for (const auto &slice : sliceList) {
                 auto start_ = parse_attr(getattr(slice, "start"));
                 auto stop_ = parse_attr(getattr(slice, "stop"));
                 auto step_ = parse_attr(getattr(slice, "step"));
                 start.push_back(start_);
                 end.push_back(stop_);
                 stride.push_back(step_);
                 i++;
               }
               for (; i < a.ndim(); i++) {
                 start.push_back(0);
                 end.push_back(a.shape().at(i));
                 stride.push_back(1);
               }
               return slice(a, start, end, stride);
             } else {
               throw std::invalid_argument("Invalid indices");
             }
           })
      .def_property_readonly("shape",
                             [](Array &a) {
                               if (!a.evaluated()) {
                                 a.eval();
                               }
                               return vector_to_tuple(a.shape());
                             })
      .def_property_readonly("strides",
                             [](Array &a) {
                               if (!a.evaluated()) {
                                 a.eval();
                               }
                               return vector_to_tuple(a.strides());
                             })
      .def_property_readonly("data_size", &Array::size)
      .def_property_readonly("dtype", &Array::dtype)
      .def_property_readonly("ndim", &Array::ndim)
      .def_property_readonly("device", [](Array &a) { return a.device().toString(); })
      .def("item",
           [](Array &a) {
             switch (a.dtype().type) {
             case Dtype::DataType::BoolType:
               return py::cast(a.item<bool>());
             case Dtype::DataType::Int16Type:
               return py::cast(a.item<int16_t>());
             case Dtype::DataType::Int32Type:
               return py::cast(a.item<int32_t>());
             case Dtype::DataType::Int64Type:
               return py::cast(a.item<int64_t>());
             case Dtype::DataType::Float32Type:
               return py::cast(a.item<float>());
             case Dtype::DataType::Float64Type:
               return py::cast(a.item<double>());
             default:
               throw std::invalid_argument("Unknown data type");
             }
           })
      .def("tolist", [](Array &a) { return to_pylist(a); });

m.def("from_numpy", [](py::buffer arr, const std::string& device = "cpu") {
    arr.inc_ref();
    py::buffer_info buffer = arr.request();
    Dtype dtype = getDtypeFromFormat(buffer.format);
    auto shape = std::vector<int>(buffer.shape.begin(), buffer.shape.end());
    auto stride =
        std::vector<int>(buffer.strides.begin(), buffer.strides.end());
    Device dev;
    if (device == "cpu") {
      dev = cpu;
    } else if (device == "gpu") {
      dev = gpu;
    } else {
      throw std::invalid_argument("Invalid device type when creating array from numpy array.");
    }
    auto result = std::make_shared<Array>(allocator::Buffer(buffer.ptr), dtype,
                                          shape, stride, dev);
    return result;
  }, "construct ainl array from numpy array", py::arg("arr"), py::arg("device") = "cpu");

  m.def("jvp", [](py::function &f, std::vector<std::shared_ptr<Tracer>> primals,
                  std::vector<std::shared_ptr<Tracer>> tangents) {
    auto func = [&f](std::vector<std::shared_ptr<Tracer>> primals)
        -> std::shared_ptr<Tracer> {
      py::tuple posArgs = py::tuple(primals.size());
      for (size_t i = 0; i < primals.size(); i++) {
        posArgs[i] = primals[i];
      }
      auto result = f(*posArgs);
      return result.cast<std::shared_ptr<Tracer>>();
    };
    auto tracer = jvp(func, primals, tangents);
    if (auto jvptracer = std::dynamic_pointer_cast<JVPTracer>(tracer)) {
      return py::make_tuple(jvptracer->primal(), jvptracer->tangent());
    } else {
      throw std::runtime_error("Invalid return type");
    }
  });

  m.def(
      "trace_impl",
      [](py::function &f, std::vector<std::shared_ptr<Tracer>> inputs) {
        auto func = [&f](std::vector<std::shared_ptr<Tracer>> inputs)
            -> std::vector<std::shared_ptr<Tracer>> {
          py::tuple posArgs = py::tuple(inputs.size());
          for (size_t i = 0; i < inputs.size(); i++) {
            posArgs[i] = inputs[i];
          }
          auto result = f(*posArgs);
          if (py::isinstance<py::tuple>(result) ||
              py::isinstance<py::list>(result)) {
            std::vector<std::shared_ptr<Tracer>> resultVec;
            for (auto &item : result.cast<py::tuple>()) {
              resultVec.push_back(item.cast<std::shared_ptr<Tracer>>());
            }
            return resultVec;
          } else {
            auto containedResult = std::vector<std::shared_ptr<Tracer>>();
            containedResult.push_back(result.cast<std::shared_ptr<Tracer>>());
            return containedResult;
          }
        };

        auto module =
            jit(func, py::str(getattr(f, "__name__")), inputs);
        return module;
      },
      "jit compilation of python function with high level tracing", py::arg(), py::arg());
}