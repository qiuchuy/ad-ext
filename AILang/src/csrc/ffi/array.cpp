#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include <sstream>

#include "array.h"
#include "ffi/array.h"
#include "ops.h"
#include "pass/stablehlo_lowering.h"
#include "primitive.h"
#include "trace.h"
#include "transformation.h"
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
      result.append(*(arr.data<T>() +
                      (offset + i * arr.strides().at(dim)) / arr.itemsize()));
    }
  } else {
    for (size_t i = 0; i < arr.shape().at(dim); i++) {
      result.append(
          toPyListRec<T>(arr, offset + i * arr.strides().at(dim), dim + 1));
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

auto parseAttr = [](const py::object &obj) -> int {
  if (py::isinstance<py::none>(obj)) {
    return 1;
  } else if (py::isinstance<py::int_>(obj)) {
    return obj.cast<int>();
  } else {
    throw std::invalid_argument("Invalid slice indices");
  }
};

void initArray(py::module &_m) {
  py::class_<ainl::core::Dtype>(_m, "dtype")
      .def(py::init<>())
      .def("__repr__", &ainl::core::Dtype::toString);
  py::class_<ainl::core::Tracer, std::shared_ptr<ainl::core::Tracer>>(_m,
                                                                      "tracer")
      .def("__repr__", &ainl::core::Tracer::toString)
      .def("__bool__",
           [](std::shared_ptr<ainl::core::Tracer> &tracer) {
             auto array =
                 ainl::core::asTracer<ainl::core::Array>(tracer->aval());
             return array->item<bool>();
           })
      .def("__len__",
           [](ainl::core::Tracer &tracer) {
             throw std::runtime_error(
                 "Cannot get the length of an abstract Tracer.");
           })
      .def("__eq__",
           [](const std::shared_ptr<ainl::core::Tracer> &tracer,
              const std::shared_ptr<ainl::core::Tracer> &other) {
             return ainl::core::unary<ainl::core::ComparePrimitive>(
                 {tracer, other}, ainl::ir::CompareOp::CompareType::EQ);
           })
      .def("__ne__",
           [](const std::shared_ptr<ainl::core::Tracer> &tracer,
              const std::shared_ptr<ainl::core::Tracer> &other) {
             return ainl::core::unary<ainl::core::ComparePrimitive>(
                 {tracer, other}, ainl::ir::CompareOp::CompareType::NE);
           })
      .def("__lt__",
           [](const std::shared_ptr<ainl::core::Tracer> &tracer,
              const std::shared_ptr<ainl::core::Tracer> &other) {
             return ainl::core::unary<ainl::core::ComparePrimitive>(
                 {tracer, other}, ainl::ir::CompareOp::CompareType::LT);
           })
      .def("__le__",
           [](const std::shared_ptr<ainl::core::Tracer> &tracer,
              const std::shared_ptr<ainl::core::Tracer> &other) {
             return ainl::core::unary<ainl::core::ComparePrimitive>(
                 {tracer, other}, ainl::ir::CompareOp::CompareType::LE);
           })
      .def("__gt__",
           [](const std::shared_ptr<ainl::core::Tracer> &tracer,
              const std::shared_ptr<ainl::core::Tracer> &other) {
             return ainl::core::unary<ainl::core::ComparePrimitive>(
                 {tracer, other}, ainl::ir::CompareOp::CompareType::GT);
           })
      .def("__ge__",
           [](const std::shared_ptr<ainl::core::Tracer> &tracer,
              const std::shared_ptr<ainl::core::Tracer> &other) {
             return ainl::core::unary<ainl::core::ComparePrimitive>(
                 {tracer, other}, ainl::ir::CompareOp::CompareType::GE);
           }) DEFINE_COMPARE_OPERATOR_ON_SCALAR(bool)
          DEFINE_COMPARE_OPERATOR_ON_SCALAR(int)
              DEFINE_COMPARE_OPERATOR_ON_SCALAR(int8_t)
                  DEFINE_COMPARE_OPERATOR_ON_SCALAR(int16_t)
                      DEFINE_COMPARE_OPERATOR_ON_SCALAR(int32_t)
                          DEFINE_COMPARE_OPERATOR_ON_SCALAR(int64_t)
                              DEFINE_COMPARE_OPERATOR_ON_SCALAR(float)
                                  DEFINE_COMPARE_OPERATOR_ON_SCALAR(double);

  py::class_<ainl::core::JVPTracer, ainl::core::Tracer,
             std::shared_ptr<ainl::core::JVPTracer>>(_m, "jvptracer");
  py::class_<ainl::core::JITTracer, ainl::core::Tracer,
             std::shared_ptr<ainl::core::JITTracer>>(_m, "jittracer");
  py::class_<ainl::core::Array, ainl::core::Tracer,
             std::shared_ptr<ainl::core::Array>>(_m, "array",
                                                 py::buffer_protocol())
      .def(py::init<>([]() { return ainl::core::Array(1.0f); }))
      .def_buffer([](ainl::core::Array &a) -> py::buffer_info {
        return py::buffer_info(
            a.data<void>(), a.itemsize(),
            py::format_descriptor<ainl::core::Dtype>::format(), a.ndim(),
            a.shape(), a.strides());
      })
      .def("__repr__",
           [](ainl::core::Array &a) {
             if (!a.evaluated()) {
               a.eval();
             }
             std::ostringstream oss;
             oss << a;
             return oss.str();
           })
      .def(
          "__iter__",
          [](ainl::core::Array &a) {
            if (!a.evaluated()) {
              a.eval();
            }
            return py::make_iterator(a.begin(), a.end());
          },
          py::keep_alive<0, 1>())
      .def("__len__",
           [](ainl::core::Array &a) {
             assert(a.ndim() >= 1);
             return a.shape().at(0);
           })
      .def("__bool__", [](ainl::core::Array &a) { return a.item<bool>(); })
      .def("__getitem__",
           [](ainl::core::Array &a, const py::object &object) {
             if (!a.evaluated()) {
               a.eval();
             }
             std::vector<int> start;
             std::vector<int> end;
             std::vector<int> stride;
             if (py::isinstance<py::slice>(object)) {
               auto slice_ = object.cast<py::slice>();
               auto start_ = parseAttr(getattr(slice_, "start"));
               auto stop_ = parseAttr((getattr(slice_, "stop")));
               auto step_ = parseAttr(getattr(slice_, "step"));
               start.push_back(start_);
               end.push_back(stop_);
               stride.push_back(step_);
               for (size_t i = 1; i < a.ndim(); i++) {
                 start.push_back(0);
                 end.push_back(a.shape().at(i));
                 stride.push_back(1);
               }
               return ainl::core::slice(a, start, end, stride);
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
                 auto start_ = parseAttr(getattr(slice, "start"));
                 auto stop_ = parseAttr(getattr(slice, "stop"));
                 auto step_ = parseAttr(getattr(slice, "step"));
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
               return ainl::core::slice(a, start, end, stride);
             } else {
               throw std::invalid_argument("Invalid indices");
             }
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
      .def("item",
           [](ainl::core::Array &a) {
             switch (a.dtype().type) {
             case ainl::core::Dtype::DataType::BoolType:
               return py::cast(a.item<bool>());
             case ainl::core::Dtype::DataType::Int16Type:
               return py::cast(a.item<int16_t>());
             case ainl::core::Dtype::DataType::Int32Type:
               return py::cast(a.item<int32_t>());
             case ainl::core::Dtype::DataType::Int64Type:
               return py::cast(a.item<int64_t>());
             case ainl::core::Dtype::DataType::Float32Type:
               return py::cast(a.item<float>());
             case ainl::core::Dtype::DataType::Float64Type:
               return py::cast(a.item<double>());
             default:
               throw std::invalid_argument("Unknown data type");
             }
           })
      .def("tolist", [](ainl::core::Array &a) { return toPyList(a); });

  _m.def("from_numpy", [](py::buffer arr) {
    arr.inc_ref();
    py::buffer_info buffer = arr.request();
    ainl::core::Dtype dtype = ainl::core::getDtypeFromFormat(buffer.format);
    auto shape = std::vector<int>(buffer.shape.begin(), buffer.shape.end());
    auto stride =
        std::vector<int>(buffer.strides.begin(), buffer.strides.end());
    auto result = std::make_shared<ainl::core::Array>(
        ainl::core::allocator::Buffer(buffer.ptr), dtype, shape, stride);
    return result;
  });

  _m.def("jvp", [](py::function &f,
                   std::vector<std::shared_ptr<ainl::core::Tracer>> primals,
                   std::vector<std::shared_ptr<ainl::core::Tracer>> tangents) {
    auto func = [&f](std::vector<std::shared_ptr<ainl::core::Tracer>> primals)
        -> std::shared_ptr<ainl::core::Tracer> {
      py::tuple posArgs = py::tuple(primals.size());
      for (size_t i = 0; i < primals.size(); i++) {
        posArgs[i] = primals[i];
      }
      auto result = f(*posArgs);
      return result.cast<std::shared_ptr<ainl::core::Tracer>>();
    };
    auto tracer = jvp(func, primals, tangents);
    if (auto jvptracer =
            std::dynamic_pointer_cast<ainl::core::JVPTracer>(tracer)) {
      return py::make_tuple(jvptracer->primal(), jvptracer->tangent());
    } else {
      throw std::runtime_error("Invalid return type");
    }
  });

  _m.def(
      "jit_impl",
      [](py::function &f,
         std::vector<std::shared_ptr<ainl::core::Tracer>> inputs,
         const std::string &target) {
        auto func =
            [&f](std::vector<std::shared_ptr<ainl::core::Tracer>> inputs)
            -> std::vector<std::shared_ptr<ainl::core::Tracer>> {
          py::tuple posArgs = py::tuple(inputs.size());
          for (size_t i = 0; i < inputs.size(); i++) {
            posArgs[i] = inputs[i];
          }
          auto result = f(*posArgs);
          if (py::isinstance<py::tuple>(result) ||
              py::isinstance<py::list>(result)) {
            std::vector<std::shared_ptr<ainl::core::Tracer>> resultVec;
            for (auto &item : result.cast<py::tuple>()) {
              resultVec.push_back(
                  item.cast<std::shared_ptr<ainl::core::Tracer>>());
            }
            return resultVec;
          } else {
            auto containedResult =
                std::vector<std::shared_ptr<ainl::core::Tracer>>();
            containedResult.push_back(
                result.cast<std::shared_ptr<ainl::core::Tracer>>());
            return containedResult;
          }
        };

        auto module =
            jit(func, py::str(getattr(f, "__name__")), target, inputs);
        if (target == "ailang") {
          return py::cast(module);
        } else if (target == "mlir") {
          return py::cast(ir::StableHLOLowering(module));
        } else {
          throw std::invalid_argument("Invalid jit target");
        }
      },
      "jit compilation of python function", py::arg(), py::arg(),
      py::arg("target") = "ailang");
}

} // namespace ainl::ffi