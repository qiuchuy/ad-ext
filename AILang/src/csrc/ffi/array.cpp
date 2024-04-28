#include <sstream>

#include "array.h"
#include "ffi/array.h"
#include "ops.h"
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
            result.append(*(arr.data<T>() + offset / arr.itemsize() + i));
        }
    } else {
        for (size_t i = 0; i < arr.shape().at(dim); i++) {
            int dimOffset = 1;
            for (size_t j = dim + 1; j < arr.shape().size(); j++) {
                dimOffset *= arr.shape()[j];
            }
            // auto dimOffset =
            //     std::accumulate(arr.shape().begin() + dim, arr.shape().end(),
            //     1,
            //                     std::multiplies<int>());
            // [TODO] why this causes a segfault?
            result.append(toPyListRec<T>(
                arr, offset + i * dimOffset * arr.itemsize(), dim + 1));
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
    py::class_<ainl::core::Tracer, std::shared_ptr<ainl::core::Tracer>>(
        _m, "tracer")
        .def(py::init<>())
        .def("__repr__", &ainl::core::Tracer::toString);
    py::class_<ainl::core::JVPTracer, ainl::core::Tracer,
               std::shared_ptr<ainl::core::JVPTracer>>(_m, "jvptracer");
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
                 LOG_DEBUG("%s", "here");
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

    _m.def(
        "jvp", [](py::function &f,
                  std::vector<std::shared_ptr<ainl::core::Tracer>> primals,
                  std::vector<std::shared_ptr<ainl::core::Tracer>> tangents) {
            auto func =
                [&f](std::vector<std::shared_ptr<ainl::core::Tracer>> primals)
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
                return py::make_tuple(jvptracer->primal(),
                                      jvptracer->tangent());
            } else {
                throw std::runtime_error("Invalid return type");
            }
        });
}

} // namespace ainl::ffi