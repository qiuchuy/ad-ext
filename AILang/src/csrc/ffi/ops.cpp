#include "ffi/ops.h"
#include "array.h"
#include "ops.h"
#include "primitive.h"
#include "transformation.h"

namespace ainl::ffi {

void initOps(py::module_ &m) {
    m.def(
        "flatten",
        [](const ainl::core::Array &input) {
            return ainl::core::flatten(input);
        },
        "Flatten the input");

    m.def(
        "flatten",
        [](const ainl::core::JVPTracer &input) {
            return op<ainl::core::JVPTracer, ainl::core::FlattenPrimitive>(
                {input});
        },
        "Flatten the input");

    m.def(
        "reshape",
        [](const ainl::core::Array &input, const std::vector<int> &shape) {
            return ainl::core::reshape(input, shape);
        },
        "Reshape the input");

    m.def(
        "reshape",
        [](const ainl::core::JVPTracer &input, const std::vector<int> &shape) {
            return op<ainl::core::JVPTracer, ainl::core::ReshapePrimitive>(
                {input}, shape);
        },
        "Reshape the input");

    m.def(
        "slice",
        [](const ainl::core::Array &input, const std::vector<int> &start,
           const std::vector<int> &end, const std::vector<int> &stride) {
            return ainl::core::slice(input, start, end, stride);
        },
        "Slice the input");

    m.def(
        "slice",
        [](const ainl::core::JVPTracer &input, const std::vector<int> &start,
           const std::vector<int> &end, const std::vector<int> &stride) {
            return op<ainl::core::JVPTracer, ainl::core::SlicePrimitive>(
                {input}, start, end, stride);
        },
        "Slice the input");

    m.def(
        "sigmoid",
        [](const ainl::core::Array &input) {
            return ainl::core::sigmoid(input);
        },
        "Get sigmoid of input");

    m.def(
        "cos",
        [](const ainl::core::Array &input) { return ainl::core::cos(input); },
        "Get sigmoid of input");
    m.def(
        "Add",
        [](const ainl::core::Array &a, const ainl::core::Array &b) {
            return ainl::core::add(a, b);
        },
        "Add the inputs");
    m.def(
        "ones",
        [](const std::vector<int> &shape) {
            return ainl::core::ones(shape, ainl::core::Float64);
        },
        "create zeros array");
    m.def(
        "zeros",
        [](const std::vector<int> &shape) {
            return ainl::core::zeros(shape, ainl::core::Float64);
        },
        "create zeros array");
    m.def(
        "mean",
        [](const ainl::core::Array &input, std::vector<int> &axes,
           bool keepdims = false) {
            return ainl::core::mean(input, axes, keepdims);
        },
        "compute  array's mean");
    m.def(
        "mean",
        [](const ainl::core::Array &input, int axis, bool keepdims = false) {
            return ainl::core::mean(input, axis, keepdims);
        },
        "compute  array's mean");
    m.def(
        "mean",
        [](const ainl::core::Array &input, bool keepdims = false) {
            return ainl::core::mean(input, keepdims);
        },
        "compute  array's mean");
    m.def(
        "var",
        [](const ainl::core::Array &input, bool keepdims = false) {
            return ainl::core::mean(input, keepdims);
        },
        "compute  array's mean");
    m.def(
        "conv",
        [](const ainl::core::Array &input, const ainl::core::Array &weight,
           const std::pair<int, int> &stride,
           const std::pair<int, int> &padding,
           const std::pair<int, int> &dilation) {
            return ainl::core::conv2d(input, weight, stride, padding, dilation);
        },
        "compute  array's conv");
}

}; // namespace ainl::ffi