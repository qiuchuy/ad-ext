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
}

}; // namespace ainl::ffi