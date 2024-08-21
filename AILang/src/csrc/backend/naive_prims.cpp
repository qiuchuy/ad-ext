#include "array.h"
#include "backend/binary_op.h"
#include "backend/compute.h"
#include "backend/shape_infer_contract.h"
#include "backend/unary_op.h"
#include "dtype.h"
#include "ops.h"
#include "primitive.h"
#include <alloca.h>
namespace ainl::core {

void TransposePrimitive::evalCPU(const std::vector<Array> &inputs,
                                 Array &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[TransposePrimitive::evalCPU] expects exactly one input array.");
    }
    // talk about dataview transformation.
    // auto input = inputs[0];
    // auto shape = input.shape();
    // auto stride = input.strides();
    // std::reverse(shape.begin(), shape.end());
    // std::reverse(stride.begin(), stride.end());
    // auto size =
    //     std::accumulate(shape.begin(), shape.end(), 1,
    //     std::multiplies<int>()) * dtypeSize(input.dtype());
    // output.copyBySharing(input, size, 0, shape, stride);
    resolvePrimShapeInferContract("transpose", inputs, output);
    transpose_dispatch(inputs[0], output);
}
void AsTypePrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    assert(inputs.size() == 1);
    auto &in = inputs[0];
    output.setDataWithBuffer(allocator::malloc(output.size()), dtype_,
                             output.shape(), output.strides());
}

void AddPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 2) {
        throw std::invalid_argument(
            "[AddPrimitive::evalCPU] expects exactly two input arrays.");
    }
    resolvePrimShapeInferContract("add", inputs, output);
    binary(inputs[0], inputs[1], output, detail::Add());
}
void BroadcastPrimitive::evalCPU(const std::vector<Array> &inputs,
                                 Array &output) {
    if (inputs.size() != 1) {
        std::invalid_argument("[BroadCastPrimitive::evalCPU] expects "
                              "exactly one input array.");
    }
    if (output.size() == 0) {
        std::invalid_argument(
            "[BroadCastPrimitive::evalCPU] output size can't be zero. ");
    }
    // with attribute we dont use register factory
    resolvePrimShapeInferContract("broadcast", inputs, output);
}

void MatMulPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 2) {
        throw std::invalid_argument(
            "[MatMulPrimitive::evalCPU] expects exactly two input arrays.");
    }

    auto input1 = inputs[0];
    auto input2 = inputs[1];
    auto input1Shape = input1.shape();
    auto input2Shape = input2.shape();
    if (input1Shape.size() != 2 || input2Shape.size() != 2) {
        throw std::invalid_argument(
            "[MatMulPrimitive::evalCPU] input arrays must have exactly two "
            "dimensions.");
    }

    if (input1Shape[1] != input2Shape[0]) {
        throw std::invalid_argument(
            "[MatMulPrimitive::evalCPU] the second dimension of the first "
            "input "
            "array must be the same as the first dimension of the second input "
            "array.");
    }

    auto outputShape = {input1Shape[0], input2Shape[1]};
    auto size = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                std::multiplies<int>());
    // this is wrong, please update it
    output.copyBySharing(input1, size, 0, outputShape);
}

} // namespace ainl::core