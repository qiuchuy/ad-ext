#include "primitive.h"
#include "backend/common/Broadcast.h"
#include "backend/common/conv.h"
#include "backend/common/reduce.h"

#include "binary.h"
#include "ops.h"
#include "transformation.h"
#include "unary.h"

#include <algorithm>
#include <memory>
#include <numeric>

#include "ast/node_contract.h"
#include "ast/type_contract.h"
#include "ir/type.h"
#include "ops.h"
#include "trace.h"
#include "transformation.h"

namespace ainl::core {

void IdentityPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void IdentityPrimitive::evalCPU(const std::vector<Array> &inputs,
                                Array &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("[IdentityPrimitive::evalCPU] expects "
                                    "exactly one input array.");
    }
    output.copyBySharing(inputs[0], 0, 0, inputs[0].shape());
}

void IdentityPrimitive::jit(const std::vector<JITTracer> &inputs,
                            JITTracer &output) {}

void IdentityPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                            JVPTracer &output) {}

std::string IdentityPrimitive::toString() const { return "Identity"; }

void AddPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void AddPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 2) {
        throw std::invalid_argument(
            "[AddPrimitive::evalCPU] expects exactly two input arrays.");
    }

    auto input1 = inputs[0];
    auto input2 = inputs[1];
    auto input1Shape = input1.shape();
    auto input2Shape = input2.shape();
    if (input1Shape != input2Shape) {
        throw std::invalid_argument("[AddPrimitive::evalCPU] input arrays "
                                    "must have the same shape, "
                                    "broadcasting is not supported yet.");
    }

    auto size = std::accumulate(input1Shape.begin(), input1Shape.end(), 1,
                                std::multiplies<int>());

    binary(input1, input2, output, detail::Add());
}

void AddPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {}

void AddPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}

std::string AddPrimitive::toString() const { return "Add"; }
// substract

void SubtractPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void SubtractPrimitive::jit(const std::vector<JITTracer> &inputs,
                            JITTracer &output) {}
void SubtractPrimitive::evalCPU(const std::vector<Array> &inputs,
                                Array &output) {
    if (inputs.size() != 2) {
        throw std::invalid_argument(
            "[SubtractPrimitive::evalCPU] expects exactly two input arrays.");
    }

    auto input1 = inputs[0];
    auto input2 = inputs[1];
    auto input1Shape = input1.shape();
    auto input2Shape = input2.shape();
    if (input1Shape != input2Shape) {
        throw std::invalid_argument("[SubtractPrimitive::evalCPU] input arrays "
                                    "must have the same shape, "
                                    "broadcasting is not supported yet.");
    }

    auto size = std::accumulate(input1Shape.begin(), input1Shape.end(), 1,
                                std::multiplies<int>());

    binary(input1, input2, output, detail::Sub());
}

void SubtractPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                            JVPTracer &output) {}

std::string SubtractPrimitive::toString() const { return "Sub"; }
// SquarePrimitive

void SquarePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void SquarePrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[SquarePrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::Square());
    } else {
        std::invalid_argument("[SquarePrimitive:evalCPU] Dtype must be "
                              "Float in SquarePrimitive.");
    }
}
void SquarePrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void SquarePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string SquarePrimitive::toString() const { return "Square"; }

// SqrtPrimitive

void SqrtPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void SqrtPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[SqrtPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        if (reverse_)
            unary(input, output, detail::Sqrt());
        else
            unary(input, output, detail::Rsqrt());

    } else {
        std::invalid_argument("[SqrtPrimitive:evalCPU] Dtype must be "
                              "Float in SqrtPrimitive.");
    }
}
void SqrtPrimitive::jit(const std::vector<JITTracer> &inputs,
                        JITTracer &output) {}
void SqrtPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}

std::string SqrtPrimitive::toString() const { return "Sqrt"; }

// max

void MaximumPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void MaximumPrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
    if (inputs.size() != 2) {
        throw std::invalid_argument(
            "[AddPrimitive::evalCPU] expects exactly two input arrays.");
    }

    auto input1 = inputs[0];
    auto input2 = inputs[1];
    binary(input1, input2, output, detail::Maximum());
}
void MaximumPrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}
void MaximumPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string MaximumPrimitive::toString() const { return "Max"; }
// min

void MinimumPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void MinimumPrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
    if (inputs.size() != 2) {
        throw std::invalid_argument(
            "[MinimumPrimitive::evalCPU] expects exactly two input arrays.");
    }

    auto input1 = inputs[0];
    auto input2 = inputs[1];
    binary(input1, input2, output, detail::Minimum());
}
void MinimumPrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}
void MinimumPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string MinimumPrimitive::toString() const { return "Min"; }

// Flatten
void FlattenPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void FlattenPrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {

    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[FlattenPrimitive::evalCPU] expects exactly one input array.");
    }

    auto input = inputs[0];
    auto inputShape = input.shape();
    auto size = std::accumulate(inputShape.begin(), inputShape.end(), 1,
                                std::multiplies<int>());
    output.copyBySharing(input, size, 0, {size});
}

void FlattenPrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}

void FlattenPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string FlattenPrimitive::toString() const { return "Flatten"; }

void FillPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void FillPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 1) {
        if (inputs.size() != 1) {
            std::invalid_argument("[FillPrimitive::evalCPU] expects "
                                  "exactly one input array.");
        }
        const auto &input = inputs[0];
        if (input.dtype() != output.dtype())
            std::runtime_error("[FillPrimitive] input and output dont have "
                               "the same dtype.");
        auto inputShape = input.shape();
        auto size = std::accumulate(inputShape.begin(), inputShape.end(), 1,
                                    std::multiplies<int>());
        output.copyBySharing(input, size, 0, inputShape);
    }
}

void FillPrimitive::jit(const std::vector<JITTracer> &inputs,
                        JITTracer &output) {}

void FillPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}

std::string FillPrimitive::toString() const { return "Fill"; }

void SlicePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void SlicePrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    // Numpy style slice, see:
    // https://numpy.org/doc/stable/user/basics.indexing.html
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[SlicePrimitive::evalCPU] expects exactly one input array.");
    }

    if (begin_.size() != end_.size() || begin_.size() != stride_.size()) {
        throw std::invalid_argument("[SlicePrimitive::evalCPU] begin, end and "
                                    "stride should have the same "
                                    "size.");
    }

    const auto input = inputs[0];
    size_t inputNdim = input.ndim();
    if (begin_.size() != inputNdim || end_.size() != inputNdim ||
        stride_.size() != inputNdim) {
        throw std::invalid_argument("[SlicePrimitive::evalCPU] begin, end and "
                                    "stride should have the same "
                                    "size as the input array.");
    }

    if (inputNdim == 0) {
        throw std::invalid_argument("[SlicePrimitive::evalCPU] Input array "
                                    "must have at least one dimension.");
    }
    auto inputShape = input.shape();

    // check input ranges: suppose input has shape (a, b, c)
    // then illegal slice ranges should be: [-a, a], [-b, b], [-c, c]
    for (size_t i = 0; i < inputNdim; i++) {
        if (begin_[i] < -inputShape[i] || begin_[i] > inputShape[i]) {
            throw std::invalid_argument(
                "[SlicePrimitive::evalCPU] Illegal slice "
                "range for input array.");
        }
    }

    for (size_t i = 0; i < inputNdim; i++) {
        if (end_[i] < -inputShape[i] || end_[i] > inputShape[i]) {
            throw std::invalid_argument(
                "[SlicePrimitive::evalCPU] Illegal slice "
                "range for input array.");
        }
    }

    // convert negative slice range into positive
    auto begin = begin_;
    for (size_t i = 0; i < begin.size(); i++) {
        if (begin[i] < 0) {
            begin[i] = inputShape[i] + begin[i];
        }
    }
    auto end = end_;
    for (size_t i = 0; i < end.size(); i++) {
        if (end[i] < 0) {
            end[i] = inputShape[i] + end[i];
        }
    }

    // calculate the offset, size of the output array
    size_t size = output.itemsize();
    for (size_t i = 0; i < output.shape().size(); i++) {
        size *= output.shape()[i];
    }
    auto offset = 0;
    for (size_t i = 0; i < inputShape.size(); i++) {
        auto dimOffset = 0;
        for (size_t j = i + 1; j < inputShape.size(); j++) {
            dimOffset += inputShape[j] * input.itemsize();
        }
        offset += begin[i] * dimOffset;
    }
    output.copyBySharing(input, size, offset, output.shape());
}

std::string SlicePrimitive::toString() const { return "Slice"; }

void SlicePrimitive::jit(const std::vector<JITTracer> &inputs,
                         JITTracer &output) {}

void SlicePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                         JVPTracer &output) {}

void ReshapePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void ReshapePrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[ReshapePrimitive::evalCPU] expects exactly one input array.");
    }

    auto input = inputs[0];
    auto inputShape = input.shape();
    // * itemsize
    auto data_size = std::accumulate(inputShape.begin(), inputShape.end(), 1,
                                     std::multiplies<int>());
    size_t size = data_size * input.itemsize();
    if (std::accumulate(shape_.begin(), shape_.end(), 1,
                        std::multiplies<int>()) != data_size) {
        throw std::invalid_argument("[ReshapePrimitive::evalCPU] The total "
                                    "number of elements in the "
                                    "input array must be the same as the "
                                    "total number of elements in "
                                    "the "
                                    "output array.");
    }

    output.copyBySharing(input, size, 0, shape_);
}

void ReshapePrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}

void ReshapePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[ReshapePrimitive::jvp] expects exactly one input tracer.");
    }
    auto input = inputs[0];

    output.setPrimal(
        reshape({input.primal()}, std::make_shared<ReshapePrimitive>(shape_)));
    output.setTangent(
        reshape({input.tangent()}, std::make_shared<ReshapePrimitive>(shape_)));
}

std::string ReshapePrimitive::toString() const { return "Reshape"; }
// transpose
void TransposePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void TransposePrimitive::evalCPU(const std::vector<Array> &inputs,
                                 Array &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[TransposePrimitive::evalCPU] expects exactly one input array.");
    }

    auto input = inputs[0];
    auto shape = input.shape();
    auto stride = input.strides();
    std::reverse(shape.begin(), shape.end());
    std::reverse(stride.begin(), stride.end());
    auto size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    output.copyBySharing(input, size, 0, shape, stride);
}

void TransposePrimitive::jit(const std::vector<JITTracer> &inputs,
                             JITTracer &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[TransposePrimitive::jit] expects exactly one input tracer.");
    }

    auto input = inputs[0];
    std::vector<ir::TypePtr> inputType = {input.value()->getType()};
    std::vector<ir::ValuePtr> inputValues = {input.value()};
    auto outputType = ir::resolveContract("transpose", inputType);

    auto module = getTracedModule();
    output.setValue(
        ir::resolveContract("transpose", module, outputType, inputValues));
    output.setTracer(
        transpose({input.tracer()}, std::make_shared<TransposePrimitive>()));
}

void TransposePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                             JVPTracer &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[TransposePrimitive::jvp] expects exactly one input tracer.");
    }
    auto input = inputs[0];

    output.setPrimal(
        transpose({input.primal()}, std::make_shared<TransposePrimitive>()));
    output.setTangent(
        transpose({input.tangent()}, std::make_shared<TransposePrimitive>()));
}

std::string TransposePrimitive::toString() const { return "Transpose"; }

// Abs
void AbsPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void AbsPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[AbsPrimitive::evalCPU] expects exactly one input array.");
    }
    auto input = inputs[0];
    // only deal with the first element
    // if sbool or unsigned copy from buffer else dispatcher
    // TODO unsigned type skip compute
    unary(input, output, detail::Abs());
}
void AbsPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {}
void AbsPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}

std::string AbsPrimitive::toString() const { return "Abs"; }

// TODO ADDMM
void AddMMPrimitive::eval(const std::vector<Array> &inputs, Array &output) {}
void AddMMPrimitive::evalCPU(const std::vector<Array> &inputs, Array &outputs) {
}
void AddMMPrimitive::jit(const std::vector<JITTracer> &inputs,
                         JITTracer &output) {}
void AddMMPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                         JVPTracer &output) {}
std::string AddMMPrimitive::toString() const { return "AddMM"; }

// Arange
void ArangePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
template <typename T> void arange(T begin, T next, Array &out, size_t size) {
    auto ptr = out.data<T>();
    auto step_size = next - begin;
    for (int i = 0; i < size; ++i) {
        ptr[i] = begin;
        begin += step_size;
    }
}
void ArangePrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() == 0) {
        std::invalid_argument(
            "[ArangePrimitive::evalCPU] expects exactly zero input array.");
    }
    output.SetDataWithBuffer(allocator::malloc(output.size()), output.dtype(),
                             output.shape(),
                             output.strides()); // malloc bytes
    if (output.dtype() == Bool) {
        std::runtime_error(
            "[ArangePrimitive:evalCPU]: Bool type unsupported for arange.");
    } else if (output.dtype() == Int8) {
        arange<int8_t>(start_, start_ + stride_, output, output.size());
    } else if (output.dtype() == Int16) {
        arange<int16_t>(start_, start_ + stride_, output, output.size());
    }

    else if (output.dtype() == Int32) {
        arange<int32_t>(start_, start_ + stride_, output, output.size());
    } else if (output.dtype() == Int64) {
        arange<int64_t>(start_, start_ + stride_, output, output.size());
    } else if (output.dtype() == Float32) {
        arange<float>(start_, start_ + stride_, output, output.size());
    } else if (output.dtype() == Float64) {
        arange<double>(start_, start_ + stride_, output, output.size());
    }
}
void ArangePrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void ArangePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string ArangePrimitive::toString() const { return "Arange"; }
// AsType
void AsTypePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void AsTypePrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    assert(inputs.size() == 1);
    auto &in = inputs[0];
    output.SetDataWithBuffer(allocator::malloc(output.size()), dtype_,
                             output.shape(), output.strides());
}
void AsTypePrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void AsTypePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string AsTypePrimitive::toString() const { return "AsType"; }

// Trigonometric functions
void ArcCosPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void ArcCosPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[ArcCosPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::ArcCos());
    } else {
        std::invalid_argument("[ArcCosPrimitive:evalCPU] Dtype must be "
                              "Float in inverse cosine.");
    }
}
void ArcCosPrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void ArcCosPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string ArcCosPrimitive::toString() const { return "ArcCos"; }

// ArcTan functions
void ArcTanPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void ArcTanPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[ArcTanPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::ArcTan());
    } else {
        std::invalid_argument("[ArcTanPrimitive:evalCPU] Dtype must be "
                              "Float in inverse Tan.");
    }
}
void ArcTanPrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void ArcTanPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string ArcTanPrimitive::toString() const { return "ArcTan"; }

// ArcSin functions
void ArcSinPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void ArcSinPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[ArcSinPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::ArcSin());
    } else {
        std::invalid_argument("[ArcSinPrimitive:evalCPU] Dtype must be "
                              "Float in inverse sine.");
    }
}
void ArcSinPrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void ArcSinPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string ArcSinPrimitive::toString() const { return "ArcSin"; }

// Trigonometric h functions
void ArcCoshPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void ArcCoshPrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[ArcCoshPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::ArcCosh());
    } else {
        std::invalid_argument("[ArcCoshPrimitive::evalCPU] Dtype must be Float "
                              "in inverse cosineh.");
    }
}
void ArcCoshPrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}
void ArcCoshPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string ArcCoshPrimitive::toString() const { return "ArcCosh"; }

// ArcTanh functions
void ArcTanhPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void ArcTanhPrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[ArcTanhPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::ArcTanh());
    } else {
        std::invalid_argument("[ArcTanhPrimitive:evalCPU] Dtype must be "
                              "Float in inverse Tanh.");
    }
}
void ArcTanhPrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}
void ArcTanhPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string ArcTanhPrimitive::toString() const { return "ArcTanh"; }

// ArcSinh functions
void ArcSinhPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void ArcSinhPrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[ArcSinhPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::ArcSinh());
    } else {
        std::invalid_argument("[ArcSinhPrimitive:evalCPU] Dtype must be "
                              "Float in inverse sine.");
    }
}
void ArcSinhPrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}
void ArcSinhPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string ArcSinhPrimitive::toString() const { return "ArcSinh"; }

// not reverse
void CosPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void CosPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[CosPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::Cos());
    } else {
        std::invalid_argument(
            "[CosPrimitive:evalCPU] Dtype must be Float in  cosine.");
    }
}
void CosPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {}
void CosPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}

std::string CosPrimitive::toString() const { return "Cos"; }

// Tan functions
void TanPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void TanPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[TanPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::Tan());
    } else {
        std::invalid_argument(
            "[TanPrimitive:evalCPU] Dtype must be Float in  Tan.");
    }
}
void TanPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {}
void TanPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}

std::string TanPrimitive::toString() const { return "Tan"; }

// Sin functions
void SinPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void SinPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[SinPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::Sin());
    } else {
        std::invalid_argument(
            "[SinPrimitive:evalCPU] Dtype must be Float in  sine.");
    }
}
void SinPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {}
void SinPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}

std::string SinPrimitive::toString() const { return "Sin"; }

// Trigonometric h functions
void CoshPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void CoshPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[CoshPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::Cosh());
    } else {
        std::invalid_argument("[CoshPrimitive::evalCPU] Dtype must be Float "
                              "in  cosineh.");
    }
}
void CoshPrimitive::jit(const std::vector<JITTracer> &inputs,
                        JITTracer &output) {}
void CoshPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}

std::string CoshPrimitive::toString() const { return "Cosh"; }

// Tanh functions
void TanhPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void TanhPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[TanhPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::Tanh());
    } else {
        std::invalid_argument(
            "[TanhPrimitive:evalCPU] Dtype must be Float in  Tanh.");
    }
}
void TanhPrimitive::jit(const std::vector<JITTracer> &inputs,
                        JITTracer &output) {}
void TanhPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}

std::string TanhPrimitive::toString() const { return "Tanh"; }

// Sinh functions
void SinhPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}
void SinhPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[SinhPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::Sinh());
    } else {
        std::invalid_argument(
            "[SinhPrimitive:evalCPU] Dtype must be Float in  sineh.");
    }
}
void SinhPrimitive::jit(const std::vector<JITTracer> &inputs,
                        JITTracer &output) {}
void SinhPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}

std::string SinhPrimitive::toString() const { return "Sinh"; }

// FIXME BroadCast in Value has error, but with no broadcast and just
// shape/operate is successful, maybe will be called frequently
void BroadCastPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
    evalCPU(inputs, out);
}

void BroadCastPrimitive::evalCPU(const std::vector<Array> &inputs,
                                 Array &output) {
    if (inputs.size() != 1) {
        std::invalid_argument("[BroadCastPrimitive::evalCPU] expects "
                              "exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.size() == 0) {
        std::invalid_argument(
            "[BroadCastPrimitive::evalCPU] output size can't be zero. ");
    }

    output.SetDataWithBuffer(allocator::malloc(output.size()), output.dtype(),
                             shape_, output.strides());
    BroadCast_dispatch(input, output);
}
void BroadCastPrimitive::jit(const std::vector<JITTracer> &inputs,
                             JITTracer &output) {}
void BroadCastPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                             JVPTracer &output) {}

std::string BroadCastPrimitive::toString() const { return "BroadCast"; }
// exp
void ExpPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
    evalCPU(inputs, out);
}
void ExpPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[ExpPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::Exp());
    } else {
        std::invalid_argument(
            "[ExpPrimitive:evalCPU] Dtype must be Float in  sineh.");
    }
}
void ExpPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {}
void ExpPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}

std::string ExpPrimitive::toString() const { return "Exp"; }
// log
void LogPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
    evalCPU(inputs, out);
}
void LogPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[LogPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        switch (base_) {
        case LogBase::e:
            unary(input, output, detail::Log());
            break;
        case LogBase::two:
            unary(input, output, detail::Log2());
            break;
        case LogBase::ten:
            unary(input, output, detail::Log10());
            break;
        default:
            break;
        }
    } else {
        std::invalid_argument(
            "[LogPrimitive:evalCPU] Dtype must be Float in log.");
    }
}
void LogPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {}
void LogPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}

std::string LogPrimitive::toString() const { return "Log"; }

// sigmoid
void SigmoidPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
    evalCPU(inputs, out);
}
void SigmoidPrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument(
            "[SigmoidPrimitive::evalCPU] expects exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.dtype() == Float32 || output.dtype() == Float64) {
        unary(input, output, detail::Sigmoid());
    } else {
        std::invalid_argument(
            "[SigmoidPrimitive:evalCPU] Dtype must be Float in  Sigmoid.");
    }
}
void SigmoidPrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}
void SigmoidPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string SigmoidPrimitive::toString() const { return "Sigmoid"; }
// Softmax
void SoftmaxPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
    evalCPU(inputs, out);
}
void SoftmaxPrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument("[SoftmaxPrimitive::evalCPU] not implement.");
    }
}
void SoftmaxPrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}
void SoftmaxPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string SoftmaxPrimitive::toString() const { return "Softmax"; }

// GetElementsNumberPrimitive
void GetElementsNumberPrimitive::eval(const std::vector<Array> &inputs,
                                      Array &out) {
    evalCPU(inputs, out);
}
void GetElementsNumberPrimitive::evalCPU(const std::vector<Array> &inputs,
                                         Array &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument("[GetElementsNumberPrimitive eval] "
                                    "inputs must have one arrays.");
    }
    output.SetDataWithBuffer(allocator::malloc(output.size()), dtype_,
                             output.shape(), output.strides());
    auto input = inputs[0];
    double numbel = 1;
    for (auto axis : axes_) {
        numbel *= input.shape()[axis];
    }
    if (inverted_) {
        numbel = 1.0 / numbel;
    }
    switch (output.dtype().type) {
    case Dtype::DataType::Any:
        std::invalid_argument(
            "[GetElementsNumberPrimitive evalCPU] not support Ant type.");
        break;
    case Dtype::DataType::BoolType:
        *output.data<bool>() = static_cast<bool>(numbel);
        break;
    case Dtype::DataType::Int8Type:
        *output.data<int8_t>() = static_cast<int8_t>(numbel);
        break;
    case Dtype::DataType::Int16Type:
        *output.data<int16_t>() = static_cast<int16_t>(numbel);
        break;
    case Dtype::DataType::Int32Type:
        *output.data<int32_t>() = static_cast<int32_t>(numbel);
        break;
    case Dtype::DataType::Int64Type:
        *output.data<int64_t>() = static_cast<int64_t>(numbel);
        break;
    case Dtype::DataType::Float32Type:
        *output.data<float>() = static_cast<float>(numbel);
        break;
    case Dtype::DataType::Float64Type:
        *output.data<double>() = static_cast<double>(numbel);
        break;
    default:
        break;
    }
}
void GetElementsNumberPrimitive::jit(const std::vector<JITTracer> &inputs,
                                     JITTracer &output) {}
void GetElementsNumberPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                                     JVPTracer &output) {}

std::string GetElementsNumberPrimitive::toString() const {
    return "GetElementsNumber";
}

// Multiply
void MultiplyPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
    evalCPU(inputs, out);
}
void MultiplyPrimitive::evalCPU(const std::vector<Array> &inputs,
                                Array &output) {
    if (inputs.size() != 2) {
        throw std::invalid_argument(
            "[Primitives Multiply eval] inputs must have two arrays.");
    }
    auto &a = inputs[0];
    auto &b = inputs[1];
    binary(a, b, output, detail::Multiply());
}
void MultiplyPrimitive::jit(const std::vector<JITTracer> &inputs,
                            JITTracer &output) {}
void MultiplyPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                            JVPTracer &output) {}

std::string MultiplyPrimitive::toString() const { return "multiply"; }

// Reduce

void ReducePrimitive::eval(const std::vector<Array> &inputs, Array &out) {
    evalCPU(inputs, out);
}
void ReducePrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[ReducePrimitive] inputs of mean shoule be one.");
    }
    auto input = inputs[0];
    switch (input.dtype().type) {
    case Dtype::DataType::Any:
        std::invalid_argument(
            "[ReducePrimitive evalcpu] not support Any type.");
        break;
    case Dtype::DataType::BoolType:
        reduce_dispatch<bool>(input, output, reduce_type_, axes_);
        break;
    case Dtype::DataType::Int8Type:
        reduce_dispatch<int8_t>(input, output, reduce_type_, axes_);
        break;
    case Dtype::DataType::Int16Type:
        reduce_dispatch<int16_t>(input, output, reduce_type_, axes_);
        break;
    case Dtype::DataType::Int32Type:
        reduce_dispatch<int32_t>(input, output, reduce_type_, axes_);
        break;
    case Dtype::DataType::Int64Type:
        reduce_dispatch<int64_t>(input, output, reduce_type_, axes_);
        break;
    case Dtype::DataType::Float32Type:
        reduce_dispatch<float>(input, output, reduce_type_, axes_);
        break;
    case Dtype::DataType::Float64Type:
        reduce_dispatch<double>(input, output, reduce_type_, axes_);
        break;
    default:
        break;
    }
}
void ReducePrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void ReducePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string ReducePrimitive::toString() const {
    switch (reduce_type_) {
    case And:
        return "And";
        break;
    case Or:
        return "Or";
        break;
    case Sum:
        return "Sum";
        break;
    case Prod:
        return "Prod";
        break;
    case Min:
        return "Min";
        break;
    case Max:
        return "Max";
        break;
    default:
        break;
    }
}

// convolutio
void ConvolutionPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
    evalCPU(inputs, out);
}
void ConvolutionPrimitive::evalCPU(const std::vector<Array> &inputs,
                                   Array &output) {
    if (inputs.size() != 2) {
        throw std::invalid_argument("[ConvolutionPrimitive evalCPU] inputs of "
                                    "conv must have both input and weight.");
    }
    output.SetDataWithBuffer(allocator::malloc(output.size()), output.dtype(),
                             output.shape(), output.strides());

    auto &input = inputs[0];
    auto &weight = inputs[1];
    // 2D convolution
    if (input.ndim() != 4) {
        throw std::invalid_argument("[ConvolutionPrimitive evalCPU] only "
                                    "support 2dConv,with N,C,W,H.");
    } else {
        conv2d_dispatch(input, weight, output, stride_, padding_, dilation_);
    }
}

void ConvolutionPrimitive::jit(const std::vector<JITTracer> &inputs,
                               JITTracer &output) {}
void ConvolutionPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                               JVPTracer &output) {}
std::string ConvolutionPrimitive::toString() const { return "Conv"; }

void MatMulPrimitive::eval(const std::vector<Array> &inputs, Array &output) {}
void MatMulPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
}
void MatMulPrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void MatMulPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}
std::string MatMulPrimitive::toString() const { return "Matmul"; }

// add::std
} // namespace ainl::core
