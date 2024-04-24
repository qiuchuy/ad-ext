#include <algorithm>
#include <numeric>

#include "ops.h"
#include "primitive.h"
#include "transformation.h"
#include "unary.h"

namespace ainl::core {

void IdentityPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void IdentityPrimitive::evalCPU(const std::vector<Array> &inputs,
                                Array &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[IdentityPrimitive::evalCPU] expects exactly one input array.");
    }
    output.copyBySharing(inputs[0], 0, 0, inputs[0].shape());
}

TypePtr IdentityPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

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
        throw std::invalid_argument(
            "[AddPrimitive::evalCPU] input arrays must have the same shape, "
            "broadcasting is not supported yet.");
    }

    auto size = std::accumulate(input1Shape.begin(), input1Shape.end(), 1,
                                std::multiplies<int>());
}

TypePtr AddPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

void AddPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}

std::string AddPrimitive::toString() const { return "Add"; }

void FlattenPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void FlattenPrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
    /*
    if (inputs.size() != 1) {
      throw std::invalid_argument(
          "[FlattenPrimitive::evalCPU] expects exactly one input array.");
    }

    auto input = inputs[0];
    auto inputShape = input.shape();
    auto size = std::accumulate(inputShape.begin(), inputShape.end(), 1,
                                std::multiplies<int>());
    output.copyBySharing(input, size, 0, {size});
    */
}

TypePtr FlattenPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

void FlattenPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string FlattenPrimitive::toString() const { return "Flatten"; }

void FillPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
    evalCPU(inputs, output);
}

void FillPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
    if (inputs.size() != 1) {
        if (inputs.size() != 1) {
            std::invalid_argument(
                "[FillPrimitive::evalCPU] expects exactly one input array.");
        }
        const auto &input = inputs[0];
        if (input.dtype() != output.dtype())
            std::runtime_error(
                "[FillPrimitive] input and output dont have the same dtype.");
        auto inputShape = input.shape();
        auto size = std::accumulate(inputShape.begin(), inputShape.end(), 1,
                                    std::multiplies<int>());
        output.copyBySharing(input, size, 0, inputShape);
    }
}

TypePtr FillPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

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

TypePtr SlicePrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

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
    auto size = std::accumulate(inputShape.begin(), inputShape.end(), 1,
                                std::multiplies<int>());
    if (std::accumulate(shape_.begin(), shape_.end(), 1,
                        std::multiplies<int>()) != size) {
        throw std::invalid_argument(
            "[ReshapePrimitive::evalCPU] The total number of elements in the "
            "input array must be the same as the total number of elements in "
            "the "
            "output array.");
    }

    output.copyBySharing(input, size, 0, shape_);
}
TypePtr ReshapePrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

void ReshapePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {
    if (inputs.size() != 1) {
        throw std::invalid_argument(
            "[ReshapePrimitive::jvp] expects exactly one input tracer.");
    }
    auto input = inputs[0];

    output.setPrimal(
        reshape_({input.primal()}, std::make_shared<ReshapePrimitive>(shape_)));
    output.setTangent(reshape_({input.tangent()},
                               std::make_shared<ReshapePrimitive>(shape_)));
}

std::string ReshapePrimitive::toString() const { return "Reshape"; }

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
void AbsPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}
TypePtr AbsPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

std::string AbsPrimitive::toString() const { return "Abs"; }

// TODO ADDMM
void AddMMPrimitive::eval(const std::vector<Array> &inputs, Array &output) {}
void AddMMPrimitive::evalCPU(const std::vector<Array> &inputs, Array &outputs) {
}
TypePtr AddMMPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
                             output.shape(), output.strides()); // malloc bytes
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
void ArangePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}
TypePtr ArangePrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void AsTypePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}
TypePtr AsTypePrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void ArcCosPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}
TypePtr ArcCosPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void ArcTanPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}
TypePtr ArcTanPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void ArcSinPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}
TypePtr ArcSinPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void ArcCoshPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}
TypePtr ArcCoshPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void ArcTanhPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}
TypePtr ArcTanhPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void ArcSinhPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}
TypePtr ArcSinhPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void CosPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}
TypePtr CosPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void TanPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}
TypePtr TanPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void SinPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}
TypePtr SinPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void CoshPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}
TypePtr CoshPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void TanhPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}
TypePtr TanhPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void SinhPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}
TypePtr SinhPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
std::string SinhPrimitive::toString() const { return "Sinh"; }

// BroadCast
void BroadCastPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
    evalCPU(inputs, out);
}
void BroadCastPrimitive::evalCPU(const std::vector<Array> &inputs,
                                 Array &output) {
    if (inputs.size() != 0) {
        std::invalid_argument("[BroadCastPrimitive::evalCPU] expects "
                              "exactly one input array.");
    }
    const auto &input = inputs[0];
    if (output.size() == 0) {
        std::invalid_argument(
            "[BroadCastPrimitive::evalCPU] output size cant be zero. ");
    }
    output.copyBySharing(input, input.data_size(), 0, shape_);
}
void BroadCastPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                             JVPTracer &output) {}
TypePtr BroadCastPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void ExpPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}
TypePtr ExpPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void LogPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}
TypePtr LogPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void SigmoidPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}
TypePtr SigmoidPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
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
void SoftmaxPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}
TypePtr SoftmaxPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}
std::string SoftmaxPrimitive::toString() const { return "Softmax"; }

// convolution

// TODO MKL
// add::std

} // namespace ainl::core
