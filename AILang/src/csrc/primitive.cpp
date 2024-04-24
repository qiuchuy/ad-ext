#include "primitive.h"

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
    throw std::invalid_argument(
        "[IdentityPrimitive::evalCPU] expects exactly one input array.");
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
    throw std::invalid_argument(
        "[AddPrimitive::evalCPU] input arrays must have the same shape, "
        "broadcasting is not supported yet.");
  }

  auto size = std::accumulate(input1Shape.begin(), input1Shape.end(), 1,
                              std::multiplies<int>());
}

void AddPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {}

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
    throw std::invalid_argument(
        "[FillPrimitive::evalCPU] expects exactly one input array.");
  }

  auto input = inputs[0];
  auto inputShape = input.shape();
  auto size = std::accumulate(inputShape.begin(), inputShape.end(), 1,
                              std::multiplies<int>());
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
    throw std::invalid_argument(
        "[SlicePrimitive::evalCPU] begin, end and stride should have the same "
        "size.");
  }

  const auto input = inputs[0];
  size_t inputNdim = input.ndim();
  if (begin_.size() != inputNdim || end_.size() != inputNdim ||
      stride_.size() != inputNdim) {
    throw std::invalid_argument(
        "[SlicePrimitive::evalCPU] begin, end and stride should have the same "
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
      throw std::invalid_argument("[SlicePrimitive::evalCPU] Illegal slice "
                                  "range for input array.");
    }
  }

  for (size_t i = 0; i < inputNdim; i++) {
    if (end_[i] < -inputShape[i] || end_[i] > inputShape[i]) {
      throw std::invalid_argument("[SlicePrimitive::evalCPU] Illegal slice "
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
  auto size = std::accumulate(inputShape.begin(), inputShape.end(), 1,
                              std::multiplies<int>());
  if (std::accumulate(shape_.begin(), shape_.end(), 1,
                      std::multiplies<int>()) != size) {
    throw std::invalid_argument(
        "[ReshapePrimitive::evalCPU] The total number of elements in the "
        "input array must be the same as the total number of elements in the "
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
  std::vector<ir::TypePtr> inputType = {input.getJITType()};
  std::vector<ir::ValuePtr> inputValues = {input.value()};
  auto outputType =
      ir::getTypeContract().resolveContract("transpose", inputType);

  auto trace = std::dynamic_pointer_cast<JITTrace>(getCurrentTrace());
  output.setValue(ir::getNodeContract().resolveContract(
      "transpose", trace->module(), outputType, inputValues));
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

void MatMulPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
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
        "[MatMulPrimitive::evalCPU] the second dimension of the first input "
        "array must be the same as the first dimension of the second input "
        "array.");
  }

  auto outputShape = {input1Shape[0], input2Shape[1]};
  auto size = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                              std::multiplies<int>());
  // this is wrong, please update it
  output.copyBySharing(input1, size, 0, outputShape);
}

void MatMulPrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[MatMulPrimitive::jit] expects exactly two input tracers.");
  }
  auto input0 = inputs[0];
  auto input1 = inputs[1];
  std::vector<ir::TypePtr> inputType = {input0.value()->getType(),
                                        input1.value()->getType()};
  std::vector<ir::ValuePtr> inputValues = {input0.value(), input1.value()};
  auto outputType = ir::getTypeContract().resolveContract("matmul", inputType);

  auto trace = std::dynamic_pointer_cast<JITTrace>(getCurrentTrace());
  output.setValue(ir::getNodeContract().resolveContract(
      "matmul", trace->module(), outputType, inputValues));
  output.setTracer(transpose({input0.tracer(), input1.tracer()},
                             std::make_shared<MatMulPrimitive>()));
}

void MatMulPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string MatMulPrimitive::toString() const { return "MatMul"; }

} // namespace ainl::core
