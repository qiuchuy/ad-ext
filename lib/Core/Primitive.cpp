
#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "ailang/Core/Array.h"
#include "ailang/Core/Ops.h"
#include "ailang/Core/Primitive.h"
#include "ailang/Core/Trace.h"
#include "ailang/Core/Transformation.h"
#include "ailang/IR/Container.h"
#include "ailang/IR/Function.h"
#include "ailang/IR/Graph.h"
#include "ailang/IR/Literal.h"
#include "ailang/IR/NodeContract.h"
#include "ailang/IR/Type.h"
#include "ailang/IR/TypeContract.h"
#include "ailang/IR/Value.h"

namespace ainl::core {

void UnaryPrimitive::eval(const std::vector<Array> &inputs,
                          std::vector<Array> &outputs) {
  eval(inputs, outputs[0]);
}

void UnaryPrimitive::evalCPU(const std::vector<Array> &inputs,
                             std::vector<Array> &outputs) {
  evalCPU(inputs, outputs[0]);
}

void UnaryPrimitive::jit(const std::vector<JITTracer> &inputs,
                         std::vector<JITTracer> &outputs) {
  jit(inputs, outputs[0]);
}

void UnaryPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                         std::vector<JVPTracer> &outputs) {
  jvp(inputs, outputs[0]);
}

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

void AddPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[AddPrimitive::jit] expects exactly two input tracers.");
  }
  auto input0 = inputs[0];
  auto input1 = inputs[1];
  std::vector<ir::TypePtr> inputType = {input0.value()->getType(),
                                        input1.value()->getType()};
  std::vector<ir::ValuePtr> inputValues = {input0.value(), input1.value()};
  // type inference
  auto outputType = ir::resolveContract("add", inputType);

  auto module = getTracedModule();

  // ir generation
  output.setValue(ir::resolveContract("add", module, outputType, inputValues));
  output.setTracer(single<AddPrimitive>({input0.tracer(), input1.tracer()}));
}

void AddPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                       JVPTracer &output) {}

std::string AddPrimitive::toString() const { return "Add"; }

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

  // calculate output shape from slice parameters and input shape
  std::vector<int> outputShape;
  for (size_t i = 0; i < inputShape.size(); i++) {
    outputShape.push_back((end[i] - begin[i]));
  }

  // calculate the offset, size of the output array
  auto size = input.itemsize();
  for (size_t i = 0; i < outputShape.size(); i++) {
    size *= outputShape[i];
  }

  auto offset = 0;
  for (size_t i = 0; i < inputShape.size(); i++) {
    auto dimOffset = 0;
    for (size_t j = i + 1; j < inputShape.size(); j++) {
      dimOffset += inputShape[j] * input.itemsize();
    }
    offset += begin[i] * dimOffset;
  }
  output.copyBySharing(input, size, offset, outputShape);
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
        "input array must be the same as the total number of elements in "
        "the "
        "output array.");
  }

  output.copyBySharing(input, size, 0, shape_);
}

// [TODO] Implement the JIT method for ReshapePrimitive
void ReshapePrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}

void ReshapePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[ReshapePrimitive::jvp] expects exactly one input tracer.");
  }
  auto input = inputs[0];
  output.setPrimal(single<ReshapePrimitive>({input.primal()}, shape_));
  output.setTangent(single<ReshapePrimitive>({input.tangent()}, shape_));
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
  output = pybind11::cast<Array>(eval_callback["transpose"](input));
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
  output.setTracer(single<TransposePrimitive>({input.tracer()}));
}

void TransposePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                             JVPTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[TransposePrimitive::jvp] expects exactly one input tracer.");
  }
  auto input = inputs[0];
  output.setPrimal(single<TransposePrimitive>({input.primal()}));
  output.setTangent(single<TransposePrimitive>({input.tangent()}));
}

std::string TransposePrimitive::toString() const { return "Transpose"; }

void MatMulPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
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

  // type inference
  auto outputType = ir::resolveContract("matmul", inputType);

  auto module = getTracedModule();

  // ir generation
  output.setValue(
      ir::resolveContract("matmul", module, outputType, inputValues));
  output.setTracer(single<MatMulPrimitive>({input0.tracer(), input1.tracer()}));
}

void MatMulPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string MatMulPrimitive::toString() const { return "MatMul"; }
// AsType
void AsTypePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void AsTypePrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void AsTypePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string AsTypePrimitive::toString() const { return "AsType"; }
// broadcast
void BroadcastPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void BroadcastPrimitive::jit(const std::vector<JITTracer> &inputs,
                             JITTracer &output) {
  // if (inputs.size() != 2) {
  //     throw std::invalid_argument(
  //         "[BroadcastPrimitive::jit] expects exactly one input tracers.");
  // }
  // auto input = inputs[0];
  // // literal or attribute
  // // std::vector<ir::ValuePtr> outputShape_;
  // // to be confirmed. literal value has no Array's achivement.so put all
  // // dims into inputValues.
  // std::vector<ir::ValuePtr> inputValues = {input.value()};

  // for (const auto &dim : shape_) {
  //     inputValues.push_back(ir::Literal::create(dim));
  // }

  // std::vector<ir::TypePtr> inputType = {input.value()->getType()};

  // auto outputType = ir::resolveContract("broadcast", inputType);

  // auto module = getTracedModule();

  // output.setValue(
  //     ir::resolveContract("broadcast", module, outputType, inputValues));
  // output.setTracer(
  //     unary<BroadcastPrimitive>({input.tracer()}, shape_)); // Args...args
}
void BroadcastPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                             JVPTracer &output) {}
std::string BroadcastPrimitive::toString() const { return "Broadcast"; }

// max
void MaximumPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
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

void MinimumPrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}
void MinimumPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string MinimumPrimitive::toString() const { return "Min"; }

// Multiply
void MultiplyPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
  evalCPU(inputs, out);
}
void MultiplyPrimitive::jit(const std::vector<JITTracer> &inputs,
                            JITTracer &output) {}
void MultiplyPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                            JVPTracer &output) {}

std::string MultiplyPrimitive::toString() const { return "multiply"; }

// substract
void SubtractPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}
void SubtractPrimitive::jit(const std::vector<JITTracer> &inputs,
                            JITTracer &output) {}

void SubtractPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                            JVPTracer &output) {}

std::string SubtractPrimitive::toString() const { return "Sub"; }

// SquarePrimitive
void SquarePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}
void SquarePrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void SquarePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string SquarePrimitive::toString() const { return "Square"; }

// Sqrt
void SqrtPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}
void SqrtPrimitive::jit(const std::vector<JITTracer> &inputs,
                        JITTracer &output) {}
void SqrtPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}

std::string SqrtPrimitive::toString() const { return "Sqrt"; }
// sigmoid
void SigmoidPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
  evalCPU(inputs, out);
}
void SigmoidPrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {}
void SigmoidPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string SigmoidPrimitive::toString() const { return "Sigmoid"; }

// Reduce
void ReducePrimitive::eval(const std::vector<Array> &inputs, Array &out) {
  evalCPU(inputs, out);
}
void ReducePrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {}
void ReducePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string ReducePrimitive::toString() const {}

// convolution
void ConvolutionPrimitive::eval(const std::vector<Array> &inputs, Array &out) {
  evalCPU(inputs, out);
}

void ConvolutionPrimitive::jit(const std::vector<JITTracer> &inputs,
                               JITTracer &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[Convolution::jit] expects exactly two input tracers.one is "
        "input, and the other is weight.");
  }
  auto input = inputs[0];
  auto weight = inputs[1];
  std::vector<ir::TypePtr> inputType = {input.value()->getType(),
                                        weight.value()->getType()};
  std::vector<ir::ValuePtr> inputValues = {input.value(), weight.value()};

  // type inference
  auto outputType = ir::resolveContract("convolution", inputType);

  auto module = getTracedModule();

  // ir generation
  output.setValue(
      ir::resolveContract("convolution", module, outputType, inputValues));
  output.setTracer(
      single<ConvolutionPrimitive>({input.tracer(), weight.tracer()}));
}
void ConvolutionPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                               JVPTracer &output) {}
std::string ConvolutionPrimitive::toString() const { return "Conv"; }
// Relu

void ReluPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}
void ReluPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[ReluPrimitive::evalCPU] expects exactly one input array.");
  }
  auto input = inputs[0];
  output = pybind11::cast<Array>(eval_callback["relu"](input));
}

void ReluPrimitive::jit(const std::vector<JITTracer> &inputs,
                        JITTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[ReluPrimitive::jit] expects exactly one input tracer.");
  }

  auto input = inputs[0];
  std::vector<ir::TypePtr> inputType = {input.value()->getType()};
  std::vector<ir::ValuePtr> inputValues = {input.value()};
  auto outputType = ir::resolveContract("relu", inputType);
  auto module = getTracedModule();
  output.setValue(ir::resolveContract("relu", module, outputType, inputValues));
  output.setTracer(single<ReluPrimitive>({input.tracer()}));
}

void ReluPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}

std::string ReluPrimitive::toString() const { return "relu"; }
// Mean
void MeanPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}
void MeanPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[MeanPrimitive::evalCPU] expects exactly one input array.");
  }
  auto input = inputs[0];
  output = pybind11::cast<Array>(eval_callback["mean"](input));
}
TypePtr MeanPrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  assert(inType->isTensorType() && "mean operator only applies to tensors.");
  auto inType = inputTypes[0];
  TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
  std::vector<ValuePtr> outTensorshape;
  for (size_t Idx = 0; Idx < inTensorShape.size(); ++Idx) {
    if (std::find(dim.begin(), dim.end(), Idx) == dim.end()) {
      outTensorshape.push_back(inTensorShape[Idx]);
    }
  }
  TypePtr elementType = inTensorType->getElementType();
  return TensorType::create(elementType, outTensorshape);
}

void MeanPrimitive::jit(const std::vector<JITTracer> &inputs,
                        JITTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[MeanPrimitive::jit] expects exactly one input tracer.");
  }
  auto input = inputs[0];
  auto outputType = inferType({input.value()->getType()});
  output.setValue(getTracedModule()->getGraph()->create<Mean>(
      outputType, input.value(), dim));
  output.setTracer(single<MeanPrimitive>({input.tracer()}, dim));
}

void MeanPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) {}

std::string MeanPrimitive::toString() const { return "mean"; }
// VariancePrimitive
void VariancePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}
void VariancePrimitive::evalCPU(const std::vector<Array> &inputs,
                                Array &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[VariancePrimitive::evalCPU] expects exactly one input array.");
  }
  auto input = inputs[0];
  output = pybind11::cast<Array>(eval_callback["var"](input));
}
TypePtr VariancePrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  assert(inType->isTensorType() && "mean operator only applies to tensors.");
  auto inType = inputTypes[0];
  TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
  std::vector<ValuePtr> outTensorshape;
  for (size_t Idx = 0; Idx < inTensorShape.size(); ++Idx) {
    if (std::find(dim.begin(), dim.end(), Idx) == dim.end()) {
      outTensorshape.push_back(inTensorShape[Idx]);
    }
  }
  TypePtr elementType = inTensorType->getElementType();
  return TensorType::create(elementType, outTensorshape);
}

void VariancePrimitive::jit(const std::vector<JITTracer> &inputs,
                            JITTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[VariancePrimitive::jit] expects exactly one input tracer.");
  }
  auto input = inputs[0];
  auto outputType = inferType({input.value()->getType()});
  output.setValue(getTracedModule()->getGraph()->create<Variance>(
      outputType, input.value(), dim));
  output.setTracer(single<VariancePrimitive>({input.tracer()}, dim));
}

void VariancePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                            JVPTracer &output) {}

std::string VariancePrimitive::toString() const { return "var"; }
// BatchnormInferencePrimitive
void BatchnormInferencePrimitive::eval(const std::vector<Array> &inputs,
                                       Array &output) {
  evalCPU(inputs, output);
}

void BatchnormInferencePrimitive::jit(const std::vector<JITTracer> &inputs,
                                      JITTracer &output) {
  if (inputs.size() != 5) {
    throw std::invalid_argument("[BatchnormInferencePrimitive::jit] "
                                "expects exactly one input tracer.");
  }
  auto input = inputs[0];
  auto scale = inputs[1];
  auto offset = inputs[2];
  auto mean = inputs[3];
  auto variance = inputs[4];
  std::vector<ir::TypePtr> inputType = {
      input.value()->getType(), scale.value()->getType(),
      offset.value()->getType(), mean.value()->getType(),
      variance.value()->getType()};
  std::vector<ir::ValuePtr> inputValues = {input.value(), scale.value(),
                                           offset.value(), mean.value(),
                                           variance.value()};
  auto outputType = ir::resolveContract("batchnorm2d", inputType);
  auto module = getTracedModule();
  output.setValue(
      ir::resolveContract("batchnorm2d", module, outputType, inputValues));
  output.setTracer(single<BatchnormInferencePrimitive>({input.tracer()}));
}

void BatchnormInferencePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                                      JVPTracer &output) {}

std::string BatchnormInferencePrimitive::toString() const {
  return "BatchnormInference";
}
// MaxPool2dPrimitive inference
void MaxPool2dPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void MaxPool2dPrimitive::jit(const std::vector<JITTracer> &inputs,
                             JITTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("[MaxPool2dPrimitive::jit] "
                                "expects exactly one input tracer.");
  }
  auto input = inputs[0];
  std::vector<ir::TypePtr> inputType = {input.value()->getType()};
  std::vector<ir::ValuePtr> inputValues = {input.value()};
  auto outputType = ir::resolveContract("maxpool2d", inputType);
  auto module = getTracedModule();
  output.setValue(
      ir::resolveContract("maxpool2d", module, outputType, inputValues));
  output.setTracer(single<MaxPool2dPrimitive>({input.tracer()}));
}

void MaxPool2dPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                             JVPTracer &output) {}

std::string MaxPool2dPrimitive::toString() const {
  return "MaxPool2dPrimitive";
}
// GetElementsNumberPrimitive
void GetElementsNumberPrimitive::eval(const std::vector<Array> &inputs,
                                      Array &out) {
  evalCPU(inputs, out);
}
void GetElementsNumberPrimitive::jit(const std::vector<JITTracer> &inputs,
                                     JITTracer &output) {}
void GetElementsNumberPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                                     JVPTracer &output) {}

std::string GetElementsNumberPrimitive::toString() const {
  return "GetElementsNumber";
}

void ComparePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void ComparePrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("[ComparePrimitive::evalCPU] expects "
                                "exactly two input arrays.");
  }

  auto lhs = inputs[0];
  auto rhs = inputs[1];
  auto dtype1 = inputs[0].dtype();
  auto dtype2 = inputs[1].dtype();
  if (dtype1.type != dtype2.type) {
    throw std::invalid_argument("[ComparePrimitive::evalCPU] input arrays "
                                "must have the same dtype.");
  }
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument("[ComparePrimitive::evalCPU] input arrays "
                                "must have the same shape.");
  }

  switch (dtype1.type) {
  case Dtype::DataType::BoolType: {
    compare<bool>(lhs, rhs, output);
    break;
  }
  case Dtype::DataType::Int8Type: {
    compare<int8_t>(lhs, rhs, output);
    break;
  }
  case Dtype::DataType::Int16Type: {
    compare<int16_t>(lhs, rhs, output);
    break;
  }
  case Dtype::DataType::Int32Type: {
    compare<int32_t>(lhs, rhs, output);
    break;
  }
  case Dtype::DataType::Int64Type: {
    compare<int64_t>(lhs, rhs, output);
    break;
  }
  case Dtype::DataType::Float32Type: {
    compare<float>(lhs, rhs, output);
    break;
  }
  case Dtype::DataType::Float64Type: {
    compare<double>(lhs, rhs, output);
    break;
  }
  }
}

void ComparePrimitive::jit(const std::vector<JITTracer> &inputs,
                           JITTracer &output) {
  auto input0 = inputs[0];
  auto input1 = inputs[1];
  std::vector<ir::TypePtr> inputType = {input0.value()->getType(),
                                        input1.value()->getType()};
  std::vector<ir::ValuePtr> inputValues = {input0.value(), input1.value()};

  std::string compareOp = ir::compareOpString[static_cast<size_t>(op_)];

  auto outputType = ir::resolveContract(compareOp, inputType);

  auto module = getTracedModule();

  output.setValue(
      ir::resolveContract(compareOp, module, outputType, inputValues));

  std::vector<std::shared_ptr<Tracer>> tracers;
  for (const auto &input : inputs) {
    tracers.push_back(input.tracer());
  }

  output.setTracer(single<ComparePrimitive>(tracers, op_));
}

void ComparePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string ComparePrimitive::toString() const { return "Compare"; }

} // namespace ainl::core
