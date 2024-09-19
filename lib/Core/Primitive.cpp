
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
#include "ailang/IR/Node.h"
#include "ailang/IR/NodeContract.h"
#include "ailang/IR/Type.h"
#include "ailang/IR/TypeContract.h"
#include "ailang/IR/Value.h"

#include <pybind11/stl.h>

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

void AddPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[AddPrimitive::evalCPU] expects exactly two input arrays.");
  }
  auto input0 = inputs[0];
  auto input1 = inputs[1];
  output = pybind11::cast<Array>(eval_callback["add"](input0, input1));
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

void MatMulPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[MatMulPrimitive::evalCPU] expects exactly two input arrays.");
  }
  auto input0 = inputs[0];
  auto input1 = inputs[1];
  output = pybind11::cast<Array>(eval_callback["matmul"](input0, input1));
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

void BroadcastPrimitive::evalCPU(const std::vector<Array> &inputs,
                                 Array &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[BroadcastPrimitive::evalCPU] expects exactly one input array.");
  }
  auto input = inputs[0];
  auto b = eval_callback["broadcast_to"](input, shape_);
  output = pybind11::cast<Array>(eval_callback["broadcast_to"](input, shape_));
}

TypePtr BroadcastPrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  if (inputTypes.size() != 1) {
    throw std::runtime_error(
        "[BroadcastPrimitive::inferType] expects exactly one input type.");
  }
  auto InputType = inputTypes[0];
  auto TensorType = asType<ir::TensorType>(InputType);
  auto ResultShape = broadcastShapes(TensorType->getConcreteShape(), shape_);
  std::vector<ValuePtr> ResultShapeValue;
  for (auto Dim : ResultShape) {
    ResultShapeValue.push_back(ir::Literal::create(Dim));
  }
  return TensorType::create(TensorType->getElementType(), ResultShapeValue);
}

void BroadcastPrimitive::jit(const std::vector<JITTracer> &inputs,
                             JITTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[BroadcastPrimitive::jit] expects exactly one input tracers.");
  }
  auto input = inputs[0];
  std::vector<ir::ValuePtr> inputValues = {input.value()};
  std::vector<ir::TypePtr> inputType = {input.value()->getType()};
  auto outputType = inferType(inputType);
  output.setValue(
      getTracedModule()->create<Broadcast>(outputType, input.value(), shape_));
  output.setTracer(single<BroadcastPrimitive>({input.tracer()}, shape_));
}

void BroadcastPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                             JVPTracer &output) {}
std::string BroadcastPrimitive::toString() const { return "Broadcast"; }

std::vector<int>
BroadcastPrimitive::broadcastShapes(const std::vector<int> &shape1,
                                    const std::vector<int> &shape2) {
  std::vector<int> resultShape;
  auto it1 = shape1.rbegin();
  auto it2 = shape2.rbegin();

  while (it1 != shape1.rend() || it2 != shape2.rend()) {
    int dim1 = (it1 != shape1.rend()) ? *it1 : 1;
    int dim2 = (it2 != shape2.rend()) ? *it2 : 1;

    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      throw std::runtime_error("Shapes are not compatible for broadcasting.");
    }

    resultShape.push_back(std::max(dim1, dim2));

    if (it1 != shape1.rend())
      ++it1;
    if (it2 != shape2.rend())
      ++it2;
  }
  std::reverse(resultShape.begin(), resultShape.end());
  return resultShape;
}

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

void MultiplyPrimitive::evalCPU(const std::vector<Array> &inputs,
                                Array &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[MultiplyPrimitive::evalCPU] expects exactly two input arrays.");
  }
  auto input0 = inputs[0];
  auto input1 = inputs[1];
  output = pybind11::cast<Array>(eval_callback["mul"](input0, input1));
}

TypePtr MultiplyPrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  assert(inputTypes.size() == 2 && "Div operator only applies to two tensors.");
  auto inType0 = inputTypes[0];
  auto inType1 = inputTypes[1];
  assert(inType0->isTensorType() && "Div operator only applies to tensors.");
  assert(inType1->isTensorType() && "Div operator only applies to tensors.");
  auto inTensorType0 = SAFE_TYPE_DOWNCAST(inType0, TensorType);
  auto inTensorType1 = SAFE_TYPE_DOWNCAST(inType1, TensorType);
  assert(inTensorType0->getElementType() == inTensorType1->getElementType() &&
         "Div operator only applies to tensors with the same element type.");
  return inTensorType0;
}

void MultiplyPrimitive::jit(const std::vector<JITTracer> &inputs,
                            JITTracer &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[MultiplyPrimitive::jit] expects exactly two input tracers.");
  }
  auto input0 = inputs[0];
  auto input1 = inputs[1];
  std::vector<ir::TypePtr> inputType = {input0.value()->getType(),
                                        input1.value()->getType()};
  std::vector<ir::ValuePtr> inputValues = {input0.value(), input1.value()};
  auto outputType = inferType(inputType);
  output.setValue(getTracedModule()->create<Mul>(outputType, input0.value(),
                                                 input1.value()));
  output.setTracer(
      single<MultiplyPrimitive>({input0.tracer(), input1.tracer()}));
}

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
void ConvolutionPrimitive::evalCPU(const std::vector<Array> &inputs,
                                   Array &output) {

  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[ConvolutionPrimitive::evalCPU] expects exactly one input array.");
  }
  auto input = inputs[0];
  auto weight = inputs[1];
  output = pybind11::cast<Array>(
      eval_callback["conv2d"](input, weight, window_strides, lhsDilation,
                              rhsDilation, padding_args, window_reversal));
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
  auto outputType =
      inferType({input.value()->getType(), weight.value()->getType()});
  output.setValue(getTracedModule()->getGraph()->create<Convolution>(
      outputType, input.value(), weight.value(), window_strides, lhsDilation,
      rhsDilation, padding_args, window_reversal));
  output.setTracer(single<ConvolutionPrimitive>(
      {input.tracer(), weight.tracer()}, window_strides, lhsDilation,
      rhsDilation, padding_args, window_reversal));
}
TypePtr
ConvolutionPrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  /*
in_channels (int) – Number of channels in the input image
out_channels (int) – Number of channels produced by the convolution
kernel_size (int or tuple) – Size of the convolving kernel
stride (int or tuple, optional) – Stride of the convolution. Default: 1
padding (int, tuple or str, optional) – Padding added to all four sides of
the input. Default: 0 padding_mode (str, optional) – 'zeros', 'reflect',
'replicate' or 'circular'. Default: 'zeros' dilation (int or tuple,
optional) – Spacing between kernel elements. Default: 1 groups (int,
optional) – Number of blocked connections from input channels to output
channels. Default: 1 bias (bool, optional) – If True, adds a learnable bias
to the output. Default: True
*/
  auto inputType = inputTypes[0];
  auto weightType = inputTypes[1];
  if (!inputType->isTensorType() || !weightType->isTensorType()) {
    throw ainl::core::AINLError(
        "convolution operator only applies to tensors.");
  }
  TensorTypePtr inputTensorType = SAFE_TYPE_DOWNCAST(inputType, TensorType);
  TensorTypePtr weightTensorType = SAFE_TYPE_DOWNCAST(weightType, TensorType);
  std::vector<ValuePtr> inputTensorShape = inputTensorType->getShape();
  std::vector<ValuePtr> weightTensorShape = weightTensorType->getShape();
  std::vector<int> inputConcreateShape = inputTensorType->getConcreteShape();
  std::vector<int> weightConcreateShape = weightTensorType->getConcreteShape();

  if (inputConcreateShape.size() != 4 && weightConcreateShape.size() != 3) {
    throw ainl::core::AINLError(
        "expected input 4d (N,H,W,C) and weight(H,W,C,O), input or weight "
        "dim is not matched.");
  }
  int N = inputConcreateShape[0];
  int C = inputConcreateShape[1];
  int H = inputConcreateShape[2];
  int W = inputConcreateShape[3];

  /*
   in stablehlo the lhs(input img) corresppponds to the NHWC layout.
   And the weights corresponds to HWIO.
   output corresponds to NHWC layout*/
  int padding_h = *padding_args.begin();
  int padding_w = *padding_args.rbegin();

  int lhs_dilation_h = lhsDilation[0];
  int lhs_dilation_w = lhsDilation[1];
  int DilationedInputH = (H - 1) * lhs_dilation_h + 1;
  int DilationedInputW = (W - 1) * lhs_dilation_w + 1;

  int stride_h = window_strides[0];
  int stride_w = window_strides[1];

  int kernel_size_h = weightConcreateShape[2];
  int kernel_size_w = weightConcreateShape[3];
  int rhs_dilation_h = rhsDilation[0];
  int rhs_dilation_w = rhsDilation[1];
  int DilationedWeightH = (kernel_size_h - 1) * rhs_dilation_h + 1;
  int DilationedWeightW = (kernel_size_w - 1) * rhs_dilation_w + 1;
  int I = weightConcreateShape[1];
  int O = weightConcreateShape[0];
  assert(C == I);
  int H_out =
      (DilationedInputH + 2 * padding_h - DilationedWeightH) / stride_h + 1;
  int W_out =
      (DilationedInputW + 2 * padding_w - DilationedWeightW) / stride_w + 1;
  std::vector<ValuePtr> outTensorShape = {
      Literal::create(N),
      Literal::create(O),
      Literal::create(H_out),
      Literal::create(W_out),
  };
  TypePtr elementType = inputTensorType->getElementType();
  return TensorType::create(elementType, outTensorShape);
}

void ConvolutionPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                               JVPTracer &output) {}
std::string ConvolutionPrimitive::toString() const { return "Conv2d"; }
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
  output = pybind11::cast<Array>(eval_callback["mean"](input, dim));
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
  output = pybind11::cast<Array>(eval_callback["var"](input, dim, ddof));
}
TypePtr VariancePrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  assert(inType->isTensorType() && "var operator only applies to tensors.");
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
      outputType, input.value(), dim, ddof));
  output.setTracer(single<VariancePrimitive>({input.tracer()}, dim, ddof));
}

void VariancePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                            JVPTracer &output) {}

std::string VariancePrimitive::toString() const { return "var"; }
// BatchnormInferencePrimitive
void BatchnormInferencePrimitive::eval(const std::vector<Array> &inputs,
                                       Array &output) {
  evalCPU(inputs, output);
}
void BatchnormInferencePrimitive::evalCPU(const std::vector<Array> &inputs,
                                          Array &output) {
  if (inputs.size() != 5)
    throw std::invalid_argument("[BatchnormInferencePrimitive::jit] "
                                "expects exactly five input "
                                "tracer.(i,s,o,m,v)");
  auto input = inputs[0];
  auto scale = inputs[1];
  auto offset = inputs[2];
  auto mean = inputs[3];
  auto variance = inputs[4];

  output = pybind11::cast<Array>(
      eval_callback["batchnorm2d"](input, scale, offset, mean, variance));
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
  auto outputType = inferType({input.value()->getType()});
  output.setValue(getTracedModule()->getGraph()->create<BatchNorm2d>(
      outputType, input.value(), scale.value(), offset.value(), mean.value(),
      variance.value()));
  output.setTracer(single<BatchnormInferencePrimitive>(
      {input.tracer(), scale.tracer(), offset.tracer(), mean.tracer(),
       variance.tracer()}));
}
TypePtr
BatchnormInferencePrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  assert(inType->isTensorType() &&
         "batchnorm operator only applies to tensors.");
  auto inType = inputTypes[0];
  TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
  TypePtr elementType = inTensorType->getElementType();
  return TensorType::create(elementType, inTensorShape);
  // copy-construct
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
void MaxPool2dPrimitive::evalCPU(const std::vector<Array> &inputs,
                                 Array &output) {
  if (inputs.size() != 1)
    throw std::invalid_argument("[MaxPool2dPrimitive::jit] "
                                "expects exactly one input ");
  auto input = inputs[0];
  output = pybind11::cast<Array>(
      eval_callback["maxpool2d"](input, window_dimensions, window_strides,
                                 base_dilations, window_dilations, padding));
}

void MaxPool2dPrimitive::jit(const std::vector<JITTracer> &inputs,
                             JITTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("[MaxPool2dPrimitive::jit] "
                                "expects exactly one input tracer.");
  }
  auto input = inputs[0];
  auto outputType = inferType({input.value()->getType()});
  output.setValue(getTracedModule()->getGraph()->create<Maxpool2d>(
      outputType, input.value(), window_dimensions, window_strides,
      base_dilations, window_dilations, padding));
  output.setTracer(single<MaxPool2dPrimitive>(
      {input.tracer()}, window_dimensions, window_strides, base_dilations,
      window_dilations, padding));
}
TypePtr MaxPool2dPrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  auto inType = inputTypes[0];
  if (!inType->isTensorType()) {
    throw ainl::core::AINLError("maxpool2d operator only applies to tensors.");
  }
  TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
  std::vector<int> inConcreateShape = inTensorType->getConcreteShape();

  std::vector<ValuePtr> outTensorShape;
  int BaseDilationH = base_dilations[2];
  int BaseDilationW = base_dilations[3];
  int WindowDilationH = window_dilations[2];
  int WindowDilationW = window_dilations[3];
  int KernelSizeH = window_dimensions[2];
  int KernelSizeW = window_dimensions[3];
  int WindowStrideH = window_strides[2];
  int WindowStrideW = window_strides[3];
  int ChannelStride = window_strides[1];
  int WindowPaddingH = padding[6];
  int WindowPaddingW = padding[7];
  int InputBatchSize = inConcreateShape[0];
  int InputChannel = inConcreateShape[1];
  int InputH = inConcreateShape[2];
  int InputW = inConcreateShape[3];

  int DilationedWeightH = (KernelSizeH - 1) * WindowDilationH + 1;
  int DilationedWeightW = (KernelSizeW - 1) * WindowDilationW + 1;
  int DilationedInputH = (InputH - 1) * BaseDilationH + 1;
  int DilationedInputW = (InputW - 1) * BaseDilationW + 1;
  int OutBatchSize = window_strides[0];
  int OutChannelSize = InputChannel / ChannelStride;
  int OutH = (DilationedInputH + 2 * WindowPaddingH - DilationedWeightH) /
                 WindowStrideH +
             1;
  int OutW = (DilationedInputW + 2 * WindowPaddingW - DilationedWeightW) /
                 WindowStrideW +
             1;

  // for (const auto &dim : inConcreateShape) {
  //   outTensorShape.push_back(Literal::create(dim));
  // }
  outTensorShape.push_back(Literal::create(OutBatchSize));
  outTensorShape.push_back(Literal::create(OutChannelSize));
  outTensorShape.push_back(Literal::create(OutH));
  outTensorShape.push_back(Literal::create(OutW));
  TypePtr elementType = inTensorType->getElementType();
  return TensorType::create(elementType, outTensorShape);
}

void MaxPool2dPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                             JVPTracer &output) {}

std::string MaxPool2dPrimitive::toString() const {
  return "MaxPool2dPrimitive";
}
// AvgPool2dPrimitive inference
void AvgPool2dPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}
void AvgPool2dPrimitive::evalCPU(const std::vector<Array> &inputs,
                                 Array &output) {
  if (inputs.size() != 1)
    throw std::invalid_argument("[AvgPool2dPrimitive::jit] "
                                "expects exactly one input ");
  auto input = inputs[0];
  output = pybind11::cast<Array>(
      eval_callback["avgpool2d"](input, window_dimensions, window_strides,
                                 base_dilations, window_dilations, padding));
}

void AvgPool2dPrimitive::jit(const std::vector<JITTracer> &inputs,
                             JITTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("[MaxPool2dPrimitive::jit] "
                                "expects exactly one input tracer.");
  }
  auto input = inputs[0];
  auto outputType = inferType({input.value()->getType()});
  output.setValue(getTracedModule()->getGraph()->create<Avgpool2d>(
      outputType, input.value(), window_dimensions, window_strides,
      base_dilations, window_dilations, padding));
  output.setTracer(single<MaxPool2dPrimitive>(
      {input.tracer()}, window_dimensions, window_strides, base_dilations,
      window_dilations, padding));
}
TypePtr AvgPool2dPrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  auto inType = inputTypes[0];
  if (!inType->isTensorType()) {
    throw ainl::core::AINLError(
        "AvgPool2dPrimitive operator only applies to tensors.");
  }
  TensorTypePtr inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  std::vector<ValuePtr> inTensorShape = inTensorType->getShape();
  std::vector<int> inConcreateShape = inTensorType->getConcreteShape();

  std::vector<ValuePtr> outTensorShape;
  int BaseDilationH = base_dilations[2];
  int BaseDilationW = base_dilations[3];
  int WindowDilationH = window_dilations[2];
  int WindowDilationW = window_dilations[3];
  int KernelSizeH = window_dimensions[2];
  int KernelSizeW = window_dimensions[3];
  int WindowStrideH = window_strides[2];
  int WindowStrideW = window_strides[3];
  int ChannelStride = window_strides[1];
  int WindowPaddingH = padding[6];
  int WindowPaddingW = padding[7];
  int InputBatchSize = inConcreateShape[0];
  int InputChannel = inConcreateShape[1];
  int InputH = inConcreateShape[2];
  int InputW = inConcreateShape[3];

  int DilationedWeightH = (KernelSizeH - 1) * WindowDilationH + 1;
  int DilationedWeightW = (KernelSizeW - 1) * WindowDilationW + 1;
  int DilationedInputH = (InputH - 1) * BaseDilationH + 1;
  int DilationedInputW = (InputW - 1) * BaseDilationW + 1;
  int OutBatchSize = window_strides[0];
  int OutChannelSize = InputChannel / ChannelStride;
  int OutH = (DilationedInputH + 2 * WindowPaddingH - DilationedWeightH) /
                 WindowStrideH +
             1;
  int OutW = (DilationedInputW + 2 * WindowPaddingW - DilationedWeightW) /
                 WindowStrideW +
             1;

  // for (const auto &dim : inConcreateShape) {
  //   outTensorShape.push_back(Literal::create(dim));
  // }
  outTensorShape.push_back(Literal::create(OutBatchSize));
  outTensorShape.push_back(Literal::create(OutChannelSize));
  outTensorShape.push_back(Literal::create(OutH));
  outTensorShape.push_back(Literal::create(OutW));
  TypePtr elementType = inTensorType->getElementType();
  return TensorType::create(elementType, outTensorShape);
}

void AvgPool2dPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                             JVPTracer &output) {}

std::string AvgPool2dPrimitive::toString() const {
  return "AvgPool2dPrimitive";
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

void ConcatPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void ConcatPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].dtype().type != inputs[0].dtype().type) {
      throw std::invalid_argument("[ConcatPrimitive::evalCPU] input arrays "
                                  "must have the same dtype.");
    }
  }
  output = pybind11::cast<Array>(eval_callback["cat"](inputs, dim));
}

void ConcatPrimitive::jit(const std::vector<JITTracer> &inputs,
                          JITTracer &output) {
  std::vector<ir::TypePtr> input_types;
  std::vector<ir::ValuePtr> input_values;
  std::vector<std::shared_ptr<Tracer>> tracers;
  for (const auto &input : inputs) {
    input_types.push_back(input.value()->getType());
    input_values.push_back(input.value());
    tracers.push_back(input.tracer());
  }
  auto output_type = inferType(input_types);
  output.setValue(getTracedModule()->getGraph()->create<Concat>(
      output_type, input_values, dim));
  output.setTracer(single<ConcatPrimitive>(tracers, dim));
}

TypePtr ConcatPrimitive::inferType(const std::vector<TypePtr> &input_types) {
  assert(input_types.size() > 0 && "Concat operator requires at least one "
                                   "input tensor.");
  std::vector<ValuePtr> out_tensor_shape;
  auto tensor_type = SAFE_TYPE_DOWNCAST(input_types[0], TensorType);
  auto element_type = tensor_type->getElementType();
  for (size_t Idx = 0; Idx < tensor_type->getShape().size(); ++Idx) {
    if (Idx == dim) {
      int concated_dim = 0;
      for (const auto &input_type : input_types) {
        auto tensor_type = SAFE_TYPE_DOWNCAST(input_type, TensorType);
        auto tensor_shape = tensor_type->getConcreteShape();
        concated_dim += tensor_shape[Idx];
      }
      out_tensor_shape.push_back(ir::Literal::create(concated_dim));
    } else {
      out_tensor_shape.push_back(tensor_type->getShape()[Idx]);
    }
  }
  // TypePtr element_type;

  // for (const auto &input_type : input_types) {
  //   TensorTypePtr tensor_type = SAFE_TYPE_DOWNCAST(input_type, TensorType);
  //   element_type = tensor_type->getElementType();
  //   std::vector<ValuePtr> tensor_shape = tensor_type->getShape();
  //   for (size_t i = 0; i < tensor_shape.size(); ++i) {
  //     if (i == dim) {
  //       uint concated_dim = 0;
  //       for ()
  //     }
  //     out_tensor_shape.push_back(tensor_shape[i]);
  //   }
  // }
  return TensorType::create(element_type, out_tensor_shape);
}

std::string ConcatPrimitive::toString() const { return "Concat"; }

void ExpPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void ExpPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[ExpPrimitive::evalCPU] expects exactly one input array.");
  }
  auto input = inputs[0];
  output = pybind11::cast<Array>(eval_callback["exp"](input));
}

TypePtr ExpPrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  assert(inputTypes.size() == 1 && "Exp operator only applies to one tensor.");
  auto inType = inputTypes[0];
  assert(inType->isTensorType() && "Exp operator only applies to tensors.");
  auto inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  return inTensorType;
}

void ExpPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[ExpPrimitive::jit] expects exactly one input tracer.");
  }
  auto input = inputs[0];
  std::vector<ir::TypePtr> inputType = {input.value()->getType()};
  auto outputType = inferType(inputType);
  output.setValue(getTracedModule()->create<Exp>(outputType, input.value()));
  output.setTracer(single<ExpPrimitive>({input.tracer()}));
}

std::string ExpPrimitive::toString() const { return "Exp"; }

void TanhPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void TanhPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[TanhPrimitive::evalCPU] expects exactly one input array.");
  }
  auto input = inputs[0];
  output = pybind11::cast<Array>(eval_callback["tanh"](input));
}

void TanhPrimitive::jit(const std::vector<JITTracer> &inputs,
                        JITTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[TanhPrimitive::jit] expects exactly one input tracer.");
  }
  auto input = inputs[0];
  std::vector<ir::TypePtr> inputType = {input.value()->getType()};
  auto outputType = inferType(inputType);
  output.setValue(getTracedModule()->create<Tanh>(outputType, input.value()));
  output.setTracer(single<TanhPrimitive>({input.tracer()}));
}

TypePtr TanhPrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  assert(inputTypes.size() == 1 && "Tanh operator only applies to one tensor.");
  auto inType = inputTypes[0];
  assert(inType->isTensorType() && "Tanh operator only applies to tensors.");
  auto inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  return inTensorType;
}

std::string TanhPrimitive::toString() const { return "Tanh"; }

void DivPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void DivPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[DivPrimitive::evalCPU] expects exactly two input arrays.");
  }
  auto input0 = inputs[0];
  auto input1 = inputs[1];
  output = pybind11::cast<Array>(eval_callback["div"](input0, input1));
}

TypePtr DivPrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  assert(inputTypes.size() == 2 && "Div operator only applies to two tensors.");
  auto inType0 = inputTypes[0];
  auto inType1 = inputTypes[1];
  assert(inType0->isTensorType() && "Div operator only applies to tensors.");
  assert(inType1->isTensorType() && "Div operator only applies to tensors.");
  auto inTensorType0 = SAFE_TYPE_DOWNCAST(inType0, TensorType);
  auto inTensorType1 = SAFE_TYPE_DOWNCAST(inType1, TensorType);
  assert(inTensorType0->getElementType() == inTensorType1->getElementType() &&
         "Div operator only applies to tensors with the same element type.");
  return inTensorType0;
}

void DivPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[DivPrimitive::jit] expects exactly two input tracers.");
  }
  auto input0 = inputs[0];
  auto input1 = inputs[1];
  std::vector<ir::TypePtr> inputType = {input0.value()->getType(),
                                        input1.value()->getType()};
  std::vector<ir::ValuePtr> inputValues = {input0.value(), input1.value()};
  auto outputType = inferType(inputType);
  output.setValue(getTracedModule()->create<Div>(outputType, input0.value(),
                                                 input1.value()));
  output.setTracer(single<DivPrimitive>({input0.tracer(), input1.tracer()}));
}

std::string DivPrimitive::toString() const { return "Div"; }

void NegPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void NegPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[NegPrimitive::evalCPU] expects exactly one input array.");
  }
  auto input = inputs[0];
  output = pybind11::cast<Array>(eval_callback["neg"](input));
}

TypePtr NegPrimitive::inferType(const std::vector<TypePtr> &inputTypes) {
  assert(inputTypes.size() == 1 && "Neg operator only applies to one tensor.");
  auto inType = inputTypes[0];
  assert(inType->isTensorType() && "Neg operator only applies to tensors.");
  auto inTensorType = SAFE_TYPE_DOWNCAST(inType, TensorType);
  return inTensorType;
}

void NegPrimitive::jit(const std::vector<JITTracer> &inputs,
                       JITTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[NegPrimitive::jit] expects exactly one input tracer.");
  }
  auto input = inputs[0];
  std::vector<ir::TypePtr> inputType = {input.value()->getType()};
  auto outputType = inferType(inputType);
  output.setValue(getTracedModule()->create<Neg>(outputType, input.value()));
  output.setTracer(single<NegPrimitive>({input.tracer()}));
}

std::string NegPrimitive::toString() const { return "Neg"; }

} // namespace ainl::core
