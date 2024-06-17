#include "primitive.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>

#include "array.h"
#include "ast/node_contract.h"
#include "ast/type_contract.h"
#include "ir/container.h"
#include "ir/function.h"
#include "ir/graph.h"
#include "ir/type.h"
#include "ir/value.h"
#include "ops.h"
#include "trace.h"
#include "transformation.h"

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
  output.setPrimal(unary<ReshapePrimitive>({input.primal()}, shape_));
  output.setTangent(unary<ReshapePrimitive>({input.tangent()}, shape_));
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
  std::vector<ir::TypePtr> inputType = {input.value()->getType()};
  std::vector<ir::ValuePtr> inputValues = {input.value()};
  auto outputType = ir::resolveContract("transpose", inputType);

  auto module = getTracedModule();
  output.setValue(
      ir::resolveContract("transpose", module, outputType, inputValues));
  output.setTracer(unary<TransposePrimitive>({input.tracer()}));
}

void TransposePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                             JVPTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[TransposePrimitive::jvp] expects exactly one input tracer.");
  }
  auto input = inputs[0];
  output.setPrimal(unary<TransposePrimitive>({input.primal()}));
  output.setTangent(unary<TransposePrimitive>({input.tangent()}));
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

  // type inference
  auto outputType = ir::resolveContract("matmul", inputType);

  auto module = getTracedModule();

  // ir generation
  output.setValue(
      ir::resolveContract("matmul", module, outputType, inputValues));
  output.setTracer(unary<MatMulPrimitive>({input0.tracer(), input1.tracer()}));
}

void MatMulPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                          JVPTracer &output) {}

std::string MatMulPrimitive::toString() const { return "MatMul"; }

void LoopPrimitive::eval(const std::vector<Array> &inputs,
                         std::vector<Array> &output) {
  evalCPU(inputs, output);
}

void LoopPrimitive::evalCPU(const std::vector<Array> &inputs,
                            std::vector<Array> &output) {
  auto inits = convertTracerSharedPtrVector(inputs);
  while (1) {
    auto cond = cond_(inits);
    if (auto array = asTracer<Array>(cond)) {
      if (!array->item<bool>()) {
        break;
      }
    }
    inits = body_(inits);
  }
  output = convertTracerVector<Array>(inits);
}

void LoopPrimitive::jit(const std::vector<JITTracer> &inputs,
                        std::vector<JITTracer> &outputs) {
  std::vector<ir::TypePtr> inputsTypes;
  // for loop primitive, we enforce the type of output variables match the type
  // of input variables at the top level
  for (auto &input : inputs) {
    inputsTypes.push_back(input.value()->getType());
  }
  auto outputTypes = ir::TupleType::createUnnamedTuple(inputsTypes);
  std::vector<ir::ValuePtr> inputValues;
  for (auto &input : inputs) {
    inputValues.push_back(input.value());
  }

  auto inits = convertTracerSharedPtrVector(inputs);
  std::vector<std::shared_ptr<Tracer>> tracers;
  for (const auto &input : inputs) {
    tracers.push_back(input.tracer());
  }

  auto module = getTracedModule();
  auto savedModule = *module;

  // jit cond_ and body_ respectively
  auto jitCond = cond_;
  auto condWrapper =
      [jitCond](const std::vector<std::shared_ptr<Tracer>> &inits) {
        auto cond = jitCond(inits);
        return std::vector<std::shared_ptr<Tracer>>{cond};
      };

  auto condModule = ainl::core::jit(condWrapper, "cond", "", tracers);

  auto bodyModule = ainl::core::jit(body_, "body", "", tracers);

  *module = savedModule;

  inputValues.push_back(condModule.get());
  inputValues.push_back(bodyModule.get());

  auto whileOp = ir::asValueType<ir::WhileOp>(
      ir::resolveContract("loop", module, outputTypes, inputValues));

  getCurrentTrace()->unpack(inits);
  auto outputTracers = op<LoopPrimitive>(inits, cond_, body_);
  for (size_t i = 0; i < whileOp->getOutputValues().size(); i++) {
    outputs[i].setValue(whileOp->getOutputValues()[i]);
    outputs[i].setTracer(outputTracers[i]);
  }
}

void LoopPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                        std::vector<JVPTracer> &output) {}

std::string LoopPrimitive::toString() const { return "Loop"; }

void ComparePrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void ComparePrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "[ComparePrimitive::evalCPU] expects exactly two input arrays.");
  }

  auto lhs = inputs[0];
  auto rhs = inputs[1];
  auto dtype1 = inputs[0].dtype();
  auto dtype2 = inputs[1].dtype();
  if (dtype1.type != dtype2.type) {
    throw std::invalid_argument(
        "[ComparePrimitive::evalCPU] input arrays must have the same dtype.");
  }
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument(
        "[ComparePrimitive::evalCPU] input arrays must have the same shape.");
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

  output.setTracer(unary<ComparePrimitive>(tracers, op_));
}

void ComparePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                           JVPTracer &output) {}

std::string ComparePrimitive::toString() const { return "Compare"; }

void IfPrimitive::eval(const std::vector<Array> &inputs,
                       std::vector<Array> &output) {
  evalCPU(inputs, output);
}

void IfPrimitive::evalCPU(const std::vector<Array> &inputs,
                          std::vector<Array> &output) {}

void IfPrimitive::jit(const std::vector<JITTracer> &inputs,
                      std::vector<JITTracer> &outputs) {
  std::vector<ir::TypePtr> inputsTypes;
  // for if primitive, we enforce the type of output variables match the type
  // of input variables at the top level
  for (auto &input : inputs) {
    inputsTypes.push_back(input.value()->getType());
  }
  auto outputTypes = ir::TupleType::createUnnamedTuple(inputsTypes);
  std::vector<ir::ValuePtr> inputValues;
  for (auto &input : inputs) {
    inputValues.push_back(input.value());
  }

  auto inits = convertTracerSharedPtrVector(inputs);
  std::vector<std::shared_ptr<Tracer>> tracers;
  for (const auto &input : inputs) {
    tracers.push_back(input.tracer());
  }

  auto module = getTracedModule();
  auto savedModule = *module;

  auto trueBranchModule =
      ainl::core::jit(trueBranch, "trueBranch", "", tracers);

  auto falseBranchModule =
      ainl::core::jit(falseBranch, "falseBranch", "", tracers);

  *module = savedModule;

  inputValues.push_back(trueBranchModule.get());
  inputValues.push_back(falseBranchModule.get());

  auto ifOp = ir::asValueType<ir::IfOp>(
      ir::resolveContract("if", module, outputTypes, inputValues));

  getCurrentTrace()->unpack(inits);
  auto outputTracers = op<IfPrimitive>(inits, trueBranch, falseBranch);

  for (size_t i = 0; i < ifOp->getOutputValues().size(); i++) {
    outputs[i].setValue(ifOp->getOutputValues()[i]);
    outputs[i].setTracer(outputTracers[i]);
  }
}

void IfPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                      std::vector<JVPTracer> &output) {}

std::string IfPrimitive::toString() const { return "If"; }

} // namespace ainl::core
