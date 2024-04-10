#include <algorithm>
#include <numeric>

#include "primitive.h"
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

TypePtr IdentityPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

JVPTracer IdentityPrimitive::jvp(const std::vector<JVPTracer> &inputs,
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

JVPTracer AddPrimitive::jvp(const std::vector<JVPTracer> &inputs,
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

JVPTracer FlattenPrimitive::jvp(const std::vector<JVPTracer> &inputs,
                                JVPTracer &output) {}

std::string FlattenPrimitive::toString() const { return "Flatten"; }

void FillPrimitive::eval(const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void FillPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {}

TypePtr FillPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

JVPTracer FillPrimitive::jvp(const std::vector<JVPTracer> &inputs,
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

TypePtr SlicePrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

JVPTracer SlicePrimitive::jvp(const std::vector<JVPTracer> &inputs,
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
TypePtr ReshapePrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

JVPTracer ReshapePrimitive::jvp(const std::vector<JVPTracer> &inputs,
                                JVPTracer &output) {
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[ReshapePrimitive::jvp] expects exactly one input tracer.");
  }

  auto input = inputs[0];

  output.setPrimal(std::make_shared<Tracer>(
      std::vector<std::shared_ptr<Tracer>>{input.primal()},
      std::make_shared<ReshapePrimitive>(shape_)));
  output.setTangent(std::make_shared<Tracer>(
      std::vector<std::shared_ptr<Tracer>>{input.tangent()},
      std::make_shared<ReshapePrimitive>(shape_)));
}

std::string ReshapePrimitive::toString() const { return "Reshape"; }

} // namespace ainl::core
