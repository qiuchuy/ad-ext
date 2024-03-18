#include "primitive.h"
#include "array.h"

namespace ainl::core {

void AddPrimitive::eval(const std::shared_ptr<BaseTrace> &trace,
                        const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void AddPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {}

TypePtr AddPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

void FillPrimitive::eval(const std::shared_ptr<BaseTrace> &trace,
                         const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void FillPrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {}

TypePtr FillPrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

void SlicePrimitive::eval(const std::shared_ptr<BaseTrace> &trace,
                          const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void SlicePrimitive::evalCPU(const std::vector<Array> &inputs, Array &output) {
  // Numpy style slice, see:
  // https://numpy.org/doc/stable/user/basics.indexing.html [TODO] high
  // dimensional slicing
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[SlicePrimitive::evalCPU] expects exactly one input array.");
  }

  const Array &input = inputs[0];
  std::vector<int> input_shape = input.shape();
  size_t input_ndim = input.ndim();

  if (input_ndim == 0) {
    throw std::invalid_argument("[SlicePrimitive::evalCPU] Input array "
                                "must have at least one dimension.");
  }

  int start = begin_;
  int end = end_;
  int stride = stride_;

  if (start < 0) {
    start += input_shape[0];
  }

  if (end < 0) {
    end += input_shape[0];
  }

  if (stride > 0 && end > input_shape[0]) {
    end = input_shape[0];
  } else if (stride < 0 && end < 0) {
    end = -1;
  }

  size_t slice_length = (end - start + stride - 1) / stride;

  Dtype dtype = input.dtype();
  std::vector<int> output_shape(slice_length);
  // Array output(std::move(output_shape), dtype);

  // size_t output_index = 0;
  // for (int i = start; i != end; i += stride) {
  //     int input_index = (i < 0) ? input_shape[0] + i : i;

  //     output.at(output_index++) = input.at(input_index);
  // }
}

TypePtr SlicePrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

void ReshapePrimitive::eval(const std::shared_ptr<BaseTrace> &trace,
                            const std::vector<Array> &inputs, Array &output) {
  evalCPU(inputs, output);
}

void ReshapePrimitive::evalCPU(const std::vector<Array> &inputs,
                               Array &output) {}
TypePtr ReshapePrimitive::typeRalation(const std::vector<TypePtr> &inTypes) {}

} // namespace ainl::core
