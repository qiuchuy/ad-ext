#include "ops.h"

#include "primitive.h"

namespace ainl::core {

Array zeros(const std::vector<int> &shape, Dtype dtype) {
  return fill(shape, Array(0, dtype), dtype);
}

Array ones(const std::vector<int> &shape, Dtype dtype) {
  return fill(shape, Array(1, dtype), dtype);
}

Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype) {
  return Array(dtype, std::make_shared<FillPrimitive>(), {value}, shape,
               value.strides());
}

Array slice(const Array &input, const std::vector<int> &start,
            const std::vector<int> &end, const std::vector<int> &stride) {
  auto outputShape = std::vector<int>();
  for (size_t i = 0; i < input.ndim(); i++) {
    auto s = (end[i] - start[i] + stride[i] - 1) / stride[i];
    if (s < 0) {
      s = 0;
    }
    outputShape.push_back(s);
  }

  return Array(input.dtype(),
               std::make_shared<SlicePrimitive>(start, end, stride), {input},
               outputShape, getStridesFromShape(outputShape, input.itemsize()));
}

Array reshape(const Array &input, const std::vector<int> &shape) {
  return Array(input.dtype(), std::make_shared<ReshapePrimitive>(shape),
               {input}, shape, getStridesFromShape(shape, input.itemsize()));
}

Array transpose(const Array &input) {
  return Array(input.dtype(), std::make_shared<TransposePrimitive>(), {input},
               input.shape(), input.strides());
}

Array matmul(const Array &lhs, const Array &rhs) {
  std::vector<int> shape = {*lhs.shape().begin(), *(rhs.shape().end())};
  return Array(lhs.dtype(), std::make_shared<MatMulPrimitive>(), {lhs, rhs},
               shape, getStridesFromShape(shape, lhs.itemsize()));
}

Array flatten(const Array &input) {
  auto shape = input.shape();
  int totalShape =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  std::vector<int> flattenShape = {totalShape};
  return Array(input.dtype(), std::make_shared<ReshapePrimitive>(flattenShape),
               {input}, flattenShape,
               getStridesFromShape(flattenShape, input.itemsize()));
}

std::vector<int> getStridesFromShape(const std::vector<int> &shape,
                                     size_t itemsize) {
  std::vector<int> strides;
  for (size_t i = 0; i < shape.size(); i++) {
    int stride = 1;
    for (size_t j = i + 1; j < shape.size(); j++) {
      stride *= shape[j];
    }
    strides.push_back(stride * itemsize);
  }
  return strides;
}

GENERIC_OP_IMPL(reshape)
GENERIC_OP_IMPL(transpose)
GENERIC_OP_IMPL(matmul)

}; // namespace ainl::core
