#include "ailang/Core/Ops.h"
#include "ailang/Core/Primitive.h"
#include <set>

namespace ainl::core {

Dtype isFloat(Dtype dtype) { return dtype < Float32 ? Float32 : dtype; }

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

// Array add(const Array &lhs, const Array &rhs) {
//     std::vector<int> shape = lhs.shape();
//     return Array(lhs.dtype(), std::make_shared<AddPrimitive>(), {lhs, rhs},
//                  lhs.shape(), getStridesFromShape(lhs.shape(),
//                  lhs.itemsize()));
// }
Array add(const Array &lhs, const Array &rhs) {
  Dtype output_type = lhs.dtype() < rhs.dtype() ? lhs.dtype() : rhs.dtype();
  auto inputs =
      broadcastArrays({astype(lhs, output_type), astype(rhs, output_type)});
  std::vector<int> output_shape = inputs[0].shape();
  return Array(output_type, std::make_shared<AddPrimitive>(), inputs,
               output_shape,
               getStridesFromShape(output_shape, dtypeSize(output_type)));
}

Array astype(const Array &input, Dtype dtype) {
  if (input.dtype() == dtype)
    return input;
  return Array(dtype, std::make_shared<AsTypePrimitive>(dtype), {input},
               input.shape(),
               getStridesFromShape(input.shape(), dtypeSize(dtype)));
}

Array maximum(const Array &lhs, const Array &rhs) {
  auto output_type = lhs.dtype();
  auto inputs =
      broadcastArrays({astype(lhs, output_type), astype(rhs, output_type)});

  std::vector<int> output_shape = inputs[0].shape();
  return Array(output_type, std::make_shared<MaximumPrimitive>(),
               std::move(inputs), output_shape,
               getStridesFromShape(output_shape, dtypeSize(output_type)));
}
Array minimum(const Array &lhs, const Array &rhs) {
  auto output_type = lhs.dtype();
  auto inputs =
      broadcastArrays({astype(lhs, output_type), astype(rhs, output_type)});

  std::vector<int> output_shape = inputs[0].shape();
  return Array(output_type, std::make_shared<MinimumPrimitive>(),
               std::move(inputs), output_shape,
               getStridesFromShape(output_shape, dtypeSize(output_type)));
}
Array multiply(const Array &rhs, const Array &lhs) {
  // promote types
  auto output_type = rhs.dtype();
  // auto inputs = broadcastArrays({a, b});
  // std::cout<<inputs[0]<<inputs[1];
  return Array(output_type, std::make_shared<MultiplyPrimitive>(), {rhs, lhs},
               rhs.shape(), rhs.strides());
}
Array sigmoid(const Array &input) {
  Dtype output_type = isFloat(input.dtype());
  return Array(output_type, std::make_shared<SigmoidPrimitive>(), {input},
               input.shape(),
               getStridesFromShape(input.shape(), dtypeSize(output_type)));
}
Array subtract(const Array &lhs, const Array &rhs) {
  Dtype output_type = lhs.dtype() < rhs.dtype() ? lhs.dtype() : rhs.dtype();
  auto inputs =
      broadcastArrays({astype(lhs, output_type), astype(rhs, output_type)});

  std::vector<int> output_shape = inputs[0].shape();
  return Array(output_type, std::make_shared<SubtractPrimitive>(),
               std::move(inputs), output_shape,
               getStridesFromShape(output_shape, dtypeSize(output_type)));
}

Array sum(const Array &input, const std::vector<int> &axes, bool keepdims) {
  if (axes.empty()) {
    return input;
  }
  auto [output_shape, sorted_axes] = getReduceShape(axes, input.shape());
  auto output_type = input.dtype();
  auto out = Array(
      output_type,
      std::make_shared<ReducePrimitive>(sorted_axes, ReducePrimitive::Sum),
      {input}, output_shape,
      getStridesFromShape(output_shape, dtypeSize(output_type)));
  if (keepdims) {
    return out;
  } else {
    return squeeze(out, sorted_axes);
  }
}

Array squeeze(const Array &input, const std::vector<int> &axes) {
  std::set<int> axes_set;
  int ndim = input.ndim();
  for (auto axis : axes) {
    auto ax_ = axis < 0 ? axis + ndim : axis;
    if (ax_ < 0 || ax_ > ndim) {
      throw std::invalid_argument("[Ops Squeeze] given axes out of range");
    }
    // must specific axis = 1
    if (input.shape()[ax_] != 1) {
      throw std::invalid_argument("[Ops Squeeze] cannot deal with axis == 1.");
    }
    axes_set.insert(ax_);
  }
  if (axes_set.size() != axes.size()) {
    throw std::invalid_argument("[Ops squeeze] Received duplicate axes.");
  }
  // make sure squeeze in order
  std::vector<int> sorted_axes(axes_set.begin(), axes_set.end());
  std::vector<int> shape;
  for (int i = 0, j = 0; i < ndim; ++i) {
    // if current axis_i is in specific axes, this dimension will be
    // squeezed，else maintain old
    if (j < sorted_axes.size() && i == sorted_axes[j]) {
      j++;
    } else {
      shape.push_back(input.shape()[i]);
    }
  }
  for (auto it : shape) {
    std::cout << it << std::endl;
  }
  // maybe shape = {}
  if (!shape.size()) {
    shape.push_back(1);
  }
  return reshape(input, shape);
}

//  / n-1
Array square(const Array &input) {
  Dtype output_type = isFloat(input.dtype());
  return Array(output_type, std::make_shared<SquarePrimitive>(), {input},
               input.shape(),
               getStridesFromShape(input.shape(), dtypeSize(output_type)));
}
Array sqrt(const Array &input) {
  Dtype output_type = isFloat(input.dtype());
  return Array(output_type, std::make_shared<SqrtPrimitive>(false), {input},
               input.shape(),
               getStridesFromShape(input.shape(), dtypeSize(output_type)));
}
Array rsqrt(const Array &input) {
  Dtype output_type = isFloat(input.dtype());
  return Array(output_type, std::make_shared<SqrtPrimitive>(true), {input},
               input.shape(),
               getStridesFromShape(input.shape(), dtypeSize(output_type)));
}

Array var(const Array &input, const std::vector<int> &axes, bool keepdims,
          int ddof) {
  int ndim = input.ndim();

  for (int axis : axes) {
    if (axis < -ndim || axis >= ndim) {
      throw std::invalid_argument(
          "[Ops.cpp var] axis is out of bounds for inputay");
    }
  }
  if (ddof != 0) {
    throw std::invalid_argument("[Ops.cpp var] only suport ddof =0.");
  }
  auto output_type = input.dtype();
  auto m2 = square(mean(input, axes, keepdims));
  auto a2 = mean(square(input), axes, keepdims);

  auto v = subtract(a2, m2);
  return v;
}

Array var(const Array &input, bool keepdims) {
  std::vector<int> axes(input.ndim());

  std::iota(axes.begin(), axes.end(), 0);
  return var(input, axes, keepdims, 0);
}
Array var(const Array &input, const std::vector<int> &axes, bool keepdims) {
  return var(input, axes, keepdims, 0);
}
Array var(const Array &input, int axis, bool keepdims) {
  return var(input, {axis}, keepdims, 0);
}

Array mean(const Array &input, bool keepdims) {
  std::vector<int> axes(input.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return mean(input, axes, keepdims);
}
Array mean(const Array &input, const std::vector<int> &axes, bool keepdims) {
  int ndim = input.ndim();
  for (int axis : axes) {
    if (axis < -ndim || axis >= ndim) {
      throw std::invalid_argument(
          "[Ops.cpp mean] axis is out of bounds for array");
    }
  }
  auto output_type = input.dtype();
  // sum
  auto normalizer = getElementsNumber(input, axes, true, output_type);
  return multiply(sum(input, axes, keepdims), normalizer);
}

Array mean(const Array &input, int axis, bool keepdims) {
  std::vector<int> axes = {axis};
  return mean(input, axes, keepdims);
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
Array broadcast_to(const Array &input, const std::vector<int> &shape) {
  if (input.shape() == shape)
    return input;
  // check is broadcastable
  std::vector<int> outputShape = broadcastShapes(input.shape(), shape);
  return Array(input.dtype(), std::make_shared<BroadcastPrimitive>(outputShape),
               {input}, outputShape,
               getStridesFromShape(outputShape, input.itemsize()));
}

// Array conv2d(const Array &input, const Array &weight,
//              const std::pair<int, int> &stride /* {2, 2}*/,
//              const std::pair<int, int> &padding /* {0, 0}*/,
//              const std::pair<int, int> &dilation /*{1, 1}*/) {
//   int spatial_dims = input.ndim() - 2;
//   if (spatial_dims < 1 || spatial_dims > 2) {
//     throw std::invalid_argument(
//         "[ops.h Conv2d cannot handle spatialdim !=1or2 yet.]");
//   }
//   // TODO more type and check
//   auto output_type = isFloat(input.dtype());

//   /* input = astype(input,out_type);
//   weight = astype(weight,out_type);*/

//   std::vector<int> output_shape = get_conv2d_output_shape(
//       input.shape(), weight.shape(), stride, padding, dilation);

//   const std::vector<int> stride_vec = {stride.first, stride.second};
//   const std::vector<int> padding_vec = {padding.first, padding.second};
//   const std::vector<int> dilation_vec = {dilation.first, dilation.second};

//   return Array(output_type, std::make_shared<ConvolutionPrimitive>(),
//                {input, weight}, output_shape,
//                getStridesFromShape(output_shape, dtypeSize(output_type)));
// }
// Activations

// ReLU
//  differentiability. relu is differentiable as it can be approximated and
//  hence the use of it in a loss function.
Array relu(const Array &input) {
  return Array(input.dtype(), std::make_shared<ReluPrimitive>(), {input},
               input.shape(), input.strides());
}

// utils
std::vector<int> get_conv2d_output_shape(const std::vector<int> &in_shape,
                                         const std::vector<int> &weight_shape,
                                         const std::pair<int, int> &stride,
                                         const std::pair<int, int> &padding,
                                         const std::pair<int, int> &dilation) {
  // 3 224 224 in channels h w
  // weight shape=(out_channels, *kernel_size, in_channels),
  // auto [N, C_in, H_in, W_in] = in_shape;
  // auto [C_out, KernelSizeH, KernelSizeW, C_in_] = weight_shape;
  // out [N, C_out, H_out, W_out]
  auto N = in_shape[0];
  auto H_in = in_shape[1];
  auto W_in = in_shape[2];
  auto C_in = in_shape[3];

  auto KernelSize0 = weight_shape[0];
  auto KernelSize1 = weight_shape[1];
  auto C_out = weight_shape[3];
  int H_out = static_cast<int>(
      (H_in + 2 * padding.first - dilation.first * (KernelSize0 - 1) - 1) /
          stride.first +
      1);
  int W_out = static_cast<int>(
      (W_in + 2 * padding.second - dilation.second * (KernelSize1 - 1) - 1) /
          stride.second +
      1);
  return {N, H_out, W_out, C_out};
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

std::vector<int> broadcastShapes(const std::vector<int> &shape1,
                                 const std::vector<int> &shape2) {
  const int ndim1 = shape1.size();
  const int ndim2 = shape2.size();
  const int maxDim = std::max(ndim1, ndim2);
  const int dimDiff = std::abs(ndim1 - ndim2);
  const std::vector<int> &largerShape = (ndim1 > ndim2) ? shape1 : shape2;
  const std::vector<int> &smallerShape = (ndim1 > ndim2) ? shape2 : shape1;
  std::vector<int> outputShape(maxDim);
  for (int i = maxDim - 1; i >= dimDiff; --i) {
    const int largerDim = largerShape[i];
    const int smallerDim = smallerShape[i - dimDiff];

    if (largerDim == smallerDim) {
      outputShape[i] = largerDim;
    } else if (largerDim == 1 || smallerDim == 1) {
      outputShape[i] = std::max(largerDim, smallerDim);
    } else {
      throw std::invalid_argument(
          "[Ops::broadcastShapeps] Shapes cannot be broadcast.");
    }
  }

  if (dimDiff > 0) {
    for (int i = dimDiff - 1; i >= 0; --i) {
      outputShape[i] = largerShape[i];
    }
  }
}

std::vector<Array> broadcastArrays(const std::vector<Array> &inputs) {
  std::vector<int> shape = inputs[0].shape();
  // for each shape to get final shape
  for (const Array &in : inputs) {
    shape = broadcastShapes(shape, in.shape());
  }
  std::vector<Array> outputs;
  for (const auto &in : inputs) {
    outputs.push_back(broadcast_to(in, shape));
  }
  return outputs;
}

Array getElementsNumber(const Array &arr, const std::vector<int> &axes,
                        bool inverted, Dtype dtype) {
  std::vector<int> normal_axes;
  for (auto &axis : axes) {
    // in [0,ndim-1]
    int normal_axis = (axis + arr.ndim()) % arr.ndim();
    if (normal_axis >= arr.ndim() || normal_axis < 0) {
      throw std::invalid_argument(
          "[OPS number_of_elements] Cannot get the shape for axis.");
    }
    normal_axes.push_back(normal_axis);
  }
  std::vector<int> output_shape{1};
  // TODO no stop grad
  return Array(arr.dtype(),
               std::make_shared<GetElementsNumberPrimitive>(normal_axes,
                                                            inverted, dtype),
               {arr},
               // Shape {1} is scalar
               output_shape,
               getStridesFromShape(output_shape, dtypeSize(arr.dtype())));
}

std::pair<std::vector<int>, std::vector<int>>
getReduceShape(const std::vector<int> &axes, const std::vector<int> &shape) {
  std::set<int> axes_set;
  int ndim = shape.size();
  for (auto axis : axes) {
    int axis_ = (axis < 0) ? axis + ndim : axis;
    if (axis < 0 || axis >= ndim) {
      throw std::invalid_argument("[GetReduceShape] given axes out of range.");
    }
    axes_set.insert(axis_);
  }

  std::vector<int> output_shape;
  for (int i = 0; i < ndim; i++) {
    if (axes_set.count(i) == 0) {
      output_shape.push_back(shape[i]);
    } else {
      output_shape.push_back(1);
    }
  }

  std::vector<int> sorted_axes(axes_set.begin(),
                               axes_set.end()); // increase

  return {output_shape, sorted_axes};
}

}; // namespace ainl::core