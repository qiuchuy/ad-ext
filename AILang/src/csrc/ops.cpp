#include "ops.h"
#include "dtype.h"
#include <set>

#include "primitive.h"

namespace ainl::core {
Dtype isFloat(Dtype dtype) { return dtype < Float32 ? Float32 : dtype; }

Array zeros(const std::vector<int> &shape, Dtype dtype = Float64) {
    return fill(shape, Array(0., dtype), dtype);
}

Array zeros_like(const Array &arr) { return zeros(arr.shape(), arr.dtype()); }

Array ones(const std::vector<int> &shape, Dtype dtype) {
    return fill(shape, Array(1., dtype), dtype);
}

Array ones_like(const Array &arr) { return ones(arr.shape(), arr.dtype()); }

// Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype) {
//     return Array(dtype, std::make_shared<FillPrimitive>(), {value}, shape,
//                  value.strides());
// }

Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype) {
    // shape check
    if (std::any_of(shape.begin(), shape.end(), [](int x) { return x < 0; })) {
        throw std::invalid_argument(
            "[ops.cpp:fill] Input array's dimension can't be negative. ");
    };
    return broadcast_to(value, shape);
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
                 outputShape,
                 getStridesFromShape(outputShape, input.itemsize()));
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
    return Array(input.dtype(),
                 std::make_shared<ReshapePrimitive>(flattenShape), {input},
                 flattenShape,
                 getStridesFromShape(flattenShape, input.itemsize()));
}
Array maximum(const Array &a, const Array &b) {
    auto output_type = a.dtype();
    auto inputs =
        broadcast_arrays({astype(a, output_type), astype(b, output_type)});

    std::vector<int> output_shape = inputs[0].shape();
    return Array(output_type, std::make_shared<MaximumPrimitive>(),
                 std::move(inputs), output_shape,
                 getStridesFromShape(output_shape, dtypeSize(output_type)));
}
Array minimum(const Array &a, const Array &b) {
    auto output_type = a.dtype();
    auto inputs =
        broadcast_arrays({astype(a, output_type), astype(b, output_type)});

    std::vector<int> output_shape = inputs[0].shape();
    return Array(output_type, std::make_shared<MinimumPrimitive>(),
                 std::move(inputs), output_shape,
                 getStridesFromShape(output_shape, dtypeSize(output_type)));
}
Array arange(double begin, double end, double stride, Dtype dtype) {
    if (std::isnan(begin) || std::isnan(end) || std::isnan(stride)) {
        std::invalid_argument("[arange] input is NaN, can't compute length.");
    }
    if (std::isinf(begin) || std::isinf(end) || std::isinf(stride)) {
        std::invalid_argument("[arange] input is Inf, can't compute length.");
    }
    double real_size = std::ceil((end - begin) / (stride));
    if (std::isinf(real_size))
        std::invalid_argument("[arange] exceed the max length.");

    int size = std::max(static_cast<int>(real_size), 0);

    return Array(dtype, std::make_shared<ArangePrimitive>(begin, end, stride),
                 {}, {size}, getStridesFromShape({size}, dtypeSize(dtype)));
};

Array arange(double begin, double end, double stride) {
    return arange(begin, end, stride, Float32);
}
Array arange(double begin, double end) {
    return arange(begin, end, 1, Float32);
}

Array abs(const Array &arr) {
    Array out =
        Array(arr.dtype(), std::make_shared<AbsPrimitive>(), {arr}, arr.shape(),
              getStridesFromShape(arr.shape(), arr.itemsize()));
    return out;
}

Array astype(const Array &arr, Dtype dtype) {
    if (arr.dtype() == dtype)
        return arr;
    return Array(dtype, std::make_shared<AsTypePrimitive>(dtype), {arr},
                 arr.shape(), getStridesFromShape(arr.shape(), arr.itemsize()));
}
Array arcsin(const Array &arr) {
    // depends on Dtype实现
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<ArcSinPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array arcsinh(const Array &arr) {
    // depends on Dtype实现
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<ArcSinhPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array arccos(const Array &arr) {
    // depends on Dtype实现
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<ArcCosPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array arccosh(const Array &arr) {
    // depends on Dtype实现
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<ArcCoshPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array arctan(const Array &arr) {
    // depends on Dtype实现
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<ArcTanPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array arcTanh(const Array &arr) {
    // depends on Dtype实现
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<ArcTanhPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array sin(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<SinPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}
Array multiply(const Array &a, const Array &b) {
    // promote types
    auto output_type = a.dtype();
    // auto inputs = broadcast_arrays({a, b});
    // std::cout<<inputs[0]<<inputs[1];
    return Array(output_type, std::make_shared<MultiplyPrimitive>(), {a, b},
                 a.shape(), a.strides());
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
            throw std::invalid_argument(
                "[GetReduceShape] given axes out of range.");
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

Array sum(const Array &arr, const std::vector<int> &axes, bool keepdims) {
    if (axes.empty()) {
        return arr;
    }
    auto [output_shape, sorted_axes] = getReduceShape(axes, arr.shape());
    auto output_type = arr.dtype();
    auto out = Array(
        output_type,
        std::make_shared<ReducePrimitive>(sorted_axes, ReducePrimitive::Sum),
        {arr}, output_shape,
        getStridesFromShape(output_shape, dtypeSize(output_type)));
    if (keepdims) {
        return out;
    } else {
        return squeeze(out, sorted_axes);
    }
}

Array squeeze(const Array &arr, const std::vector<int> &axes) {
    std::set<int> axes_set;
    int ndim = arr.ndim();
    for (auto axis : axes) {
        auto ax_ = axis < 0 ? axis + ndim : axis;
        if (ax_ < 0 || ax_ > ndim) {
            throw std::invalid_argument(
                "[Ops Squeeze] given axes out of range");
        }
        // must specific axis = 1
        if (arr.shape()[ax_] != 1) {
            throw std::invalid_argument(
                "[Ops Squeeze] cannot deal with axis == 1.");
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
            shape.push_back(arr.shape()[i]);
        }
    }
    for (auto it : shape) {
        std::cout << it << std::endl;
    }
    // maybe shape = {}
    if (!shape.size()) {
        shape.push_back(1);
    }
    return reshape(arr, shape);
}

//  / n-1
Array square(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<SquarePrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}
Array sqrt(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<SqrtPrimitive>(false), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}
Array rsqrt(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<SqrtPrimitive>(true), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array var(const Array &arr, const std::vector<int> &axes, bool keepdims,
          int ddof) {
    int ndim = arr.ndim();

    for (int axis : axes) {
        if (axis < -ndim || axis >= ndim) {
            throw std::invalid_argument(
                "[Ops.cpp var] axis is out of bounds for array");
        }
    }
    if (ddof != 0) {
        throw std::invalid_argument("[Ops.cpp var] only suport ddof =0.");
    }
    auto output_type = arr.dtype();
    auto m2 = square(mean(arr, axes, keepdims));
    auto a2 = mean(square(arr), axes, keepdims);

    auto v = subtract(a2, m2);
    return v;
}

Array var(const Array &arr, bool keepdims) {
    std::vector<int> axes(arr.ndim());

    std::iota(axes.begin(), axes.end(), 0);
    return var(arr, axes, keepdims, 0);
}
Array var(const Array &arr, const std::vector<int> &axes, bool keepdims) {
    return var(arr, axes, keepdims, 0);
}
Array var(const Array &arr, int axis, bool keepdims) {
    return var(arr, {axis}, keepdims, 0);
}

Array mean(const Array &arr, bool keepdims) {
    std::vector<int> axes(arr.ndim());
    std::iota(axes.begin(), axes.end(), 0);
    return mean(arr, axes, keepdims);
}
Array mean(const Array &arr, const std::vector<int> &axes, bool keepdims) {
    int ndim = arr.ndim();
    for (int axis : axes) {
        if (axis < -ndim || axis >= ndim) {
            throw std::invalid_argument(
                "[Ops.cpp mean] axis is out of bounds for array");
        }
    }
    auto output_type = arr.dtype();
    // sum
    auto normalizer = getElementsNumber(arr, axes, true, output_type);
    return multiply(sum(arr, axes, keepdims), normalizer);
}

Array mean(const Array &arr, int axis, bool keepdims = false) {
    return mean(arr, {axis}, keepdims);
}

Array sinh(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<SinhPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}
Array cos(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<CosPrimitive>(), {arr},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array cosh(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<CoshPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}
Array tan(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<TanPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array tanh(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<TanhPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array log(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type,
                 std::make_shared<LogPrimitive>(LogPrimitive::LogBase::e),
                 {input}, arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array exp(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<ExpPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array softmax(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<SoftmaxPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}

Array sigmoid(const Array &arr) {
    Dtype output_type = isFloat(arr.dtype());
    auto input = astype(arr, output_type);
    return Array(output_type, std::make_shared<SigmoidPrimitive>(), {input},
                 arr.shape(),
                 getStridesFromShape(arr.shape(), dtypeSize(output_type)));
}
Array subtract(const Array &a, const Array &b) {
    Dtype output_type = a.dtype() < b.dtype() ? a.dtype() : b.dtype();
    auto inputs =
        broadcast_arrays({astype(a, output_type), astype(b, output_type)});

    std::vector<int> output_shape = inputs[0].shape();
    return Array(output_type, std::make_shared<SubtractPrimitive>(),
                 std::move(inputs), output_shape,
                 getStridesFromShape(output_shape, dtypeSize(output_type)));
}

Array add(const Array &a, const Array &b) {
    Dtype output_type = a.dtype() < b.dtype() ? a.dtype() : b.dtype();
    auto inputs =
        broadcast_arrays({astype(a, output_type), astype(b, output_type)});

    std::vector<int> output_shape = inputs[0].shape();
    return Array(output_type, std::make_shared<AddPrimitive>(),
                 std::move(inputs), output_shape,
                 getStridesFromShape(output_shape, dtypeSize(output_type)));
}
// Convolution

Array conv2d(const Array &input, const Array &weight,
             const std::pair<int, int> &stride /* {1, 1}*/,
             const std::pair<int, int> &padding /* {0, 0}*/,
             const std::pair<int, int> &dilation /*{1, 1}*/) {
    int spatial_dims = input.ndim() - 2;
    if (spatial_dims < 1 || spatial_dims > 2) {
        throw std::invalid_argument(
            "[ops.h Conv2d cannot handle spatialdim !=1or2 yet.]");
    }
    // TODO more type and check
    auto output_type = isFloat(input.dtype());

    /* input = astype(input,out_type);
    weight = astype(weight,out_type);*/

    std::vector<int> output_shape = get_conv2d_output_shape(
        input.shape(), weight.shape(), stride, padding, dilation);

    const std::vector<int> stride_vec = {stride.first, stride.second};
    const std::vector<int> padding_vec = {padding.first, padding.second};
    const std::vector<int> dilation_vec = {dilation.first, dilation.second};

    return Array(output_type,
                 std::make_shared<ConvolutionPrimitive>(stride_vec, padding_vec,
                                                        dilation_vec),
                 {input, weight}, output_shape,
                 getStridesFromShape(output_shape, dtypeSize(output_type)));
}

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
    auto C_in = in_shape[1];
    auto H_in = in_shape[2];
    auto W_in = in_shape[3];
    auto C_out = weight_shape[0];
    auto KernelSize0 = weight_shape[1];
    auto KernelSize1 = weight_shape[2];

    int H_out = static_cast<int>(
        (H_in + 2 * padding.first - dilation.first * (KernelSize0 - 1) - 1) /
            stride.first +
        1);
    int W_out = static_cast<int>(
        (W_in + 2 * padding.second - dilation.second * (KernelSize1 - 1) - 1) /
            stride.second +
        1);

    return {N, C_out, H_out, W_out};
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

Array broadcast_to(const Array &arr, const std::vector<int> &shape) {
    if (arr.shape() == shape) {
        return arr;
    }
    // check is broadcastable
    auto outputShape = broadcast_shapes(arr.shape(), shape);
    return Array(arr.dtype(), std::make_shared<BroadCastPrimitive>(outputShape),
                 {arr}, outputShape,
                 getStridesFromShape(outputShape, arr.itemsize()));
}
std::vector<int> broadcast_shapes(const std::vector<int> &s1,
                                  const std::vector<int> &s2) {

    int ndim1 = s1.size();                // 3
    int ndim2 = s2.size();                // 3
    int max_dim = std::max(ndim1, ndim2); // 3
    int diff = std::abs(ndim1 - ndim2);   // 0
    const std::vector<int> &large = ndim1 > ndim2 ? s1 : s2;
    const std::vector<int> &small = ndim1 > ndim2 ? s2 : s1;
    // 3 3
    std::vector<int> output_shape(max_dim);     // 3
    for (int i = max_dim - 1; i >= diff; i--) { // 2 0
        int l = large[i];                       // 2
        int s = small[i - diff];                // 2-0
        if (l == s)
            output_shape[i] = l;
        else if (l == 1 || s == 1) {
            output_shape[i] = l * s;
        } else
            throw std::invalid_argument("Shapes  cannot be broadcast.");
    }
    if (diff > 0) {
        for (int i = diff - 1; i >= 0; --i) {
            output_shape[i] = large[i];
        }
    }

    return output_shape;
}

std::vector<Array> broadcast_arrays(const std::vector<Array> &inputs) {
    std::vector<int> shape = inputs[0].shape();
    for (const Array &in : inputs) {
        shape = broadcast_shapes(shape, in.shape());
    }
    std::vector<Array> outputs;
    for (const auto &in : inputs) {
        outputs.push_back(broadcast_to(in, shape));
    }
    return outputs;
}

GENERIC_OP_IMPL(reshape)
GENERIC_OP_IMPL(transpose)  
GENERIC_OP_IMPL(matmul)

}; // namespace ainl::core
