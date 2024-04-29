#include "ops.h"

namespace ainl::core {
Dtype isFloat(Dtype dtype) { return dtype < Float32 ? Float32 : dtype; }

Array zeros(const std::vector<int> &shape, Dtype dtype) {
    return fill(shape, Array(0, dtype), dtype);
}

Array zeros_like(const Array &arr) { return zeros(arr.shape(), arr.dtype()); }

Array ones(const std::vector<int> &shape, Dtype dtype) {
    return fill(shape, Array(1, dtype), dtype);
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
    auto in = broadcast_to(astype(value, dtype), shape);
    // return Array(Float, std::make_shared<FillPrimitive>(), {value});
    return Array(dtype, std::make_shared<FillPrimitive>(), {in}, shape,
                 in.strides());
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

Array add(const Array &a, const Array &b) {
    // Dtype output_type = a.dtype() < b.dtype() ? a.dtype() : b.dtype();
    // auto inputs =
    //     broadcast_arrays({astype(a, output_type), astype(b, output_type)});
    // std::vector<int> output_shape = inputs[0].shape();
    // return Array(output_type, std::make_shared<AddPrimitive>(),
    //              std::move(inputs), output_shape,
    //              getStridesFromShape(output_shape, dtypeSize(output_type)));
    Dtype output_type = a.dtype() < b.dtype() ? a.dtype() : b.dtype();
    std::vector<int> output_shape = a.shape();
    // std::move？

    return Array(output_type, std::make_shared<AddPrimitive>(), {a, b},
                 output_shape,
                 getStridesFromShape(output_shape, dtypeSize(output_type)));
}

// Convolution

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
    broadcast_shapes(arr.shape(), shape);
    return Array(arr.dtype(), std::make_shared<BroadCastPrimitive>(shape),
                 {arr}, std::move(shape),
                 getStridesFromShape(shape, arr.itemsize()));
}
std::vector<int> broadcast_shapes(const std::vector<int> &s1,
                                  const std::vector<int> &s2) {
    int ndim1 = s1.size();
    int ndim2 = s2.size();
    int max_dim = std::max(ndim1, ndim2);
    int diff = std::abs(ndim1 - ndim2);
    const std::vector<int> &large = ndim1 > ndim2 ? s1 : s2;
    const std::vector<int> &small = ndim1 > ndim2 ? s2 : s1;

    std::vector<int> output_shape(max_dim);
    for (int i = max_dim - 1; i > diff; i--) {
        int l = large[i];
        int s = small[i - diff];
        if (l == s)
            output_shape[i] = l;
        else if (l == 1 || s == 1)
            output_shape[i] = l * s;
        else
            throw std::invalid_argument("Shapes  cannot be broadcast.");
    }

    for (size_t i = diff - 1; i >= 0; --i) {
        output_shape[i] = large[i];
    }

    return output_shape;
}

std::vector<Array> broadcast_arrays(const std::vector<Array> &inputs) {
    std::vector<int> shape;
    for (const Array &in : inputs) {
        shape = broadcast_shapes(shape, in.shape());
    }
    std::vector<Array> outputs;
    for (const auto &in : inputs) {
        outputs.push_back(broadcast_to(in, shape));
    }
    return outputs;
}

GENERIC_OP_IMPL(reshape_)

}; // namespace ainl::core
