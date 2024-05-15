#pragma once

#include "array.h"
#include "transformation.h"
namespace ainl::core {

// mean var

Array zeros(const std::vector<int> &shape, Dtype dtype);
Array zeros_like(const Array &arr);
Array ones(const std::vector<int> &shape, Dtype dtype);
Array ones_like(const Array &arr);
Array eye(int n, int m, int k, Dtype dtype);
Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype);
inline Array fill(const std::vector<int> &shape, const Array &value) {
    return fill(shape, value, value.dtype());
}
Array slice(const Array &input, const std::vector<int> &start,
            const std::vector<int> &end, const std::vector<int> &stride);
Array reshape(const Array &input, const std::vector<int> &shape);
Array flatten(const Array &input);
Array astype(const Array &arr, Dtype dtype);
// arange
Array arange(double begin, double end, double stride, Dtype dtype);
Array arange(double begin, double end, Dtype dtype);
Array arange(double begin, double end);
Array astype(const Array &arr, Dtype dtype);
Array abs(const Array &arr);
Array add(const Array &a, const Array &b);
Array arcsin(const Array &arr);
Array arcsinh(const Array &arr);
Array arccos(const Array &arr);
Array arccosh(const Array &arr);
Array arctan(const Array &arr);
Array arctanh(const Array &arr);
Array sin(const Array &arr);
Array sinh(const Array &arr);
Array cosh(const Array &arr);
Array cos(const Array &arr);
Array tan(const Array &arr);
Array tanh(const Array &arr);
Array exp(const Array &arr);
Array log(const Array &arr);
Array multiply(const Array &a, const Array &b);
Array getElementsNumber(const Array &arr, const std::vector<int> &axes,
                        bool inverted, Dtype dtype);
Array mean(const Array &arr, bool keepdims = false);
inline Array mean(const Array &arr) { return mean(arr, false); }

Array mean(const Array &arr, std::vector<int> &axes, bool keepdims = false);
Array mean(const Array &arr, int axis, bool keepdims);

Array var(const Array &arr, bool keepdims = false);
inline Array var(const Array &arr) { return var(arr, false); }
Array var(const Array &arr, const std::vector<int> &axes,
          bool keepdims = false);
Array var(const Array &arr, int axis, bool keepdims = false);
Array softmax(const Array &arr);
Array sigmoid(const Array &arr);
// broadcast
Array broadcast_to(const Array &arr, const std::vector<int> &shape);
std::vector<int> broadcast_shapes(const std::vector<int> &s1,
                                  const std::vector<int> &s2);
std::vector<Array> broadcast_arrays(const std::vector<Array> &inputs);
Array squeeze(const Array &arr, const std::vector<int> &axes);
Array sum(const Array &arr, const std::vector<int> &axes, bool keepdims);

// convolution
Array conv2d(const Array &input, const Array &weight,
             const std::pair<int, int> &stride,
             const std::pair<int, int> &padding,
             const std::pair<int, int> &dilation);

std::vector<int> get_conv2d_output_shape(const std::vector<int> &in_shape,
                                         const std::vector<int> &weight_shape,
                                         const std::pair<int, int> &stride,
                                         const std::pair<int, int> &padding,
                                         const std::pair<int, int> &dilation);

#define GENERIC_OP_DECL(name)                                                  \
    std::shared_ptr<Tracer> name(                                              \
        const std::vector<std::shared_ptr<Tracer>> &inputs,                    \
        const std::shared_ptr<Primitive> &prim);

#define GENERIC_OP_IMPL(name)                                                  \
    std::shared_ptr<Tracer> name(                                              \
        const std::vector<std::shared_ptr<Tracer>> &inputs,                    \
        const std::shared_ptr<Primitive> &prim) {                              \
        return TracerFactory::createTracer(inputs, prim);                      \
    }

GENERIC_OP_DECL(reshape_)

std::vector<int> getStridesFromShape(const std::vector<int> &shape,
                                     size_t itemsize);

} // namespace ainl::core
