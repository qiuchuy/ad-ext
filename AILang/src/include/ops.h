#pragma once
#include "array.h"
#include "transformation.h"
#include <set>

namespace ainl::core {

Dtype isFloat(Dtype dtype);
Array zeros(const std::vector<int> &shape, Dtype dtype);
Array ones(const std::vector<int> &shape, Dtype dtype);
Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype);
Array slice(const Array &input, const std::vector<int> &start,
            const std::vector<int> &end, const std::vector<int> &stride);
Array astype(const Array &arr, Dtype dtype);
Array reshape(const Array &input, const std::vector<int> &shape);
Array transpose(const Array &input);
Array matmul(const Array &lhs, const Array &rhs);
Array flatten(const Array &input);
Array add(const Array &lhs, const Array &rhs);
Array maximum(const Array &a, const Array &b);
Array minimum(const Array &a, const Array &b);
Array multiply(const Array &a, const Array &b);
Array sigmoid(const Array &arr);
Array subtract(const Array &a, const Array &b);
Array exp(const Array &arr);
Array sum(const Array &arr, const std::vector<int> &axes, bool keepdims);
Array squeeze(const Array &arr, const std::vector<int> &axes);
Array broadcast_to(const Array &input, const std::vector<int> &shape);
Array square(const Array &arr);
Array sqrt(const Array &arr);
Array rsqrt(const Array &arr);
Array var(const Array &arr, const std::vector<int> &axes, bool keepdims,
          int ddof);
Array var(const Array &arr, bool keepdims);
Array var(const Array &arr, const std::vector<int> &axes, bool keepdims);
Array var(const Array &arr, int axis, bool keepdims);
Array mean(const Array &arr, bool keepdims);
Array mean(const Array &arr, const std::vector<int> &axes, bool keepdims);
Array mean(const Array &arr, int axis, bool keepdims);
Array flatten(const Array &input);
Array conv2d(const Array &input, const Array &weight,
             const std::pair<int, int> &stride,
             const std::pair<int, int> &padding,
             const std::pair<int, int> &dilation);
std::vector<int> get_conv2d_output_shape(const std::vector<int> &in_shape,
                                         const std::vector<int> &weight_shape,
                                         const std::pair<int, int> &stride,
                                         const std::pair<int, int> &padding,
                                         const std::pair<int, int> &dilation);
std::vector<int> broadcastShapes(const std::vector<int> &shape1,
                                 const std::vector<int> &shape2);
std::vector<Array> broadcastArrays(const std::vector<Array> &inputs);
std::pair<std::vector<int>, std::vector<int>>
getReduceShape(const std::vector<int> &axes, const std::vector<int> &shape);
Array getElementsNumber(const Array &arr, const std::vector<int> &axes,
                        bool inverted, Dtype dtype);

template <typename PrimTy, typename... Args>
std::shared_ptr<Tracer>
unary(const std::vector<std::shared_ptr<Tracer>> &inputs, Args &&...args) {
    assert(!inputs.empty());
    auto promotedInputs = inputs;
    // [todo] debug this
    // ainl::core::getCurrentTrace()->pack(promotedInputs);
    auto tracerType = promotedInputs[0]->getTracerTy();
    switch (tracerType) {
    case ainl::core::Tracer::TracerTy::ArrayTy:
        return (std::make_shared<ainl::core::Array>(
            promotedInputs,
            std::make_shared<PrimTy>(std::forward<Args>(args)...)));
    case ainl::core::Tracer::TracerTy::JVPTracerTy:
        return (std::make_shared<ainl::core::JVPTracer>(
            promotedInputs,
            std::make_shared<PrimTy>(std::forward<Args>(args)...)));
    case ainl::core::Tracer::TracerTy::JITTracerTy:
        return (JITTracer::create(
            promotedInputs,
            std::make_shared<PrimTy>(std::forward<Args>(args)...)));
    default:
        throw std::runtime_error("Unsupported tracer type in op unary.");
    }
}

template <typename PrimTy, typename... Args>
std::vector<std::shared_ptr<Tracer>>
op(const std::vector<std::shared_ptr<Tracer>> &inputs, Args &&...args) {
    auto promotedInputs = inputs;
    // ainl::core::getCurrentTrace()->pack(promotedInputs);
    auto tracerType = promotedInputs[0]->getTracerTy();
    std::vector<std::shared_ptr<ainl::core::Tracer>> tracers;
    for (const auto &input : inputs) {
        switch (tracerType) {
        case ainl::core::Tracer::TracerTy::ArrayTy:
            tracers.push_back(std::make_shared<ainl::core::Array>(
                promotedInputs,
                std::make_shared<PrimTy>(std::forward<Args>(args)...)));
            break;
        case ainl::core::Tracer::TracerTy::JVPTracerTy:
            tracers.push_back(std::make_shared<ainl::core::JVPTracer>(
                promotedInputs,
                std::make_shared<PrimTy>(std::forward<Args>(args)...)));
            break;
        case ainl::core::Tracer::TracerTy::JITTracerTy:
            tracers.push_back(JITTracer::create(
                promotedInputs,
                std::make_shared<PrimTy>(std::forward<Args>(args)...)));
            break;
        default:
            throw std::runtime_error("Unsupported tracer type in op prim.");
        }
    }
    for (size_t i = 0; i < tracers.size(); i++) {
        auto siblings = tracers;
        siblings.erase(siblings.begin() + i);
        tracers[i]->setSiblings(siblings);
        tracers[i]->setIdx(i);
    }

    return tracers;
}

std::vector<int> getStridesFromShape(const std::vector<int> &shape,
                                     size_t itemsize);

} // namespace ainl::core
