#pragma once

#include "ailang/Core/Array.h"
#include "ailang/Core/Trace.h"
#include "ailang/Core/Transformation.h"

namespace ainl::core {

Array zeros(const std::vector<int> &shape, Dtype dtype);
Array ones(const std::vector<int> &shape, Dtype dtype);
Array fill(const std::vector<int> &shape, const Array &value, Dtype dtype);
Array slice(const Array &input, const std::vector<int> &start,
            const std::vector<int> &end, const std::vector<int> &stride);
Array reshape(const Array &input, const std::vector<int> &shape);
Array transpose(const Array &input);
Array matmul(const Array &lhs, const Array &rhs);
Array flatten(const Array &input);

class TracerBuilder {
public:
  static TracerBuilder &getInstance() {
    static TracerBuilder instance;
    return instance;
  }

  TracerBuilder(const TracerBuilder &) = delete;
  TracerBuilder(TracerBuilder &&) = delete;
  TracerBuilder &operator=(const TracerBuilder &) = delete;
  TracerBuilder &operator=(TracerBuilder &&) = delete;

  template <typename PrimTy, typename... Args>
  std::vector<std::shared_ptr<Tracer>>
  build(const std::vector<std::shared_ptr<Tracer>> &inputs, unsigned output_num,
        Args... args) {
    auto trace_mode = findTopTrace(inputs)->mode;
    auto tracers = createTracers<PrimTy>(trace_mode, inputs, output_num,
                                         std::forward<Args>(args)...);
    setupSiblings(tracers);
    assert(!tracers.empty() && "Expect at least one output tracer.");
    return tracers;
  }

private:
  TracerBuilder() { initializeBuilders(); }
  ~TracerBuilder() = default;

  template <typename PrimTy, typename... Args>
  using TracerCreator = std::function<std::shared_ptr<Tracer>(
      const std::shared_ptr<Tracer> &, Args...)>;

  std::unordered_map<BaseTrace::TraceMode,
                     std::function<std::shared_ptr<Tracer>(
                         const std::vector<std::shared_ptr<Tracer>> &,
                         const std::shared_ptr<Primitive> &)>>
      tracerBuilders;

  void initializeBuilders() {
    tracerBuilders[BaseTrace::TraceMode::eval] =
        [](const std::vector<std::shared_ptr<Tracer>> &inputs,
           const std::shared_ptr<Primitive> &prim) {
          return std::make_shared<Array>(inputs, prim);
        };

    tracerBuilders[BaseTrace::TraceMode::jit] =
        [](const std::vector<std::shared_ptr<Tracer>> &inputs,
           const std::shared_ptr<Primitive> &prim) {
          return std::make_shared<JITTracer>(inputs, prim);
        };

    tracerBuilders[BaseTrace::TraceMode::jvp] =
        [](const std::vector<std::shared_ptr<Tracer>> &inputs,
           const std::shared_ptr<Primitive> &prim) {
          return std::make_shared<JVPTracer>(inputs, prim);
        };
  }

  template <typename PrimTy, typename... Args>
  std::vector<std::shared_ptr<Tracer>>
  createTracers(BaseTrace::TraceMode mode,
                const std::vector<std::shared_ptr<Tracer>> &inputs,
                unsigned output_num, Args... args) {
    std::vector<std::shared_ptr<Tracer>> tracers;
    tracers.reserve(output_num);

    auto prim = std::make_shared<PrimTy>(std::forward<Args>(args)...);
    auto builder = tracerBuilders[mode];

    for (unsigned i = 0; i < output_num; i++) {
      tracers.push_back(
          builder(inputs, std::static_pointer_cast<Primitive>(prim)));
    }

    return tracers;
  }

  void setupSiblings(std::vector<std::shared_ptr<Tracer>> &tracers) {
    for (size_t i = 0; i < tracers.size(); i++) {
      auto siblings = tracers;
      siblings.erase(siblings.begin() + i);
      tracers[i]->setSiblings(siblings);
      tracers[i]->setIdx(i);
    }
  }
};

template <typename PrimTy, typename... Args>
std::vector<std::shared_ptr<Tracer>>
op(const std::vector<std::shared_ptr<Tracer>> &inputs, unsigned output_num,
   Args &&... args) {
  auto &builder = TracerBuilder::getInstance();
  return builder.build<PrimTy>(inputs, output_num, std::forward<Args>(args)...);
}

template <typename PrimTy, typename... Args>
std::shared_ptr<Tracer>
single(const std::vector<std::shared_ptr<Tracer>> &inputs, Args &&... args) {
  return op<PrimTy>(inputs, 1, std::forward<Args>(args)...)[0];
}

std::vector<int> getStridesFromShape(const std::vector<int> &shape,
                                     size_t itemsize);

Array add(const Array &lhs, const Array &rhs);
Array astype(const Array &input, Dtype dtype);
Array maximum(const Array &lhs, const Array &rhs);
Array minimum(const Array &lhs, const Array &rhs);
Array multiply(const Array &rhs, const Array &lhs);
Array sigmoid(const Array &input);
Array subtract(const Array &lhs, const Array &rhs);
Array sum(const Array &input, const std::vector<int> &axes, bool keepdims);
Array squeeze(const Array &input, const std::vector<int> &axes);
Array square(const Array &input);
Array sqrt(const Array &input);
Array rsqrt(const Array &input);
Array var(const Array &input, const std::vector<int> &axes, bool keepdims,
          int ddof);
Array var(const Array &input, bool keepdims);
Array var(const Array &input, const std::vector<int> &axes, bool keepdims);
Array var(const Array &input, int axis, bool keepdims);
Array mean(const Array &input, bool keepdims);
Array mean(const Array &input, const std::vector<int> &axes, bool keepdims);
Array mean(const Array &input, int axis, bool keepdims = false);
Array flatten(const Array &input);
Array broadcast_to(const Array &input, const std::vector<int> &shape);
Array conv2d(const Array &input, const Array &weight,
             const std::pair<int, int> &stride,
             const std::pair<int, int> &padding,
             const std::pair<int, int> &dilation);
Array relu(const Array &input);

std::vector<int> get_conv2d_output_shape(const std::vector<int> &in_shape,
                                         const std::vector<int> &weight_shape,
                                         const std::pair<int, int> &stride,
                                         const std::pair<int, int> &padding,
                                         const std::pair<int, int> &dilation);
std::vector<int> getStridesFromShape(const std::vector<int> &shape,
                                     size_t itemsize);
std::vector<int> broadcastShapes(const std::vector<int> &shape1,
                                 const std::vector<int> &shape2);
std::vector<Array> broadcastArrays(const std::vector<Array> &inputs);
Array getElementsNumber(const Array &arr, const std::vector<int> &axes,
                        bool inverted, Dtype dtype);
std::pair<std::vector<int>, std::vector<int>>
getReduceShape(const std::vector<int> &axes, const std::vector<int> &shape);

} // namespace ainl::core
