#include "ailang/Core/Ops.h"
#include "ailang/Core/Primitive.h"
#include "ailang/Core/Trace.h"
#include "ailang/Utils/Logger.h"

#include <initializer_list>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace {

using namespace ainl::core;

class PyTracerBuilder {
public:
  static PyTracerBuilder &getInstance() {
    static PyTracerBuilder instance;
    return instance;
  }
  PyTracerBuilder() = default;
  virtual ~PyTracerBuilder() = default;
  PyTracerBuilder(const PyTracerBuilder &) = delete;
  PyTracerBuilder(PyTracerBuilder &&) = delete;
  template <typename PrimTy, typename... Args>
  py::object build(const std::vector<std::shared_ptr<Tracer>> &inputs,
                   unsigned output_size, Args... args) {
    auto &builder = TracerBuilder::getInstance();
    auto tracers = builder.build<PrimTy>(inputs, output_size, args...);
    return py::cast(tracers);
  }
};

template <typename PrimTy, typename... Args>
py::object pyop(const std::vector<std::shared_ptr<Tracer>> &inputs,
                unsigned output_num, Args &&...args) {
  auto &builder = PyTracerBuilder::getInstance();
  return builder.build<PrimTy>(inputs, output_num, std::forward<Args>(args)...);
}

template <typename PrimTy, typename... Args>
py::object pyunary(const std::vector<std::shared_ptr<Tracer>> &inputs,
                   Args &&...args) {
  auto result = pyop<PrimTy>(inputs, 1, std::forward<Args>(args)...);
  if (py::isinstance<py::list>(result)) {
    return py::cast<py::list>(result)[0];
  } else if (py::isinstance<py::tuple>(result)) {
    return py::cast<py::tuple>(result)[0];
  } else {
    throw std::invalid_argument(
        "Expect a list or tuple of tracers as output in unary operators");
  }
}

} // anonymous namespace

void init_ailang_op(py::module_ &m) {
  m.def("flatten", [](const std::shared_ptr<Tracer> &input) {
    return pyunary<FlattenPrimitive>({input});
  });

  m.def("reshape", [](const std::shared_ptr<Tracer> &input,
                      const std::vector<int> &shape) {
    return pyunary<ReshapePrimitive>({input}, shape);
  });

  m.def("slice",
        [](const std::shared_ptr<Tracer> &input, const std::vector<int> &start,
           const std::vector<int> &end, const std::vector<int> &stride) {
          return pyunary<SlicePrimitive>({input}, start, end, stride);
        });

  m.def("transpose", [](const std::shared_ptr<Tracer> &input) {
    return pyunary<TransposePrimitive>({input});
  });

  m.def("matmul", [](const std::shared_ptr<Tracer> &lhs,
                     const std::shared_ptr<Tracer> &rhs) {
    return pyunary<MatMulPrimitive>({lhs, rhs});
  });

  m.def("add", [](const std::shared_ptr<ainl::core::Tracer> &lhs,
                  const std::shared_ptr<ainl::core::Tracer> &rhs) {
    return pyunary<ainl::core::AddPrimitive>({lhs, rhs});
    return pyunary<ainl::core::AddPrimitive>({lhs, rhs});
  });
  m.def("conv2d", [](const std::shared_ptr<ainl::core::Tracer> &inputValue,
                     const std::shared_ptr<ainl::core::Tracer> &weightValue,
                     const std::vector<int64_t> &window_stride,
                     const std::vector<int64_t> &lhs_dilation,
                     const std::vector<int64_t> &rhs_dilation,
                     const std::vector<int64_t> &padding_args,
                     const std::vector<int64_t> &window_reversal) {
    return pyunary<ainl::core::ConvolutionPrimitive>(
        {inputValue, weightValue}, window_stride, lhs_dilation, rhs_dilation,
        padding_args, window_reversal);
  });

  m.def("relu", [](const std::shared_ptr<ainl::core::Tracer> &input) {
    return pyunary<ainl::core::ReluPrimitive>({input});
  });
  m.def("batchnorm2d", [](const std::shared_ptr<ainl::core::Tracer> &input,
                          const std::shared_ptr<ainl::core::Tracer> &scale,
                          const std::shared_ptr<ainl::core::Tracer> &offset,
                          const std::shared_ptr<ainl::core::Tracer> &mean,
                          const std::shared_ptr<ainl::core::Tracer> &variance) {
    return pyunary<ainl::core::BatchnormInferencePrimitive>(
        {input, scale, offset, mean, variance});
  });

  m.def("mean", [](const std::shared_ptr<ainl::core::Tracer> &input,
                   const std::vector<int64_t> &dim) {
    return pyunary<ainl::core::MeanPrimitive>({input}, dim);
  });
  m.def("cat",
        [](const std::vector<std::shared_ptr<ainl::core::Tracer>> &inputs,
           int dim) {
          return pyunary<ainl::core::ConcatPrimitive>(inputs, dim);
        });
  m.def("exp", [](const std::shared_ptr<ainl::core::Tracer> &input) {
    return pyunary<ainl::core::ExpPrimitive>({input});
  });
  m.def("tanh", [](const std::shared_ptr<ainl::core::Tracer> &input) {
    return pyunary<ainl::core::TanhPrimitive>({input});
  });
  m.def("neg", [](const std::shared_ptr<ainl::core::Tracer> &input) {
    return pyunary<ainl::core::NegPrimitive>({input});
  });
  m.def("div", [](const std::shared_ptr<ainl::core::Tracer> &lhs,
                  const std::shared_ptr<ainl::core::Tracer> &rhs) {
    return pyunary<ainl::core::DivPrimitive>({lhs, rhs});
  });
  m.def("mul", [](const std::shared_ptr<ainl::core::Tracer> &lhs,
                  const std::shared_ptr<ainl::core::Tracer> &rhs) {
    return pyunary<ainl::core::MultiplyPrimitive>({lhs, rhs});
  });
  m.def("broadcast_to", [](const std::shared_ptr<ainl::core::Tracer> &input,
                           const std::vector<int> &shape) {
    return pyunary<ainl::core::BroadcastPrimitive>({input}, shape);
  });
  m.def("maxpool2d", [](const std::shared_ptr<ainl::core::Tracer> &inputValue,
                        const std::vector<int64_t> &window_dimensions,
                        const std::vector<int64_t> &window_strides,
                        const std::vector<int64_t> &base_dilations,
                        const std::vector<int64_t> &window_dilations,
                        const std::vector<int64_t> &padding) {
    return pyunary<ainl::core::MaxPool2dPrimitive>(
        {inputValue}, window_dimensions, window_strides, base_dilations,
        window_dilations, padding);
  });
  m.def("avgpool2d", [](const std::shared_ptr<ainl::core::Tracer> &inputValue,
                        const std::vector<int64_t> &window_dimensions,
                        const std::vector<int64_t> &window_strides,
                        const std::vector<int64_t> &base_dilations,
                        const std::vector<int64_t> &window_dilations,
                        const std::vector<int64_t> &padding) {
    return pyunary<ainl::core::AvgPool2dPrimitive>(
        {inputValue}, window_dimensions, window_strides, base_dilations,
        window_dilations, padding);
  });
  m.def("var", [](const std::shared_ptr<ainl::core::Tracer> &input,
                  const std::vector<int64_t> &dim) {
    return pyunary<ainl::core::VariancePrimitive>({input}, dim);
  });
}