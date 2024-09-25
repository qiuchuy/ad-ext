#pragma once

#include "ailang/Core/Allocator.h"
#include "ailang/Core/Array.h"
#include "ailang/Core/Device.h"
#include "ailang/IR/Node.h"
#include "ailang/IR/Use.h"
#include "ailang/IR/Value.h"

#include <pybind11/pybind11.h>
#include <vector>

extern std::map<std::string, pybind11::function> eval_callback;

using namespace ainl::ir;

namespace ainl::core {

class Array;
class BaseTrace;
class Tracer;
class JVPTracer;
class JITTracer;

class Primitive {
public:
  Primitive() : device(cpu) {}

  const Device &getDevice() const { return device; }

  virtual ~Primitive() = default;
  Primitive(const Primitive &other) = delete;
  Primitive &operator=(const Primitive &other) = delete;
  Primitive &operator=(Primitive &&other) = delete;
  Primitive(Primitive &&other) = delete;

  virtual void eval(const std::vector<Array> &inputs,
                    std::vector<Array> &output) = 0;
  virtual void evalCPU(const std::vector<Array> &inputs,
                       std::vector<Array> &output) = 0;
  virtual void jit(const std::vector<JITTracer> &inputs,
                   std::vector<JITTracer> &output) = 0;
  virtual void jvp(const std::vector<JVPTracer> &inputs,
                   std::vector<JVPTracer> &output) = 0;

  virtual TypePtr inferType(const std::vector<TypePtr> &inputTypes) = 0;

  virtual std::string toString() const = 0;
  operator std::string() const { return toString(); }

  friend std::ostream &operator<<(std::ostream &os, const Primitive &prim) {
    os << prim.toString();
    return os;
  }

private:
  Device device;
};

class UnaryPrimitive : public Primitive {
public:
  UnaryPrimitive() = default;
  virtual void eval(const std::vector<Array> &inputs, Array &output) = 0;
  virtual void evalCPU(const std::vector<Array> &inputs, Array &output) = 0;
  virtual void jit(const std::vector<JITTracer> &inputs, JITTracer &output) = 0;
  virtual void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) = 0;

  void eval(const std::vector<Array> &inputs,
            std::vector<Array> &outputs) override;
  void evalCPU(const std::vector<Array> &inputs,
               std::vector<Array> &outputs) override;

  void jit(const std::vector<JITTracer> &inputs,
           std::vector<JITTracer> &outputs) override;

  void jvp(const std::vector<JVPTracer> &inputs,
           std::vector<JVPTracer> &outputs) override;
};

class IdentityPrimitive : public UnaryPrimitive {
public:
  IdentityPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};

class AddPrimitive : public UnaryPrimitive {
public:
  AddPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};

class FlattenPrimitive : public UnaryPrimitive {
public:
  FlattenPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};

class FillPrimitive : public UnaryPrimitive {
public:
  FillPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};

class SlicePrimitive : public UnaryPrimitive {
public:
  SlicePrimitive() = default;
  explicit SlicePrimitive(const std::vector<int> &begin,
                          const std::vector<int> &end)
      : begin_(begin), end_(end) {
    stride_ = std::vector<int>(begin.size(), 1);
  }
  explicit SlicePrimitive(const std::vector<int> &begin,
                          const std::vector<int> &end,
                          const std::vector<int> &stride)
      : begin_(begin), end_(end), stride_(stride) {}
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;

private:
  std::vector<int> begin_;
  std::vector<int> end_;
  std::vector<int> stride_;
};

class ReshapePrimitive : public UnaryPrimitive {
public:
  ReshapePrimitive() = default;
  explicit ReshapePrimitive(const std::vector<int> &shape) : shape_(shape) {}
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;

private:
  std::vector<int> shape_;
};

class TransposePrimitive : public UnaryPrimitive {
public:
  TransposePrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};

class MatMulPrimitive : public UnaryPrimitive {
public:
  MatMulPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};

class AsTypePrimitive : public UnaryPrimitive {
public:
  explicit AsTypePrimitive(Dtype dtype) : dtype_(dtype) {};
  AsTypePrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override {}
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;

private:
  Dtype dtype_;
};

// BroadcastPrimitive
class BroadcastPrimitive : public UnaryPrimitive {
public:
  BroadcastPrimitive() = default;
  explicit BroadcastPrimitive(const std::vector<int> &shape) : shape_(shape) {}
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;

private:
  std::vector<int> shape_;
  std::vector<int> broadcastShapes(const std::vector<int> &shape1,
                                   const std::vector<int> &shape2);
};

class MaximumPrimitive : public UnaryPrimitive {
public:
  MaximumPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override {}
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};
class MinimumPrimitive : public UnaryPrimitive {
public:
  MinimumPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override {}
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};

class MultiplyPrimitive : public UnaryPrimitive {
public:
  MultiplyPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;
};

class SubtractPrimitive : public UnaryPrimitive {
public:
  SubtractPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override {}
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};

class SquarePrimitive : public UnaryPrimitive {
public:
  SquarePrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override {}
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};

class ReducePrimitive : public UnaryPrimitive {
public:
  enum ReduceType { And, Or, Sum, Prod, Min, Max };
  explicit ReducePrimitive(const std::vector<int> &axes, ReduceType reduce_type)
      : axes_(axes), reduce_type_(reduce_type) {}
  ReducePrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override {}
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;

private:
  ReduceType reduce_type_;
  std::vector<int> axes_;
};
class GetElementsNumberPrimitive : public UnaryPrimitive {
public:
  explicit GetElementsNumberPrimitive(const std::vector<int> &axes,
                                      bool inverted, Dtype dtype)
      : axes_(axes), inverted_(inverted), dtype_(dtype) {}
  GetElementsNumberPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override {}
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;

private:
  std::vector<int> axes_;
  bool inverted_;
  Dtype dtype_;
};

class SigmoidPrimitive : public UnaryPrimitive {
public:
  SigmoidPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override {}
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};

class ConvolutionPrimitive : public UnaryPrimitive {
public:
  explicit ConvolutionPrimitive(const std::vector<int64_t> &window_strides,
                                const std::vector<int64_t> &lhsDilation,
                                const std::vector<int64_t> &rhsDilation,
                                const std::vector<int64_t> &padding_args,
                                const std::vector<int64_t> &window_reversal)
      : window_strides(window_strides), lhsDilation(lhsDilation),
        rhsDilation(rhsDilation), padding_args(padding_args),
        window_reversal(window_reversal) {};
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;

private:
  std::vector<int64_t> window_strides;
  std::vector<int64_t> lhsDilation;
  std::vector<int64_t> rhsDilation;
  std::vector<int64_t> padding_args;
  std::vector<int64_t> window_reversal;
};

class ReluPrimitive : public UnaryPrimitive {
public:
  ReluPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
};
class SqrtPrimitive : public UnaryPrimitive {
public:
  SqrtPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;
};
class MeanPrimitive : public UnaryPrimitive {
public:
  MeanPrimitive(const std::vector<int64_t> &dim) : dim(dim) {}
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;

private:
  std::vector<int64_t> dim;
};
class VariancePrimitive : public UnaryPrimitive {
public:
  VariancePrimitive(const std::vector<int64_t> &dim, const int ddof)
      : dim(dim), ddof(ddof) {}
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;

private:
  std::vector<int64_t> dim;
  int ddof;
};
class BatchnormInferencePrimitive : public UnaryPrimitive {
public:
  BatchnormInferencePrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;
};
class MaxPool2dPrimitive : public UnaryPrimitive {
public:
  MaxPool2dPrimitive(const std::vector<int64_t> &window_dimensions,
                     const std::vector<int64_t> &window_strides,
                     const std::vector<int64_t> &base_dilations,
                     const std::vector<int64_t> &window_dilations,
                     const std::vector<int64_t> &padding)
      : window_dimensions(window_dimensions), window_strides(window_strides),
        base_dilations(base_dilations), window_dilations(window_dilations),
        padding(padding) {}
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;

private:
  std::vector<int64_t> window_dimensions;
  std::vector<int64_t> window_strides;
  std::vector<int64_t> base_dilations;
  std::vector<int64_t> window_dilations;
  std::vector<int64_t> padding;
};
class AvgPool2dPrimitive : public UnaryPrimitive {
public:
  AvgPool2dPrimitive(const std::vector<int64_t> &window_dimensions,
                     const std::vector<int64_t> &window_strides,
                     const std::vector<int64_t> &base_dilations,
                     const std::vector<int64_t> &window_dilations,
                     const std::vector<int64_t> &padding)
      : window_dimensions(window_dimensions), window_strides(window_strides),
        base_dilations(base_dilations), window_dilations(window_dilations),
        padding(padding) {}
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;

private:
  std::vector<int64_t> window_dimensions;
  std::vector<int64_t> window_strides;
  std::vector<int64_t> base_dilations;
  std::vector<int64_t> window_dilations;
  std::vector<int64_t> padding;
};

class ComparePrimitive : public UnaryPrimitive {

public:
  explicit ComparePrimitive(ir::CompareOp::CompareType op) : op_(op) {}
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override {
    throw std::runtime_error("Not implemented");
  };
  std::string toString() const override;
  ir::CompareOp::CompareType op_;

private:
  template <typename T> void compare(Array &lhs, Array &rhs, Array &output) {
    // [yuqiuchu] replace with elementwise comparison
    auto lval = lhs.template item<T>();
    auto rval = rhs.template item<T>();
    bool result;
    switch (op_) {
    case ir::CompareOp::CompareType::EQ:
      result = lval == rval;
      break;
    case ir::CompareOp::CompareType::NE:
      result = lval != rval;
      break;
    case ir::CompareOp::CompareType::LT:
      result = lval < rval;
      break;
    case ir::CompareOp::CompareType::LE:
      result = lval <= rval;
      break;
    case ir::CompareOp::CompareType::GT:
      result = lval > rval;
      break;
    case ir::CompareOp::CompareType::GE:
      result = lval >= rval;
      break;
    }
    output = Array(result);
  }
};

class ConcatPrimitive : public UnaryPrimitive {
public:
  explicit ConcatPrimitive(int dim) : dim(dim) {}
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override {
    throw std::runtime_error("Not implemented");
  };
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;

private:
  int dim;
};

class ExpPrimitive : public UnaryPrimitive {
public:
  ExpPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override {
    throw std::runtime_error("Not implemented");
  };
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;
};

class TanhPrimitive : public UnaryPrimitive {
public:
  TanhPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override {
    throw std::runtime_error("Not implemented");
  };
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;
};

class DivPrimitive : public UnaryPrimitive {
public:
  DivPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override {
    throw std::runtime_error("Not implemented");
  };
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;
};

class NegPrimitive : public UnaryPrimitive {
public:
  NegPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &out) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override {
    throw std::runtime_error("Not implemented");
  };
  TypePtr inferType(const std::vector<TypePtr> &inputTypes) override;
  std::string toString() const override;
};

} // namespace ainl::core
