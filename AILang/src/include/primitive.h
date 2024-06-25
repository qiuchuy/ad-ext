#pragma once

#include "allocator.h"
#include "array.h"
#include "device.h"
#include "ir/node.h"

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
  std::string toString() const override;
};

class AddPrimitive : public UnaryPrimitive {
public:
  AddPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  std::string toString() const override;
};

class FlattenPrimitive : public UnaryPrimitive {
public:
  FlattenPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  std::string toString() const override;
};

class FillPrimitive : public UnaryPrimitive {
public:
  FillPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
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
  std::string toString() const override;
};

class MatMulPrimitive : public UnaryPrimitive {
public:
  MatMulPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
  std::string toString() const override;
};

class LoopPrimitive : public Primitive {
public:
  LoopPrimitive() = default;
  LoopPrimitive(const std::function<std::shared_ptr<Tracer>(
                    const std::vector<std::shared_ptr<Tracer>> &)> &cond,
                const std::function<std::vector<std::shared_ptr<Tracer>>(
                    const std::vector<std::shared_ptr<Tracer>> &)> &body)
      : cond_(cond), body_(body) {}
  void eval(const std::vector<Array> &inputs,
            std::vector<Array> &output) override;
  void evalCPU(const std::vector<Array> &inputs,
               std::vector<Array> &output) override;
  void jit(const std::vector<JITTracer> &inputs,
           std::vector<JITTracer> &output) override;
  void jvp(const std::vector<JVPTracer> &inputs,
           std::vector<JVPTracer> &output) override;
  std::string toString() const override;

private:
  std::function<std::shared_ptr<Tracer>(
      const std::vector<std::shared_ptr<Tracer>> &)>
      cond_;
  std::function<std::vector<std::shared_ptr<Tracer>>(
      const std::vector<std::shared_ptr<Tracer>> &)>
      body_;
};

class IfPrimitive : public Primitive {
public:
  IfPrimitive() = default;
  IfPrimitive(
      const std::function<std::vector<std::shared_ptr<Tracer>>()> &trueBranch,
      const std::function<std::vector<std::shared_ptr<Tracer>>()> &falseBranch)
      : trueBranch(trueBranch), falseBranch(falseBranch) {}
  void eval(const std::vector<Array> &inputs,
            std::vector<Array> &output) override;
  void evalCPU(const std::vector<Array> &inputs,
               std::vector<Array> &output) override;
  void jit(const std::vector<JITTracer> &inputs,
           std::vector<JITTracer> &output) override;
  void jvp(const std::vector<JVPTracer> &inputs,
           std::vector<JVPTracer> &output) override;
  std::string toString() const override;

private:
  std::function<std::vector<std::shared_ptr<Tracer>>()> trueBranch;
  std::function<std::vector<std::shared_ptr<Tracer>>()> falseBranch;
};

class ComparePrimitive : public UnaryPrimitive {

public:
  explicit ComparePrimitive(ir::CompareOp::CompareType op) : op_(op) {}
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  void jit(const std::vector<JITTracer> &inputs, JITTracer &output) override;
  void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
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

} // namespace ainl::core
