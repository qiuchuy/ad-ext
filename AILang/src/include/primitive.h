#pragma once

#include "array.h"
#include "device.h"

class Type;
using TypePtr = std::shared_ptr<Type>;

namespace ainl::core {

class Array;
class BaseTrace;
class Tracer;
class JVPTracer;

class Primitive {
public:
  Primitive() : device(cpu) {}

  const Device &getDevice() const { return device; }

  virtual ~Primitive() = default;
  Primitive(const Primitive &other) = delete;
  Primitive &operator=(const Primitive &other) = delete;
  Primitive &operator=(Primitive &&other) = delete;
  Primitive(Primitive &&other) = delete;

  virtual void eval(const std::vector<Array> &inputs, Array &output) = 0;
  virtual void evalCPU(const std::vector<Array> &inputs, Array &output) = 0;
  virtual TypePtr typeRalation(const std::vector<TypePtr> &inTypes) = 0;
  virtual JVPTracer jvp(const std::vector<JVPTracer> &inputs,
                        JVPTracer &output) = 0;
  virtual std::string toString() const = 0;
  operator std::string() const { return toString(); }

  friend std::ostream &operator<<(std::ostream &os, const Primitive &prim) {
    os << prim.toString();
    return os;
  }

private:
  Device device;
};

class IdentityPrimitive : public Primitive {
public:
  IdentityPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
  JVPTracer jvp(const std::vector<JVPTracer> &inputs,
                JVPTracer &output) override;
  std::string toString() const override;
};

class AddPrimitive : public Primitive {
public:
  AddPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
  JVPTracer jvp(const std::vector<JVPTracer> &inputs,
                JVPTracer &output) override;
  std::string toString() const override;
};

class FlattenPrimitive : public Primitive {
public:
  FlattenPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
  JVPTracer jvp(const std::vector<JVPTracer> &inputs,
                JVPTracer &output) override;
  std::string toString() const override;
};

class FillPrimitive : public Primitive {
public:
  FillPrimitive() = default;
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
  JVPTracer jvp(const std::vector<JVPTracer> &inputs,
                JVPTracer &output) override;
  std::string toString() const override;
};

class SlicePrimitive : public Primitive {
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
  TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
  JVPTracer jvp(const std::vector<JVPTracer> &inputs,
                JVPTracer &output) override;
  std::string toString() const override;

private:
  std::vector<int> begin_;
  std::vector<int> end_;
  std::vector<int> stride_;
};

class ReshapePrimitive : public Primitive {
public:
  ReshapePrimitive() = default;
  explicit ReshapePrimitive(const std::vector<int> &shape) : shape_(shape) {}
  void eval(const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
  JVPTracer jvp(const std::vector<JVPTracer> &inputs,
                JVPTracer &output) override;
  std::string toString() const override;

private:
  std::vector<int> shape_;
};

} // namespace ainl::core
