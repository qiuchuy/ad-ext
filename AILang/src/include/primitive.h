#pragma once

#include "array.h"
#include "device.h"

class Type;
using TypePtr = std::shared_ptr<Type>;

namespace ainl::core {

class Array;
class BaseTrace;

class Primitive {
public:
  Primitive() : device(cpu) {}

  const Device &getDevice() const { return device; }

  virtual ~Primitive() = default;
  Primitive(const Primitive &other) = delete;
  Primitive &operator=(const Primitive &other) = delete;
  Primitive &operator=(Primitive &&other) = delete;
  Primitive(Primitive &&other) = delete;

  virtual void eval(const std::shared_ptr<BaseTrace> &trace,
                    const std::vector<Array> &inputs, Array &output) = 0;
  virtual void evalCPU(const std::vector<Array> &inputs, Array &output) = 0;

  virtual TypePtr typeRalation(const std::vector<TypePtr> &inTypes) = 0;

private:
  Device device;
};

class AddPrimitive : public Primitive {
public:
  void eval(const std::shared_ptr<BaseTrace> &trace,
            const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
};

class FillPrimitive : public Primitive {
public:
  void eval(const std::shared_ptr<BaseTrace> &trace,
            const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
};

class SlicePrimitive : public Primitive {
public:
  explicit SlicePrimitive(int begin, int end)
      : begin_(begin), end_(end), stride_(1) {}
  explicit SlicePrimitive(int begin, int end, int stride)
      : begin_(begin), end_(end), stride_(stride) {}
  void eval(const std::shared_ptr<BaseTrace> &trace,
            const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;
  TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;

private:
  int begin_;
  int end_;
  int stride_;
};

class ReshapePrimitive : public Primitive {
public:
  explicit ReshapePrimitive(const std::vector<int> &shape) : shape(shape) {}
  void eval(const std::shared_ptr<BaseTrace> &trace,
            const std::vector<Array> &inputs, Array &output) override;
  void evalCPU(const std::vector<Array> &inputs, Array &output) override;

  TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;

private:
  std::vector<int> shape;
};

} // namespace ainl::core
