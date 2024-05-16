#pragma once

#include "array.h"
#include "backend/common/compute.h"
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
    virtual void jvp(const std::vector<JVPTracer> &inputs,
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
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;
};

class AddPrimitive : public Primitive {
  public:
    AddPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &output) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;
};

class SubtractPrimitive : public Primitive {
  public:
    SubtractPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &output) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;
};

class SquarePrimitive : public Primitive {
  public:
    SquarePrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &output) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;
};

class SqrtPrimitive : public Primitive {
  public:
    SqrtPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &output) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;
};
class FlattenPrimitive : public Primitive {
  public:
    FlattenPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &output) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;
};

class FillPrimitive : public Primitive {
  public:
    FillPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &output) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
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
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
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
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;

  private:
    std::vector<int> shape_;
};

// Abs
class AbsPrimitive : public Primitive {
  public:
    AbsPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &output) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;
};

class AddMMPrimitive : public Primitive {
  public:
    AddMMPrimitive() = default;
    explicit AddMMPrimitive(float alpha, float beta)
        : alpha_(alpha), beta_(beta){};
    void eval(const std::vector<Array> &inputs, Array &output) override;
    void evalCPU(const std::vector<Array> &inputs, Array &outputs) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;

  private:
    float alpha_;
    float beta_;
};

// Arange
class ArangePrimitive : public Primitive {

  public:
    ArangePrimitive() = default;
    explicit ArangePrimitive(double start, double end)
        : start_(start), end_(end){};
    explicit ArangePrimitive(double start, double end, double stride)
        : start_(start), end_(end), stride_(stride){};
    void eval(const std::vector<Array> &inputs, Array &output) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;

  private:
    double start_;
    double end_;
    double stride_;
};

// ArcCos
class ArcCosPrimitive : public Primitive {
  public:
    ArcCosPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    std::string toString() const override;
};
class ArcCoshPrimitive : public Primitive {
  public:
    ArcCoshPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    std::string toString() const override;
};
class ArcSinPrimitive : public Primitive {
  public:
    ArcSinPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    std::string toString() const override;
};
class ArcSinhPrimitive : public Primitive {
  public:
    ArcSinhPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};
class ArcTanhPrimitive : public Primitive {
  public:
    ArcTanhPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class ArcTanPrimitive : public Primitive {
  public:
    ArcTanPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class AsTypePrimitive : public Primitive {
  public:
    explicit AsTypePrimitive(Dtype dtype) : dtype_(dtype){};
    AsTypePrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;

  private:
    Dtype dtype_;
};
// BroadCast skip
class BroadCastPrimitive : public Primitive {
  public:
    BroadCastPrimitive() = default;
    explicit BroadCastPrimitive(const std::vector<int> &shape)
        : shape_(shape) {}
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    std::string toString() const override;

  private:
    std::vector<int> shape_;
};

class CeilPrimitive : public Primitive {
  public:
    CeilPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};
// cancatenate skip

class CopyPrimitive : public Primitive {
  public:
    CopyPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    std::string toString() const override;
};

class CosPrimitive : public Primitive {
  public:
    CosPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};
class CoshPrimitive : public Primitive {
  public:
    CoshPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class DividePrimitive : public Primitive {
  public:
    DividePrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;

    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    std::string toString() const override;
};

class ExpPrimitive : public Primitive {
  public:
    ExpPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    std::string toString() const override;
};

class FloorPrimitive : public Primitive {
  public:
    FloorPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class LogPrimitive : public Primitive {
  public:
    enum LogBase { two, ten, e };
    explicit LogPrimitive(LogBase base) : base_(base){};
    LogPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    std::string toString() const override;

  private:
    LogBase base_;
};
class MatmulPrimitive : public Primitive {
  public:
    MatmulPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class MaximumPrimitive : public Primitive {
  public:
    MaximumPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class MinimumPrimitive : public Primitive {
  public:
    MinimumPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class SigmoidPrimitive : public Primitive {
  public:
    SigmoidPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    std::string toString() const override;
};

class SinPrimitive : public Primitive {
  public:
    SinPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};
class SinhPrimitive : public Primitive {
  public:
    SinhPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class SoftmaxPrimitive : public Primitive {
  public:
    SoftmaxPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class TanhPrimitive : public Primitive {
  public:
    TanhPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class TanPrimitive : public Primitive {
  public:
    TanPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    std::string toString() const override;
};

class TransposePrimitive : public Primitive {
  public:
    explicit TransposePrimitive(std::vector<int> &axes) : axes_(axes){};
    TransposePrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;

    std::string toString() const override;

  private:
    std::vector<int> axes_;
};
class MultiplyPrimitive : public Primitive {
  public:
    MultiplyPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;
};

class GetElementsNumberPrimitive : public Primitive {
  public:
    explicit GetElementsNumberPrimitive(const std::vector<int> &axes,
                                        bool inverted, Dtype dtype)
        : axes_(axes), inverted_(inverted), dtype_(dtype) {}
    GetElementsNumberPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;

  private:
    std::vector<int> axes_;
    bool inverted_;
    Dtype dtype_;
};

// MeanPrimitive
class MeanPrimitive : public Primitive {
  public:
    explicit MeanPrimitive(const std::vector<int> &axes, bool keepdims)
        : axes_(axes), keepdims_(keepdims) {}
    MeanPrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;

  private:
    std::vector<int> axes_;
    bool keepdims_;
};
class ReducePrimitive : public Primitive {
  public:
    enum ReduceType { And, Or, Sum, Prod, Min, Max };
    explicit ReducePrimitive(const std::vector<int> &axes,
                             ReduceType reduce_type)
        : axes_(axes), reduce_type_(reduce_type) {}
    ReducePrimitive() = default;
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;

  private:
    ReduceType reduce_type_;
    std::vector<int> axes_;
};

class ConvolutionPrimitive : public Primitive {
  public:
    explicit ConvolutionPrimitive(const std::vector<int> &stride,
                                  const std::vector<int> &padding,
                                  const std::vector<int> &dilation)
        : stride_(stride), padding_(padding), dilation_(dilation){};
    void eval(const std::vector<Array> &inputs, Array &out) override;
    void evalCPU(const std::vector<Array> &inputs, Array &output) override;
    TypePtr typeRalation(const std::vector<TypePtr> &inTypes) override;
    void jvp(const std::vector<JVPTracer> &inputs, JVPTracer &output) override;
    std::string toString() const override;

  private:
    std::vector<int> stride_;
    std::vector<int> padding_;
    std::vector<int> dilation_;

}; // namespace ainl::core
} // namespace ainl::core