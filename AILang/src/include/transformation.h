#pragma once
#include "array.h"
#include "ir/function.h"

namespace ainl::core {

class JVPTrace : public BaseTrace {
public:
  JVPTrace() = default;
  virtual void pack(std::vector<std::shared_ptr<Tracer>> &inputs);
  virtual void unpack(std::vector<std::shared_ptr<Tracer>> &inputs);
  virtual void process(const std::shared_ptr<Primitive> &prim,
                       const std::vector<std::shared_ptr<Tracer>> &inputs,
                       std::shared_ptr<Tracer> &output);
};

class JVPTracer : public Tracer {

public:
  JVPTracer(const std::shared_ptr<Tracer> &primal,
            const std::shared_ptr<Tracer> &tangent)
      : primal_(primal), tangent_(tangent) {}
  std::vector<std::shared_ptr<Tracer>> subtracers() const override;
  virtual bool evaluated() const override;
  std::shared_ptr<Tracer> primal() { return primal_; }
  std::shared_ptr<Tracer> tangent() { return tangent_; }
  void setPrimal(std::shared_ptr<Tracer> primal) { primal_ = primal; }
  void setTangent(std::shared_ptr<Tracer> tangent) { tangent_ = tangent; }

private:
  std::shared_ptr<Tracer> primal_;
  std::shared_ptr<Tracer> tangent_;
};

std::shared_ptr<Tracer>
jvp(std::function<std::shared_ptr<Tracer>(std::vector<std::shared_ptr<Tracer>>)>
        f,
    std::vector<std::shared_ptr<Tracer>> primals,
    std::vector<std::shared_ptr<Tracer>> tangents);

} // namespace ainl::core