#pragma once
#include "array.h"
#include "ir/function.h"

namespace ainl::core {

class JVPTrace : public BaseTrace {
public:
  JVPTrace();
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

private:
  std::shared_ptr<Tracer> primal_;
  std::shared_ptr<Tracer> tangent_;
};

} // namespace ainl::core