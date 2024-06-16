#pragma once

#include "array.h"
#include "ir/function.h"
#include "ir/type.h"
#include "primitive.h"
#include "trace.h"
#include <stdexcept>

namespace ainl::core {

class JVPTrace : public BaseTrace {
public:
  JVPTrace(int level) : BaseTrace(level) {}
  void pack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void unpack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void process(const std::shared_ptr<Primitive> &prim,
               const std::vector<std::shared_ptr<Tracer>> &inputs,
               const std::vector<std::shared_ptr<Tracer>> &output);
  std::string toString() const override;

private:
  void update(const std::vector<std::shared_ptr<Tracer>> &inputs,
              const std::vector<JVPTracer> &output);
};

class JITTrace : public BaseTrace {
public:
  JITTrace(const ir::ModulePtr &module, int level)
      : BaseTrace(level), module_(module) {}
  void pack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void unpack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void process(const std::shared_ptr<Primitive> &prim,
               const std::vector<std::shared_ptr<Tracer>> &inputs,
               const std::vector<std::shared_ptr<Tracer>> &output);
  std::string toString() const override;
  ir::ModulePtr module() { return module_; }

private:
  ir::ModulePtr module_;
  void update(const std::vector<std::shared_ptr<Tracer>> &inputs,
              const std::vector<JITTracer> &output);
};

class JITTracer : public Tracer {
public:
  JITTracer(const std::shared_ptr<Tracer> &tracer, const ir::ValuePtr &value)
      : Tracer({}, nullptr), tracer_(tracer), value_(value) {}
  JITTracer(const std::vector<std::shared_ptr<Tracer>> &inputs,
            const std::shared_ptr<Primitive> &prim)
      : Tracer(inputs, prim) {}
  bool evaluated() const override;
  Tracer::TracerTy getTracerTy() const override {
    return TracerTy::JITTracerTy;
  }
  ir::TypePtr getJITType() override;
  std::shared_ptr<Tracer> aval() override;
  std::string toString() const override;
  std::shared_ptr<Tracer> tracer() const { return tracer_; }
  void setTracer(const std::shared_ptr<Tracer> &tracer) { tracer_ = tracer; }
  void setValue(const ir::ValuePtr &value) { value_ = value; }
  ir::ValuePtr value() const { return value_; }
  std::shared_ptr<Tracer> clone() override {
    return std::make_shared<JITTracer>(*this);
  }

private:
  std::shared_ptr<Tracer> tracer_;
  ir::ValuePtr value_;
};

class JVPTracer : public Tracer {
public:
  JVPTracer(const std::vector<std::shared_ptr<Tracer>> &inputs,
            const std::shared_ptr<Primitive> &prim)
      : Tracer(inputs, prim) {}
  JVPTracer(const std::shared_ptr<Tracer> &primal,
            const std::shared_ptr<Tracer> &tangent)
      : Tracer({}, nullptr), primal_(primal), tangent_(tangent) {}
  bool evaluated() const override;
  Tracer::TracerTy getTracerTy() const override {
    return TracerTy::JVPTracerTy;
  }
  ir::TypePtr getJITType() override;
  std::shared_ptr<Tracer> aval() override;
  std::string toString() const override;
  std::shared_ptr<Tracer> primal() { return primal_; }
  std::shared_ptr<Tracer> tangent() { return tangent_; }
  void setPrimal(const std::shared_ptr<Tracer> &primal) { primal_ = primal; }
  void setTangent(const std::shared_ptr<Tracer> &tangent) {
    tangent_ = tangent;
  }
  std::shared_ptr<Tracer> clone() override {
    return std::make_shared<JVPTracer>(*this);
  }

private:
  std::shared_ptr<Tracer> primal_;
  std::shared_ptr<Tracer> tangent_;
};

std::shared_ptr<Tracer>
jvp(std::function<std::shared_ptr<Tracer>(std::vector<std::shared_ptr<Tracer>>)>
        f,
    std::vector<std::shared_ptr<Tracer>> primals,
    std::vector<std::shared_ptr<Tracer>> tangents);

ir::ModulePtr jit(std::function<std::vector<std::shared_ptr<Tracer>>(
                      std::vector<std::shared_ptr<Tracer>>)>
                      f,
                  std::string funcName, std::string target,
                  const std::vector<std::shared_ptr<Tracer>> &inputs);

void eval(const std::vector<std::shared_ptr<Tracer>> &inputs);

} // namespace ainl::core