#pragma once

#include "array.h"
#include "ir/function.h"
#include "ir/type.h"
#include "primitive.h"
#include "trace.h"

namespace ainl::core {

class JVPTrace : public BaseTrace {
public:
  JVPTrace() = default;
  void pack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void unpack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void process(const std::shared_ptr<Primitive> &prim,
               const std::vector<std::shared_ptr<Tracer>> &inputs,
               std::shared_ptr<Tracer> &output);
  std::string toString() const override;
};

class JITTrace : public BaseTrace {
public:
  JITTrace(const ir::ModulePtr &module) : module_(module) {}
  void pack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void unpack(std::vector<std::shared_ptr<Tracer>> &inputs);
  void process(const std::shared_ptr<Primitive> &prim,
               const std::vector<std::shared_ptr<Tracer>> &inputs,
               std::shared_ptr<Tracer> &output);
  std::string toString() const override;
  ir::ModulePtr module() { return module_; }

private:
  ir::ModulePtr module_;
};

class JITTracer : public Tracer {
public:
  JITTracer(const std::shared_ptr<Tracer> &tracer, const ir::ValuePtr &value)
      : tracer_(tracer), value_(value) {}
  JITTracer(const std::vector<std::shared_ptr<Tracer>> &inputs,
            const std::shared_ptr<Primitive> &prim)
      : Tracer(inputs, prim) {}
  bool evaluated() const override;
  ir::TypePtr getJITType() override;
  std::shared_ptr<Tracer> aval() override;
  std::string toString() const override;
  std::shared_ptr<Tracer> tracer() { return tracer_; }
  void setTracer(const std::shared_ptr<Tracer> &tracer) { tracer_ = tracer; }
  void setValue(const ir::ValuePtr &value) { value_ = value; }
  ir::ValuePtr value() { return value_; }

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
      : primal_(primal), tangent_(tangent) {}
  bool evaluated() const override;
  ir::TypePtr getJITType() override;
  std::shared_ptr<Tracer> aval() override;
  std::string toString() const override;
  std::shared_ptr<Tracer> primal() { return primal_; }
  std::shared_ptr<Tracer> tangent() { return tangent_; }
  void setPrimal(const std::shared_ptr<Tracer> &primal) { primal_ = primal; }
  void setTangent(const std::shared_ptr<Tracer> &tangent) {
    tangent_ = tangent;
  }

private:
  std::shared_ptr<Tracer> primal_;
  std::shared_ptr<Tracer> tangent_;
};

class TracerFactory {
public:
  static std::shared_ptr<Tracer>
  createTracer(const std::vector<std::shared_ptr<Tracer>> &inputs,
               const std::shared_ptr<Primitive> &prim) {
    if (std::all_of(inputs.begin(), inputs.end(), [](const auto &input) {
          return std::dynamic_pointer_cast<JVPTracer>(input) != nullptr;
        })) {
      return std::make_shared<JVPTracer>(inputs, prim);
    }
    if (std::all_of(inputs.begin(), inputs.end(), [](const auto &input) {
          return std::dynamic_pointer_cast<Array>(input) != nullptr;
        })) {
      return std::make_shared<Array>(inputs, prim);
    }
  }
};

std::shared_ptr<Tracer>
jvp(std::function<std::shared_ptr<Tracer>(std::vector<std::shared_ptr<Tracer>>)>
        f,
    std::vector<std::shared_ptr<Tracer>> primals,
    std::vector<std::shared_ptr<Tracer>> tangents);

ir::ModulePtr
jit(std::function<std::shared_ptr<Tracer>(std::vector<std::shared_ptr<Tracer>>)>
        f,
    std::string funcName, const std::vector<std::shared_ptr<Tracer>> &inputs);

} // namespace ainl::core