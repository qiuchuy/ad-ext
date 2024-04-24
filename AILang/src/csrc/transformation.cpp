#include "transformation.h"

#include <memory>

#include "ir/function.h"
#include "ir/node.h"
#include "ir/type.h"
#include "trace.h"

namespace ainl::core {

bool JVPTracer::evaluated() const {
  return primal_ != nullptr && tangent_ != nullptr;
}

std::shared_ptr<Tracer> JVPTracer::aval() {
  if (primal_ == nullptr) {
    throw std::runtime_error("Primal is not set");
  }
  return primal_->aval();
}

ir::TypePtr JVPTracer::getJITType() {
  if (primal_ == nullptr) {
    throw std::runtime_error("Primal is not set");
  }
  return primal_->getJITType();
}

std::string JVPTracer::toString() const { return "jvptracer"; }

void JVPTrace::pack(std::vector<std::shared_ptr<Tracer>> &inputs) {
  /*
  for (auto &input : inputs) {
  auto array = std::dynamic_pointer_cast<Array>(input);
  if (array) {
    input = std::make_shared<JVPTracer>(
        input,
        std::make_shared<Array>(
            fill(array->shape(), Array(1.0f, array->dtype()), array->dtype())));
  }
  }
  */
}

void JVPTrace::unpack(std::vector<std::shared_ptr<Tracer>> &inputs) {}
void JVPTrace::process(const std::shared_ptr<Primitive> &prim,
                       const std::vector<std::shared_ptr<Tracer>> &inputs,
                       std::shared_ptr<Tracer> &output) {
  auto arrays = tracerVectorConversion<JVPTracer, Tracer>(inputs);

  if (auto primOutput = std::dynamic_pointer_cast<JVPTracer>(output)) {
    prim->jvp(arrays, *primOutput);
  } else {
    throw std::runtime_error("[jvp] Output is not an jvp tracer");
  }
}

std::string JVPTrace::toString() const { return "jvp"; }

std::shared_ptr<Tracer>
jvp(std::function<std::shared_ptr<Tracer>(std::vector<std::shared_ptr<Tracer>>)>
        f,
    std::vector<std::shared_ptr<Tracer>> primals,
    std::vector<std::shared_ptr<Tracer>> tangents) {
  if (primals.size() != tangents.size()) {
    throw std::runtime_error("Number of primals and tangents must match");
  }
  pushTrace(std::make_shared<JVPTrace>());

  size_t inputSize = primals.size();

  std::vector<std::shared_ptr<Tracer>> jvpTracers;
  for (size_t i = 0; i < inputSize; i++) {
    jvpTracers.push_back(std::make_shared<JVPTracer>(primals[i], tangents[i]));
  }
  auto result = f(jvpTracers);
  result->eval();
  popLastTrace();
  return result;
}

bool JITTracer::evaluated() const { return tracer_ != nullptr; }

ir::TypePtr JITTracer::getJITType() {
  if (tracer_ == nullptr) {
    throw std::runtime_error(
        "Trying to get the jit type of an empty tracer inside JITTracer");
  }
  return tracer_->getJITType();
}

std::shared_ptr<Tracer> JITTracer::aval() {
  if (tracer_ == nullptr) {
    throw std::runtime_error(
        "Trying to get the aval of an empty tracer inside JITTracer");
  }
  return tracer_->aval();
}

std::string JITTracer::toString() const { return "jittracer"; }

void JITTrace::pack(std::vector<std::shared_ptr<Tracer>> &inputs) {
  /*
  for (auto &input : inputs) {
  auto array = std::dynamic_pointer_cast<Array>(input);
  if (array) {
    input = std::make_shared<JITTracer>(input);
  }
  }
  */
}

void JITTrace::unpack(std::vector<std::shared_ptr<Tracer>> &inputs) {}

void JITTrace::process(const std::shared_ptr<Primitive> &prim,
                       const std::vector<std::shared_ptr<Tracer>> &inputs,
                       std::shared_ptr<Tracer> &output) {
  auto arrays = tracerVectorConversion<JITTracer, Tracer>(inputs);

  if (auto primOutput = std::dynamic_pointer_cast<JITTracer>(output)) {
    prim->jit(arrays, *primOutput);
  } else {
    throw std::runtime_error("[jit] Output is not an jit tracer");
  }
}

std::string JITTrace::toString() const { return "jit"; }

ir::ModulePtr
jit(std::function<std::shared_ptr<Tracer>(std::vector<std::shared_ptr<Tracer>>)>
        f,
    std::string funcName, const std::vector<std::shared_ptr<Tracer>> &inputs) {
  std::vector<ir::TypePtr> types;
  for (auto &input : inputs) {
    types.push_back(input->getJITType());
  }
  auto argType = ir::TupleType::createUnnamedTuple(types);
  auto module = ir::ALModule::create(funcName, argType);
  auto params = module->getParams();

  pushTrace(std::make_shared<JITTrace>(module));
  std::vector<std::shared_ptr<Tracer>> jittracers;

  for (size_t i = 0; i < params.size(); i++) {
    auto jittracer = std::make_shared<JITTracer>(inputs[i], params[i]);
    jittracers.push_back(jittracer);
  }

  auto result = std::dynamic_pointer_cast<JITTracer>(f(jittracers));
  result->eval();
  popLastTrace();
  module->setReturnType(result->value()->getType());
  module->getGraph()->create<ir::ReturnOp>(
      std::dynamic_pointer_cast<JITTracer>(result)->value());
  return module;
}

} // namespace ainl::core