#include "trace.h"
#include "primitive.h"

#include "utils/logger.h"

namespace ainl::core {

EvaluationTrace::EvaluationTrace() {}

void EvaluationTrace::pack(std::vector<std::shared_ptr<Tracer>> &inputs) {}

void EvaluationTrace::unpack(std::vector<std::shared_ptr<Tracer>> &inputs) {}

void EvaluationTrace::process(
    const std::shared_ptr<Primitive> &prim,
    const std::vector<std::shared_ptr<Tracer>> &inputs,
    std::shared_ptr<Tracer> &output) {
  auto arrays = tracerVectorConversion<Array, Tracer>(inputs);
  if (auto primOutput = std::dynamic_pointer_cast<Array>(output)) {
    prim->eval(arrays, *primOutput);
  } else {
    throw std::runtime_error("[eval] Output is not an array.");
  }
}

TraceManager::TraceManager() {
  auto evalTrace = std::make_shared<EvaluationTrace>();
  traceStack.push(std::dynamic_pointer_cast<BaseTrace>(evalTrace));
}

TraceManager &traceManager() {
  static TraceManager manager;
  return manager;
}

std::shared_ptr<BaseTrace> popLastTrace() {
  return traceManager().popLastTrace();
}

void pushTrace(std::shared_ptr<BaseTrace> trace) {
  traceManager().pushTrace(std::move(trace));
}

std::shared_ptr<BaseTrace> getCurrentTrace() {
  return traceManager().getCurrentTrace();
}

bool hasRemainingTrace() { return traceManager().hasRemainingTrace(); }

} // namespace ainl::core