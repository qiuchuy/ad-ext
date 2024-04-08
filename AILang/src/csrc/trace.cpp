#include "trace.h"
#include "primitive.h"

namespace ainl::core {

EvaluationTrace::EvaluationTrace() {}

void EvaluationTrace::pack(std::vector<std::shared_ptr<Tracer>> &inputs) {}

void EvaluationTrace::unpack(std::vector<std::shared_ptr<Tracer>> &inputs) {}

void EvaluationTrace::process(
    const std::shared_ptr<Primitive> &prim,
    const std::vector<std::shared_ptr<Tracer>> &inputs,
    std::shared_ptr<Tracer> &output) {
  auto arrayPointers = tracerAsArrays(inputs);
  std::vector<Array> arrays;
  for (auto &arrayPointer : arrayPointers) {
    arrays.push_back(*arrayPointer);
  }
  auto primOutput = std::dynamic_pointer_cast<Array>(output);
  prim->evalCPU(arrays, *primOutput);
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

std::shared_ptr<BaseTrace> getCurrentTrace() {
  return traceManager().getCurrentTrace();
}

bool hasRemainingTrace() { return traceManager().hasRemainingTrace(); }

} // namespace ainl::core