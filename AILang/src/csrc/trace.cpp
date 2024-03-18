#include "trace.h"
#include "primitive.h"

namespace ainl::core {

EvaluationTrace::EvaluationTrace() {}

void EvaluationTrace::pack(Array &array) {
}

void EvaluationTrace::unpack(Array &array) {
}

void EvaluationTrace::process(const std::shared_ptr<Primitive> &prim,
                              std::vector<Array> &inputs, Array &output) {
    for (auto &input : inputs) {
        // pack(input);
    }
    prim->eval(shared_from_this(), inputs, output);
    for (auto &input : inputs) {
        // unpack(input);
    }
}

void JITTrace::pack(Array &array) {
}

void JITTrace::unpack(Array &array) {
}

void JITTrace::process(const std::shared_ptr<Primitive> &prim,
                       std::vector<Array> &inputs, Array &output) {
    for (auto &input : inputs) {
        // pack(input);
    }
    prim->eval(shared_from_this(), inputs, output);
    for (auto &input : inputs) {
        // unpack(input);
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

std::shared_ptr<BaseTrace> getTopTrace() {
    return traceManager().getTopTrace();
}

} // namespace ainl::core