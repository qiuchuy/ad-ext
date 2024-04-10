#include "transformation.h"
#include "ir/type.h"
#include "ops.h"
#include "trace.h"

namespace ainl::core {

std::vector<std::shared_ptr<Tracer>> JVPTracer::subtracers() const {
  return {primal_, tangent_};
}

bool JVPTracer::evaluated() const {
  return primal_->evaluated() && tangent_->evaluated();
}

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
  popLastTrace();
  return result;
}

} // namespace ainl::core