#include "ir/type.h"

#include "trace.h"
#include "transformation.h"

namespace ainl::core {

void JVPTrace::pack(std::vector<std::shared_ptr<Tracer>> &inputs) {}
void JVPTrace::unpack(std::vector<std::shared_ptr<Tracer>> &inputs) {}
void JVPTrace::process(const std::shared_ptr<Primitive> &prim,
                       const std::vector<std::shared_ptr<Tracer>> &inputs,
                       std::shared_ptr<Tracer> &output) {}

} // namespace ainl::core