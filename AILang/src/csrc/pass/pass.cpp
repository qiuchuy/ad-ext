#include "pass/stablehlo_lowering.h"
#include "utils/logger.h"

namespace ainl::ir {

void Pipeline::registerPass(const std::string &name,
                            std::unique_ptr<Pass> pass) {
  passes[name] = std::move(pass);
}

void Pipeline::run(ModulePtr module) {
  for (auto &pass : passes) {
    LOG_DEBUG("[pass] Running pass: %s\n", pass.first);
    pass.second->run(module);
  }
  passes.clear();
}

Pipeline &pipeline() {
  static Pipeline pipeline;
  return pipeline;
}

void pipelineRegisterPass(const std::string &name, std::unique_ptr<Pass> pass) {
  pipeline().registerPass(name, std::move(pass));
}

void runPipelineOnModule(ModulePtr module) { pipeline().run(module); }



} // namespace ainl::ir