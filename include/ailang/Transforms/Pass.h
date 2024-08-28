#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"

#include "ailang/IR/Function.h"

namespace ainl::ir {

class Pass {
public:
  Pass() = default;
  Pass(const Pass &other) = delete;
  Pass(Pass &&other) = delete;
  Pass &operator=(const Pass &other) = delete;
  Pass &operator=(Pass &&other) = delete;
  virtual ~Pass() = default;
  virtual void run(ModulePtr module) = 0;
};

class Pipeline {
public:
  Pipeline() = default;
  Pipeline(const Pipeline &other) = delete;
  Pipeline(Pipeline &&other) = delete;
  Pipeline &operator=(const Pipeline &other) = delete;
  Pipeline &operator=(Pipeline &&other) = delete;
  void registerPass(const std::string &name, std::unique_ptr<Pass> pass);
  void run(ModulePtr module);

private:
  std::unordered_map<std::string, std::unique_ptr<Pass>> passes;
};

Pipeline &pipeline();
void pipelineRegisterPass(const std::string &name, std::unique_ptr<Pass> pass);
void runPipelineOnModule(ModulePtr module);

} // namespace ainl::ir