#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"

#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>

#include "pass/stablehlo_lowering.h"


using namespace mlir;

namespace ainl::ir {

StableHLOLoweringPass::StableHLOLoweringPass(MLIRContext& context) : builder(&context) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
}

mlir::ModuleOp StableHLOLoweringPass::module() { return theModule; }

void StableHLOLoweringPass::run(ModulePtr module) {
  auto graph = module->getGraph();
  auto loweringVisitor = new StableHLOLoweringVisitor(theModule);
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  for (auto block : *graph) {
    for (auto node : *block) {
      node->accept(loweringVisitor);
    }
  }
  if (failed(theModule.verify())) {
    theModule.emitError("module verification error");
    return;
  }
}

void StableHLOLoweringVisitor::visit(NodePtr node) {

}

void StableHLOLoweringVisitor::visit(ParamPtr node) {

}

void StableHLOLoweringVisitor::visit(ReturnOpPtr node) {

}

void StableHLOLoweringVisitor::visit(TransposePtr node) {

}

void StableHLOLoweringVisitor::visit(MatmulPtr node) {

}

mlir::OwningOpRef<mlir::ModuleOp> StableHLOLowering(ModulePtr module) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  // context.loadDialect<mlir::stablehlo::StablehloDialect>();
  auto loweringPass = std::make_unique<StableHLOLoweringPass>(context);
  // loweringPass->module()->setName(module->getName());
  loweringPass->run(module);
  return loweringPass->module();
}

} // namespace ainl::ir
