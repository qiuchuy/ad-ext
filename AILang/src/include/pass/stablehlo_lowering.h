#pragma once

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"


#include "ir/function.h"
#include "ir/ir_visitor.h"
#include "pass/pass.h"

namespace ainl::ir {

class StableHLOLoweringPass : public Pass {
public:
  StableHLOLoweringPass(mlir::MLIRContext &context);
  void run(ModulePtr module) override;
  mlir::ModuleOp module();
private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
};

class StableHLOLoweringVisitor : public IRVisitor {
public:
  StableHLOLoweringVisitor(mlir::ModuleOp &theModule) : theModule(theModule) {}
  void visit(NodePtr node) override;
  void visit(ParamPtr node) override;
  void visit(ReturnOpPtr node) override;
  void visit(TransposePtr node) override;
  void visit(MatmulPtr node) override;

private:
  mlir::ModuleOp &theModule;
};

mlir::OwningOpRef<mlir::ModuleOp> StableHLOLowering(ModulePtr module);

} // namespace ainl::ir