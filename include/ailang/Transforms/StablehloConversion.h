#pragma once

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include "ailang/IR/Function.h"
#include "ailang/IR/IRVisitor.h"
#include "ailang/IR/Node.h"
#include "ailang/Transforms/Pass.h"

namespace ainl::ir {

class StableHLOLoweringPass : public Pass, public IRVisitor {
public:
  StableHLOLoweringPass(mlir::MLIRContext &context, const std::string &name);
  void run(ModulePtr module) override;
  mlir::ModuleOp module();

  void visit(NodePtr node) override;
  void visit(ParamPtr node) override;
  void visit(ReturnOpPtr node) override;
  void visit(TransposePtr node) override;
  void visit(ConvolutionPtr node) override;
  void visit(BatchNorm2dPtr node) override;
  void visit(ReluPtr node) override;
  void visit(SqrtPtr node) override;
  void visit(MeanPtr node) override;
  void visit(SumPtr node) override;
  void visit(VariancePtr node) override;
  void visit(MatmulPtr node) override;
  void visit(AddPtr node) override;
  void visit(PowPtr node) override;
  void visit(Maxpool2dPtr node) override;
  void visit(Avgpool2dPtr node) override;
  void visit(CompareOpPtr node) override;
  void visit(ConcatPtr node) override;
  void visit(ExpPtr node) override;
  void visit(TanhPtr node) override;
  void visit(NegPtr node) override;
  void visit(SelectPtr node) override;
  void visit(DivPtr node) override;
  void visit(BroadcastPtr node) override;
  void visit(MulPtr node) override;
  void visit(ConstantDefPtr node) override;

private:
  mlir::func::FuncOp createFunctionOpFromModule(ModulePtr module);
  void insertValueMapping(ValuePtr value, mlir::Value mlirValue);
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  llvm::DenseMap<ValuePtr, mlir::Value> valueMap;
};

std::string StableHLOLowering(ModulePtr module);
mlir::RankedTensorType
createRankedTensorTypeFromTensorType(TypePtr type, mlir::MLIRContext &context);
mlir::Type createTypeFromElementType(TypePtr type, mlir::MLIRContext &context);
std::string mlirModuleToString(mlir::ModuleOp module);

} // namespace ainl::ir