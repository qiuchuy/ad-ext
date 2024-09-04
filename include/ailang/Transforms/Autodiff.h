#pragma once

#include "ailang/IR/Function.h"
#include "ailang/IR/IRVisitor.h"
#include "ailang/IR/Node.h"
#include "ailang/IR/Use.h"
#include "ailang/Transforms/Pass.h"

namespace ainl::ir {
class AutodiffPass : public Pass, public IRVisitor {
public:
  AutodiffPass() = default;
  void run(ModulePtr Module) override;

  void visit(NodePtr Node) override{};
  void visit(ParamPtr Node) override;
  void visit(ReturnOpPtr Node) override;
  void visit(TransposePtr Node) override{};
  void visit(ConvolutionPtr Node) override{};
  void visit(BatchNorm2dPtr Node) override{};
  void visit(ReluPtr Node) override{};
  void visit(MeanPtr Node) override{};
  void visit(VariancePtr Node) override {};
  void visit(MatmulPtr Node) override {};
  void visit(AddPtr Node) override{};
  void visit(Maxpool2dPtr Node) override{};
  void visit(CompareOpPtr Node) override{};

private:
  llvm::DenseMap<ValuePtr, ValuePtr> TangentMap;
};

void autodiffOnModule(ModulePtr Module);
} // namespace ainl::ir