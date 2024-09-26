#pragma once

#include "ailang/IR/Function.h"
#include "ailang/IR/IRVisitor.h"
#include "ailang/IR/Node.h"
#include "ailang/IR/Use.h"
#include "ailang/Transforms/Pass.h"
#include "mlir/IR/IRMapping.h"

namespace ainl::ir {
class AutoDiff : public Pass, public IRVisitor {
public:
  AutoDiff(ModulePtr Module) : Module(Module) {}
  void run(ModulePtr Module) override;

  void visit(NodePtr Node) override {};
  void visit(ParamPtr Node) override;
  void visit(ReturnOpPtr Node) override;
  void visit(TransposePtr Node) override;
  void visit(ConvolutionPtr Node) override {};
  void visit(BatchNorm2dPtr Node) override {};
  void visit(ReluPtr Node) override {};
  void visit(SqrtPtr Node) override {};
  void visit(MeanPtr Node) override {};
  void visit(VariancePtr Node) override {};
  void visit(MatmulPtr Node) override;
  void visit(AddPtr Node) override;
  void visit(SumPtr Node) override;
  void visit(Maxpool2dPtr Node) override {};
  void visit(Avgpool2dPtr Node) override {};
  void visit(CompareOpPtr Node) override {};
  void visit(ConcatPtr Node) override {};
  void visit(ExpPtr Node) override;
  void visit(TanhPtr Node) override;
  void visit(NegPtr Node) override;
  void visit(DivPtr Node) override;
  void visit(BroadcastPtr Node) override;
  void visit(MulPtr Node) override;
  void visit(ConstantDefPtr Node) override;

private:
  llvm::DenseMap<ValuePtr, ValuePtr> AdjointMap;
  ModulePtr Module;
  ValuePtr ReturnValue;

private:
  ValuePtr getAdjoint(ValuePtr Value) {
    if (AdjointMap.find(Value) == AdjointMap.end()) {
      throw std::runtime_error("Adjoint not found");
    }
    return AdjointMap[Value];
  }

  void setAdjoint(ValuePtr Value, ValuePtr Adjoint) {
    if (AdjointMap.find(Value) != AdjointMap.end()) {
      // accumulate adjoint
      auto *ValueAdjoint = AdjointMap[Value];
      auto *NewAdjoint =
          Module->create<Add>(Adjoint->getType(), Adjoint, ValueAdjoint);
      AdjointMap[Value] = NewAdjoint;
    } else {
      AdjointMap[Value] = Adjoint;
    }
  }

  bool hasAdjoint(ValuePtr Value) {
    return AdjointMap.find(Value) != AdjointMap.end();
  }
};

void autodiffOnModule(ModulePtr Module);
} // namespace ainl::ir