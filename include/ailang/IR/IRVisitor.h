#pragma once

#include "ailang/AST/ASTNode.h"
#include "ailang/IR/Node.h"

namespace ainl::ir {

class IRVisitor {
public:
  virtual void visit(NodePtr node) = 0;
  virtual void visit(ParamPtr node) = 0;
  virtual void visit(ReturnOpPtr node) = 0;
  virtual void visit(TransposePtr node) = 0;
  virtual void visit(ConvolutionPtr node) = 0;
  virtual void visit(MatmulPtr node) = 0;
  virtual void visit(AddPtr node) = 0;
  virtual void visit(PowPtr node) = 0;
  virtual void visit(ReluPtr node) = 0;
  virtual void visit(MeanPtr node) = 0;
  virtual void visit(SlicePtr node) = 0;
  virtual void visit(SqrtPtr node) = 0;
  virtual void visit(SumPtr node) = 0;
  virtual void visit(MaxPtr node) = 0;
  virtual void visit(VariancePtr node) = 0;
  virtual void visit(BatchNorm2dPtr node) = 0;
  virtual void visit(BroadcastPtr node) = 0;
  virtual void visit(Maxpool2dPtr node) = 0;
  virtual void visit(Avgpool2dPtr node) = 0;
  virtual void visit(CompareOpPtr node) = 0;
  virtual void visit(ConcatPtr node) = 0;
  virtual void visit(ExpPtr node) = 0;
  virtual void visit(TanhPtr node) = 0;
  virtual void visit(NegPtr node) = 0;
  virtual void visit(DivPtr node) = 0;
  virtual void visit(MulPtr node) = 0;
  virtual void visit(SelectPtr node) = 0;
  virtual void visit(ConstantDefPtr node) = 0;
  virtual void visit(ReshapePtr node) = 0;
  virtual void visit(ReversePtr node) = 0;
  virtual void visit(ScatterAddMaxPtr node) = 0;
  ~IRVisitor() = default;
};

} // namespace ainl::ir