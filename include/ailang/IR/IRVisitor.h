#pragma once

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
  virtual void visit(ReluPtr node) = 0;
  virtual void visit(MeanPtr node) = 0;
  virtual void visit(VariancePtr node) = 0;
  virtual void visit(BatchNorm2dPtr node) = 0;
  virtual void visit(Maxpool2dPtr node) = 0;
  virtual void visit(CompareOpPtr node) = 0;
  ~IRVisitor() = default;
};

} // namespace ainl::ir