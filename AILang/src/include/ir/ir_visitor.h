
#pragma once

#include "ir/node.h"

namespace ainl::ir {

class IRVisitor {
  public:
    virtual void visit(NodePtr node) = 0;
    virtual void visit(ParamPtr node) = 0;
    virtual void visit(ReturnOpPtr node) = 0;
    virtual void visit(TransposePtr node) = 0;
    virtual void visit(MatmulPtr node) = 0;
    ~IRVisitor() = default;
};

} // namespace ainl::ir
