#pragma once

#include "ailang/AST/ASTNode.h"
#include "ailang/IR/Function.h"
#include "ailang/IR/Value.h"

namespace ainl::core {

#define DISPATCH_BINARYOP(op)                                                  \
  if (OpKind == #op)                                                           \
    return ainl::ir::BinaryOpNode::BinaryOpKind::op;
#define DISPATCH_UNARYOP(op)                                                   \
  if (OpKind == #op)                                                           \
    return ainl::ir::UnaryOpNode::UnaryOpKind::op;
#define DISPATCH_COMPAREOP(op)                                                 \
  if (OpKind == #op)                                                           \
    return ainl::ir::CompareNode::CompareOpKind::op;
ainl::ir::BinaryOpNode::BinaryOpKind BinaryOpASTHelper(std::string OpKind);
ainl::ir::UnaryOpNode::UnaryOpKind UnaryOpASTHelper(std::string OpKind);
ainl::ir::CompareNode::CompareOpKind CompareOpASTHelper(std::string OpKind);

std::string trim(const std::string &str);
int caseInsensitiveStrcmp(const std::string &str1, const std::string &str2);
ir::ValuePtr createFloatConstantValue(ir::ModulePtr M, float Value,
                                      ainl::ir::TypePtr Type);
} // namespace ainl::core