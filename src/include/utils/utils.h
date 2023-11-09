#ifndef AINL_SRC_INCLUDE_UTILS_H
#define AINL_SRC_INCLUDE_UTILS_H

#include "ast_node.h"

#define DISPATCH_BINARYOP(op)                                                  \
    if (OpKind == #op)                                                         \
        return BinaryOpNode::BinaryOpKind::op;
#define DISPATCH_UNARYOP(op)                                                   \
    if (OpKind == #op)                                                         \
        return UnaryOpNode::UnaryOpKind::op;
#define DISPATCH_COMPAREOP(op)                                                 \
    if (OpKind == #op)                                                         \
        return CompareOpNode::CompareOpKind::op;
BinaryOpNode::BinaryOpKind BinaryOpASTHelper(std::string OpKind);
UnaryOpNode::UnaryOpKind UnaryOpASTHelper(std::string OpKind);
CompareOpNode::CompareOpKind CompareOpASTHelper(std::string OpKind);

#endif // AINL_SRC_INCLUDE_UTILS_H
