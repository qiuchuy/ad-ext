#include "ast/ast.h"

namespace ainl::ir {

void ASTNode::accept(Visitor *visitor) {}
void StmtNode::accept(Visitor *visitor) {}
void ExprNode::accept(Visitor *visitor) {}
} // namespace ainl::ir