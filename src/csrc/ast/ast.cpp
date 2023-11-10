#include "ast.h"

void ASTNode::accept(Visitor *visitor) {}
void StmtNode::accept(Visitor *visitor) {}
void ExprNode::accept(Visitor *visitor) {}