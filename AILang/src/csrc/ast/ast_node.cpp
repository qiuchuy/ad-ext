#include "ast/ast_node.h"
#include "ast/type_infer.h"
#include "ast/visitor.h"

namespace ainl::ir {

std::array<std::string, 4> UnaryOpString = {"+", "-", "~", "!"};
std::array<std::string, 13> BinaryOpString = {
    "+", "-", "*", "/", "//", "%", "**", "<<", ">>", "|", "^", "&", "@"};
std::array<std::string, 10> CompareOpString = {
    "==", "!=", "<", "<=", ">", ">=", "is", "is not", "in", "not in"};

void ModuleNode::accept(Visitor *visitor) {
    for (const auto &stmt : stmts) {
        stmt->accept(visitor);
    }
    visitor->visitModule(this);
}

void VarNode::accept(Visitor *visitor) { visitor->visitVar(this); }

void VarDefNode::accept(Visitor *visitor) { visitor->visitVarDef(this); }

void BindNode::accept(Visitor *visitor) {
    source->accept(visitor);
    for (const auto &target : targets) {
        target->accept(visitor);
    }
    visitor->visitBind(this);
}

void TupleNode::accept(Visitor *visitor) {
    for (const auto &expr : elems) {
        expr->accept(visitor);
    }
    visitor->visitTuple(this);
}

void ConstantNode::accept(Visitor *visitor) { visitor->visitConstant(this); }

void UnaryOpNode::accept(Visitor *visitor) {
    value->accept(visitor);
    visitor->visitUnaryOp(this);
}

void BinaryOpNode::accept(Visitor *visitor) {
    op1->accept(visitor);
    op2->accept(visitor);
    visitor->visitBinaryOp(this);
}

void FunctionDefNode::accept(Visitor *visitor) {
    visitor->visitFunctionDef(this);
    for (const auto &stmt : body) {
        stmt->accept(visitor);
    }
}

void ReturnNode::accept(Visitor *visitor) {
    value->accept(visitor);
    visitor->visitReturn(this);
}

void CompareNode::accept(Visitor *visitor) {
    left->accept(visitor);
    for (const auto &comparator : comparators) {
        comparator->accept(visitor);
    }
    visitor->visitCompare(this);
}

void WhileNode::accept(Visitor *visitor) {
    cond->accept(visitor);
    for (const auto &stmt : body) {
        stmt->accept(visitor);
    }
    visitor->visitWhile(this);
}

void CallNode::accept(Visitor *visitor) {
    // func->accept(visitor);
    for (const auto &arg : args) {
        arg->accept(visitor);
    }
    visitor->visitCall(this);
}

void IfNode::accept(Visitor *visitor) {
    cond->accept(visitor);
    for (const auto &stmt : thenBranch) {
        stmt->accept(visitor);
    }
    for (const auto &stmt : elseBranch) {
        stmt->accept(visitor);
    }
    visitor->visitIf(this);
}
} // namespace ainl::ir