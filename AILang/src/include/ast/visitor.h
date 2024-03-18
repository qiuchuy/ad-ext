#pragma once

#include "ast/ast_node.h"

namespace ainl::ir {

#define AST_PTR_TYPE_DECL(name) class name##Node;

AST_PTR_TYPE_DECL(Module)
AST_PTR_TYPE_DECL(Stmt)
AST_PTR_TYPE_DECL(Expr)
AST_PTR_TYPE_DECL(Bind)
AST_PTR_TYPE_DECL(VarDef)
AST_PTR_TYPE_DECL(Var)
AST_PTR_TYPE_DECL(Tuple)
AST_PTR_TYPE_DECL(Constant)
AST_PTR_TYPE_DECL(FunctionDef)
AST_PTR_TYPE_DECL(UnaryOp)
AST_PTR_TYPE_DECL(Call)
AST_PTR_TYPE_DECL(While)
AST_PTR_TYPE_DECL(Return)
AST_PTR_TYPE_DECL(If)
AST_PTR_TYPE_DECL(Compare)

class Visitor {
  public:
    ~Visitor() = default;
    virtual void visitModule(ModuleNode *node) = 0;
    virtual void visitStmt(StmtNode *node) = 0;
    virtual void visitExpr(ExprNode *node) = 0;
    virtual void visitBind(BindNode *node) = 0;
    virtual void visitVar(VarNode *node) = 0;
    virtual void visitVarDef(VarDefNode *node) = 0;
    virtual void visitTuple(TupleNode *node) = 0;
    virtual void visitConstant(ConstantNode *node) = 0;
    virtual void visitFunctionDef(FunctionDefNode *node) = 0;
    virtual void visitBinaryOp(BinaryOpNode *node) = 0;
    virtual void visitUnaryOp(UnaryOpNode *node) = 0;
    virtual void visitCall(CallNode *node) = 0;
    virtual void visitWhile(WhileNode *node) = 0;
    virtual void visitReturn(ReturnNode *node) = 0;
    virtual void visitCompare(CompareNode *node) = 0;
    virtual void visitIf(IfNode *node) = 0;
};
} // namespace ainl::ir