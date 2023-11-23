#ifndef AINL_SRC_INCLUDE_IR_BUILDING_H
#define AINL_SRC_INCLUDE_IR_BUILDING_H

#include <stack>

#include "function.h"
#include "graph.h"
#include "symbol.h"
#include "visitor.h"

class IRBuilder : public Visitor {
  public:
    IRBuilder() = default;
    explicit IRBuilder(const std::vector<std::string> &params)
        : params(params) {}
    void visitModule(ModuleNode *node) override;
    void visitStmt(StmtNode *node) override;
    void visitExpr(ExprNode *node) override;
    void visitBind(BindNode *node) override;
    void visitVar(VarNode *node) override;
    void visitVarDef(VarDefNode *node) override;
    void visitTuple(TupleNode *node) override;
    void visitConstant(ConstantNode *node) override;
    void visitFunctionDef(FunctionDefNode *node) override;
    void visitBinaryOp(BinaryOpNode *node) override;
    void visitUnaryOp(UnaryOpNode *node) override;
    void visitCall(CallNode *node) override;
    void visitWhile(WhileNode *node) override;
    void visitReturn(ReturnNode *node) override;
    void visitCompare(CompareNode *node) override;
    void visitIf(IfNode *node) override;

    ModulePtr getModule() { return module; }

  private:
    std::vector<std::string> params;
    ModulePtr module;
    std::stack<ValuePtr> values;
};

#endif // AINL_SRC_INCLUDE_IR_BUILDING_H
