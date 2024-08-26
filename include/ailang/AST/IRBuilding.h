#pragma once

#include <stack>
#include <utility>

#include "ailang/IR/NodeContract.h"
#include "ailang/AST/ASTVisitor.h"
#include "ailang/IR/Function.h"
#include "ailang/IR/Graph.h"
#include "ailang/IR/Node.h"
#include "ailang/IR/Symbol.h"
#include "ailang/Utils/Utils.h"

namespace ainl::ir {

class IRBuilder : public Visitor {
public:
  IRBuilder() = default;
  explicit IRBuilder(const std::vector<std::string> &params) : params(params) {}
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
  ValuePtr getTOSValue() {
    auto tos = valueStack.top();
    valueStack.pop();
    return tos;
  }
  std::vector<std::string> params;
  ModulePtr module;
  std::stack<ValuePtr> valueStack;
};

} // namespace ainl::ir