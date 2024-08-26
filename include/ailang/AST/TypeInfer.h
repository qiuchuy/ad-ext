#pragma once

#include <any>
#include <functional>
#include <map>
#include <utility>

#include "ailang/IR/TypeContract.h"
#include "ailang/AST/ASTVisitor.h"
#include "ailang/IR/Literal.h"

namespace ainl::ir {

class TypeInfer : public Visitor {
public:
  TypeInfer() = default;
  TypeInfer(const std::vector<std::string> &args,
            const std::vector<TypePtr> &types);
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

private:
  std::map<std::string, TypePtr> typedParams;
  std::string curFunc;
};

} // namespace ainl::ir