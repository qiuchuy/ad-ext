#ifndef AINL_SRC_INCLUDE_TYPE_INFER_H
#define AINL_SRC_INCLUDE_TYPE_INFER_H

#include <any>
#include <functional>
#include <map>
#include <utility>

#include "visitor.h"

class TypeContract {
  public:
    using AnyFunction = std::function<TypePtr(std::vector<std::any>)>;

    void registerContract(const std::string &name, AnyFunction func) {
        functions[name] = std::move(func);
    }

    TypePtr resolveContract(const std::string &name,
                            std::vector<std::any> args) {
        if (functions.find(name) == functions.end()) {
            throw AINLError(
                "This operator has not been registered into the library yet.");
        }
        return functions[name](std::move(args));
    }

  private:
    std::map<std::string, AnyFunction> functions;
};

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

    void initLibraryOperatorTypeContract();

  private:
    std::map<std::string, TypePtr> typedParams;
    std::string curFunc;
    TypeContract contract;
};

#endif
