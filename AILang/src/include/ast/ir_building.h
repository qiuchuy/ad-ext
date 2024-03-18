#pragma once

#include <stack>
#include <utility>

#include "ir/function.h"
#include "ir/graph.h"
#include "ir/symbol.h"
#include "utils/utils.h"
#include "ast/visitor.h"

namespace ainl::ir {

class NodeContract {
  public:
    using AnyFunction = std::function<ValuePtr(GraphPtr, TypePtr nodeType,
                                               std::vector<ValuePtr>)>;

    void registerContract(const std::string &name, AnyFunction func) {
        functions[name] = std::move(func);
    }

    ValuePtr resolveContract(const std::string &name, GraphPtr graph,
                             TypePtr nodeType, std::vector<ValuePtr> args) {
        if (functions.find(name) == functions.end()) {
            // throw AINLError(
                // "This operator has not been registered into the library yet.");
        }
        return functions[name](std::move(graph), std::move(nodeType),
                               std::move(args));
    }

  private:
    std::map<std::string, AnyFunction> functions;
};

class IRBuilder : public Visitor {
  public:
    IRBuilder() = default;
    explicit IRBuilder(const std::vector<std::string> &params)
        : params(params) {
        initLibraryOperatorNodeContract();
    }
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
    void initLibraryOperatorNodeContract();

  private:
    ValuePtr getTOSValue() {
        auto tos = valueStack.top();
        valueStack.pop();
        return tos;
    }
    std::vector<std::string> params;
    ModulePtr module;
    std::stack<ValuePtr> valueStack;
    NodeContract contract;
};

}