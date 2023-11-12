#ifndef AINL_SRC_INCLUDE_TYPE_INFER_H
#define AINL_SRC_INCLUDE_TYPE_INFER_H

#include <map>
#include <functional>

#include "visitor.h"

template <typename... Args>
struct ContractHolder {
    static std::map<std::string, TypePtr (*)(Args...)> contractMap;
};

template <class... Args>
std::map<std::string, TypePtr (*)(Args...)> ContractHolder<Args...>::contractMap;

class TypeContract {
public:
    template <typename... Args>
    static void registerContract(std::string name, TypePtr (*contract)(Args...)) {
        ContractHolder<Args...>::contractMap[name] = contract;
    }

    template <typename... Args>
    static TypePtr query(const std::string& name, Args &&... args) {
        return ContractHolder<Args...>::contractMap[name](std::forward<Args>(args)...);
    }
};

#define REGISTER_TYPE_CONTRACT(name, contract) \
    TypeContract::registerContract(name, contract);

/*
template <typename NodeType, typename... ARGS>
NodePtr Graph::create(ARGS &&...args) {
    NodePtr node = new NodeType(std::forward<ARGS>(args)...);
    node->graph = shared_from_this();
    node->block = endBlock;
    insertNodeAtEnd(node);
    return node;
}
*/



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

#endif
