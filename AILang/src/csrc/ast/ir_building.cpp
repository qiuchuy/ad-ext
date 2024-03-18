#include "ast/ir_building.h"
#include "ir/graph.h"

namespace ainl::ir {

ValuePtr reluNodeContract(const GraphPtr &graph, const TypePtr &nodeType,
                          const ValuePtr &inValue) {
    if (!inValue->getType()->isTensorType()) {
        throw ainl::core::AINLError("relu operator only applies to tensors.");
    }
    return graph->create<Relu>(nodeType, inValue);
}

ValuePtr transposeNodeContract(const GraphPtr &graph, const TypePtr &nodeType,
                               const ValuePtr &inValue) {
    if (!inValue->getType()->isTensorType()) {
        throw ainl::core::AINLError(
            "transpose operator only applies to tensors.");
    }
    return graph->create<Transpose>(nodeType, inValue);
}
ValuePtr matmulNodeContract(const GraphPtr &graph, const TypePtr &nodeType,
                            const ValuePtr &lhs, const ValuePtr &rhs) {
    // Type Checking
    if (!lhs->getType()->isTensorType() || !rhs->getType()->isTensorType()) {
        throw ainl::core::AINLError(
            "matmul operator only applies to two tensors.");
    }
    // Construct the Result Node
    return graph->create<Matmul>(nodeType, lhs, rhs);
}

ValuePtr maxpool2dNodeContract(const GraphPtr &graph, const TypePtr &nodeType,
                               const ValuePtr &inValue) {
    if (!inValue->getType()->isTensorType()) {
        throw ainl::core::AINLError(
            "maxpool2d operator only applies to tensors.");
    }
    return graph->create<Maxpool2d>(nodeType, inValue);
}
ValuePtr convolutionNodeContract(const GraphPtr &graph, const TypePtr &nodeType,
                                 const ValuePtr &inValue) {
    if (!inValue->getType()->isTensorType()) {
        throw ainl::core::AINLError(
            "convolution operator only applies to tensors.");
    }
    return graph->create<Convolution>(nodeType, inValue);
}
ValuePtr batchnorm2dNodeContract(const GraphPtr &graph, const TypePtr &nodeType,
                                 const ValuePtr &inValue) {
    if (!inValue->getType()->isTensorType()) {
        throw ainl::core::AINLError(
            "batchnorm2d operator only applies to tensors.");
    }
    return graph->create<BatchNorm2d>(nodeType, inValue);
}

void IRBuilder::initLibraryOperatorNodeContract() {
    contract.registerContract("matmul", [](const GraphPtr &graph,
                                           const TypePtr &nodeType,
                                           std::vector<ValuePtr> args) {
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator matmul");
        }
        return matmulNodeContract(graph, nodeType, (args[0]), (args[1]));
    });
    contract.registerContract("relu", [](const GraphPtr &graph,
                                         const TypePtr &nodeType,
                                         std::vector<ValuePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator relu");
        }
        return reluNodeContract(graph, nodeType, (args[0]));
    });
    contract.registerContract("transpose", [](const GraphPtr &graph,
                                              const TypePtr &nodeType,
                                              std::vector<ValuePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator transpose");
        }
        return transposeNodeContract(graph, nodeType, (args[0]));
    });
    contract.registerContract("maxpool2d", [](const GraphPtr &graph,
                                              const TypePtr &nodeType,
                                              std::vector<ValuePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator maxpool2d");
        }
        return maxpool2dNodeContract(graph, nodeType, (args[0]));
    });
    contract.registerContract("convolution", [](const GraphPtr &graph,
                                                const TypePtr &nodeType,
                                                std::vector<ValuePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator convolution");
        }
        return convolutionNodeContract(graph, nodeType, (args[0]));
    });
    contract.registerContract("batchnorm2d", [](const GraphPtr &graph,
                                                const TypePtr &nodeType,
                                                std::vector<ValuePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator batchnorm2d");
        }
        return batchnorm2dNodeContract(graph, nodeType, (args[0]));
    });
}

void IRBuilder::visitVarDef(VarDefNode *node) {}
void IRBuilder::visitBind(BindNode *node) {}
void IRBuilder::visitIf(IfNode *node) {}
void IRBuilder::visitWhile(WhileNode *node) {}
void IRBuilder::visitTuple(TupleNode *node) {}
void IRBuilder::visitCall(CallNode *node) {
    std::string funcName = node->getCallFunction()->getName();
    // for now, we use "::" to specify namespace
    size_t lastNamespace = funcName.find_last_of("::");
    if (lastNamespace != std::string::npos) {
        // This is a library function call
        std::string libraryFunction = (funcName.substr(lastNamespace + 1));
        std::vector<Expr> callArgs = node->getCallArgs();
        std::vector<ValuePtr> argValues;
        for (size_t i = 0; i < callArgs.size(); i++) {
            argValues.push_back(getTOSValue());
        }
        ValuePtr callResult = contract.resolveContract(
            libraryFunction, module->getGraph(), node->getType(), argValues);
        valueStack.push(callResult);
    } else {
        // [TODO]
    }
}
void IRBuilder::visitConstant(ConstantNode *node) {}
void IRBuilder::visitUnaryOp(UnaryOpNode *node) {}
void IRBuilder::visitStmt(StmtNode *node) {}
void IRBuilder::visitCompare(CompareNode *node) {}
void IRBuilder::visitReturn(ReturnNode *node) {
    ValuePtr returnValue = valueStack.top();
    valueStack.pop();
    module->getGraph()->create<ReturnOp>(returnValue);
}

void IRBuilder::visitExpr(ExprNode *node) {}
void IRBuilder::visitModule(ModuleNode *node) {
    // module = std:    :make_shared<ALModule>();
}
void IRBuilder::visitVar(VarNode *node) {
    ValuePtr value = env->lookup(node->getName())->getValue();
    valueStack.push(value);
}

void IRBuilder::visitFunctionDef(FunctionDefNode *node) {
    std::string funcName = node->getName();
    TypePtr funcType = env->lookup(funcName)->getType();
    if (funcType->isFunctionType()) {
        FunctionTypePtr funcTypePtr =
            SAFE_TYPE_DOWNCAST(funcType, FunctionType);
        TypePtr argType = funcTypePtr->getArgType();
        TypePtr returnType = funcTypePtr->getReturnType();
        module = std::make_shared<ALModule>(funcName, argType, returnType);
        std::vector<ValuePtr> paramValues = module->getParams();
        for (size_t idx = 0; idx < params.size(); idx++) {
            env->lookup(params[idx])->setValue(paramValues[idx]);
        }
    } else {
        // throw AINLError("function " + funcName +
        // " is not been assigned to FunctionType.");
    }
}
void IRBuilder::visitBinaryOp(BinaryOpNode *node) {}
} // namespace ainl::ir