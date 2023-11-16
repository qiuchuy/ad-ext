#include "type_infer.h"
#include "symbol.h"
#include "utils.h"

TypePtr matmulContract(const TypePtr &lhsType, const TypePtr &rhsType) {
    // Type Checking
    assert(lhsType->isTensorType() && rhsType->isTensorType());
    if (!lhsType->isTensorType() || !rhsType->isTensorType()) {
        throw AINLError("matmul operator only applies to two tensors.");
    }
    TensorTypePtr lhsTensorType = SAFE_TYPE_DOWNCAST(lhsType, TensorType);
    TensorTypePtr rhsTensorType = SAFE_TYPE_DOWNCAST(rhsType, TensorType);
    std::vector<ValuePtr> lhsShape = lhsTensorType->getShape();
    std::vector<ValuePtr> rhsShape = rhsTensorType->getShape();
    if (*lhsShape.back() != *rhsShape.front()) {
        throw AINLError("tensor shapes are not matched for matmul.");
    }

    // Construct the result type
    std::vector<ValuePtr> lhsShapeWithoutLastDim(lhsShape.begin(),
                                                 lhsShape.end() - 1);
    std::vector<ValuePtr> rhsShapeWithoutFirstDim(rhsShape.begin() + 1,
                                                  rhsShape.end());
    std::vector<ValuePtr> matmulShape(lhsShapeWithoutLastDim.begin(),
                                      lhsShapeWithoutLastDim.end());
    matmulShape.insert(matmulShape.end(), rhsShapeWithoutFirstDim.begin(),
                       rhsShapeWithoutFirstDim.end());
    TypePtr elementType = lhsTensorType->getElementType();
    return TensorType::create(elementType, matmulShape);
}

void TypeInfer::initLibraryOperatorTypeContract() {
    contract.registerContract("matmul", [](std::vector<std::any> args) {
        if (args.size() != 2) {
            throw AINLError("Invalid argument number for operator matmul");
        }
        return matmulContract(std::any_cast<TypePtr>(args[0]),
                              std::any_cast<TypePtr>(args[1]));
    });
}

TypeInfer::TypeInfer(const std::vector<std::string> &args,
                     const std::vector<TypePtr> &types) {

    assert(args.size() == types.size());
    initLibraryOperatorTypeContract();
    size_t len = args.size();
    for (size_t i = 0; i < len; i++)
        typedParams[args[i]] = types[i];
}

void TypeInfer::visitModule(ModuleNode *node) {}

void TypeInfer::visitBinaryOp(BinaryOpNode *node) {
    TypePtr lhsType = node->getLHS()->getType();
    TypePtr rhsType = node->getRHS()->getType();
    if (lhsType->compare(*rhsType)) {
        node->setType(rhsType);
    } else {
        node->setType(lhsType);
    }
}

void TypeInfer::visitCall(CallNode *node) {
    std::string funcName = node->getCallFunction()->getName();
    // for now, we use "::" to specify namespace
    size_t lastNamespace = funcName.find_last_of("::");
    if (lastNamespace != std::string::npos) {
        // This is a library function call
        std::string libraryFunction = (funcName.substr(lastNamespace + 1));
        std::vector<Expr> callArgs = node->getCallArgs();
        std::vector<std::any> argTypes;
        for (const auto &arg : callArgs) {
            argTypes.push_back(arg->getType());
        }
        TypePtr returnType =
            contract.resolveContract(libraryFunction, argTypes);
        node->setType(returnType);
    } else {
        TypePtr returnType =
            SAFE_TYPE_DOWNCAST(
                (env->lookup(node->getCallFunction()->getName())->getType()),
                FunctionType)
                ->getReturnType();
        node->setType(returnType);
    }
}

void TypeInfer::visitConstant(ConstantNode *node) {
    std::string value = node->getValue();
    // [TODO] A naive method, improve this
    if (value.find('.') != std::string::npos) {
        node->setType(FloatTypePtr::get());
    } else if (value == "True" || value == "False") {
        node->setType(BoolTypePtr::get());
    } else {
        node->setType(IntTypePtr::get());
    }
}

void TypeInfer::visitExpr(ExprNode *node) {}

void TypeInfer::visitFunctionDef(FunctionDefNode *node) {
    env = new Environment();
    auto params = node->getParams();
    for (const auto &param : params) {
        env->insertSymbol(param, typedParams[param]);
    }
    curFunc = node->getName();
}

void TypeInfer::visitReturn(ReturnNode *node) {
    std::vector<TypePtr> types;
    for (const auto &param : typedParams) {
        types.push_back(param.second);
    }
    TypePtr returnType = node->getReturnValue()->getType();
    FunctionTypePtr type =
        FunctionType::create(TupleType::createUnnamedTuple(types), returnType);
    env->insertSymbol(curFunc, type);
}

void TypeInfer::visitCompare(CompareNode *node) {
    node->setType(BoolTypePtr::get());
}

void TypeInfer::visitStmt(StmtNode *node) {}

void TypeInfer::visitUnaryOp(UnaryOpNode *node) {
    Expr operand = node->getOperand();
    node->setType(operand->getType());
}

void TypeInfer::visitVar(VarNode *node) {
    TypePtr type = env->lookup(node->getName())->getType();
    node->setType(type);
}

void TypeInfer::visitVarDef(VarDefNode *node) {}

void TypeInfer::visitTuple(TupleNode *node) {
    std::vector<TypePtr> types;
    for (const auto &expr : node->getElems()) {
        types.push_back(expr->getType());
    }
    node->setType(TupleType::createUnnamedTuple(types));
}

void TypeInfer::visitBind(BindNode *node) {
    TypePtr sourceType = node->getSource()->getType();
    auto targets = node->getTargets();
    size_t targetSize = targets.size();
    if (targetSize == 0)
        throw AINLError("Illegal Bind statement.");
    if (targetSize > 1) {
        for (auto &target : targets) {
            target->setType(sourceType);
            env->insertSymbol(target->getName(), sourceType);
        }
        return;
    }
    auto target = targets.back();
    if (target->isTupleNode()) {
        assert(sourceType->isTupleType());
        TupleTypePtr sourceTupleType =
            SAFE_TYPE_DOWNCAST(sourceType, TupleType);
        Tuple targetTupleNode = SAFE_AST_DOWNCAST(target, TupleNode);
        assert(sourceTupleType->getTypes().size() ==
               targetTupleNode->getElems().size());
        size_t len = sourceTupleType->getTypes().size();
        for (size_t i = 0; i < len; i++) {
            targetTupleNode->getElems()[i]->setType(
                sourceTupleType->getTypes()[i]);
            env->insertSymbol(targetTupleNode->getElems()[i]->getName(),
                              sourceTupleType->getTypes()[i]);
        }
    } else {
        target->setType(sourceType);
        env->insertSymbol(target->getName(), sourceType);
    }
}

void TypeInfer::visitWhile(WhileNode *node) {}

void TypeInfer::visitIf(IfNode *node) {}
