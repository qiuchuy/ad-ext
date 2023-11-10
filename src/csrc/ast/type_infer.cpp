#include "type_infer.h"
#include "symbol.h"

TypeInfer::TypeInfer(const std::vector<std::string> &args, const std::vector<TypePtr> &types) {
    assert(args.size() == types.size());
    size_t len = args.size();
    for (size_t i = 0; i < len; i++)
        typedParams[args[i]] = types[i];
}

void TypeInfer::visitModule(ModuleNode *node) {

}

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
    TypePtr returnType = SAFE_TYPE_DOWNCAST((env->lookup(node->getFunction()->getName())->getType()), FunctionType)->getReturnType();
    node->setType(returnType);
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
    for (const auto& param : params) {
        env->insertSymbol(param, typedParams[param]);
    }
    curFunc = node->getName();
}

void TypeInfer::visitReturn(ReturnNode *node) {
    std::vector<TypePtr> types;
    for (const auto& param : typedParams) {
        types.push_back(param.second);
    }
    TypePtr returnType = node->getReturnValue()->getType();
    FunctionTypePtr type = FunctionType::create(TupleType::createUnnamedTuple(types), returnType);
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

void TypeInfer::visitTuple(TupleNode *node) {
    std::vector<TypePtr> types;
    for (const auto& expr : node->getElems()) {
        types.push_back(expr->getType());
    }
    node->setType(TupleType::createUnnamedTuple(types));
}

void TypeInfer::visitVarDef(VarDefNode *node) {
    TypePtr sourceType = node->getSource()->getType();
    auto targets = node->getTargets();
    size_t targetSize = targets.size();
    if (targetSize == 0)
        throw AINLError("Illegal VarDef statement.");
    if (targetSize > 1) {
        for (auto& target : targets) {
            target->setType(sourceType);
            env->insertSymbol(target->getName(), sourceType);
        }
        return ;
    }
    auto target = targets.back();
    if (target->isTupleNode()) {
        assert(sourceType->isTupleType());
        TupleTypePtr sourceTupleType = SAFE_TYPE_DOWNCAST(sourceType, TupleType);
        Tuple targetTupleNode = SAFE_AST_DOWNCAST(target, TupleNode);
        assert(sourceTupleType->getTypes().size() == targetTupleNode->getElems().size());
        size_t len = sourceTupleType->getTypes().size();
        for (size_t i = 0; i < len; i++) {
            targetTupleNode->getElems()[i]->setType(sourceTupleType->getTypes()[i]);
            env->insertSymbol(targetTupleNode->getElems()[i]->getName(), sourceTupleType->getTypes()[i]);
        }
    } else {
        target->setType(sourceType);
    }
}

void TypeInfer::visitWhile(WhileNode *node) {}

void TypeInfer::visitIf(IfNode *node) {}
