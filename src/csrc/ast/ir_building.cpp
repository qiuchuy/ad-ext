#include "ir_building.h"

void IRBuilder::visitVarDef(VarDefNode *node) {}
void IRBuilder::visitBind(BindNode *node) {}
void IRBuilder::visitIf(IfNode *node) {}
void IRBuilder::visitWhile(WhileNode *node) {}
void IRBuilder::visitTuple(TupleNode *node) {}
void IRBuilder::visitCall(CallNode *node) {}
void IRBuilder::visitConstant(ConstantNode *node) {}
void IRBuilder::visitUnaryOp(UnaryOpNode *node) {}
void IRBuilder::visitStmt(StmtNode *node) {}
void IRBuilder::visitCompare(CompareNode *node) {}
void IRBuilder::visitReturn(ReturnNode *node) {}
void IRBuilder::visitExpr(ExprNode *node) {}
void IRBuilder::visitModule(ModuleNode *node) {
    // module = std::make_shared<ALModule>();
}
void IRBuilder::visitVar(VarNode *node) {}
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
        throw AINLError("function " + funcName +
                        " is not been assigned to FunctionType.");
    }
}
void IRBuilder::visitBinaryOp(BinaryOpNode *node) {}
