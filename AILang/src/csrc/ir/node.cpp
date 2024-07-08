#include "ir/node.h"

#include <memory>
#include <utility>

#include "ast/ast.h"
#include "ir/block.h"
#include "ir/function.h"
#include "ir/ir_visitor.h"

namespace ainl::ir {

int Node::LOCAL_COUNT = 0;

Node::Node() { init(); }

Node::Node(const TypePtr &type) : Value(type) {
    init();
    prefix = LOCAL_PREFIX;
    name = LOCAL_NAME_PREFIX + std::to_string(LOCAL_COUNT++);
}

void Node::setUse(ValuePtr value, int idx) {
    auto use = new Use(this, value, idx);
    value->insertUseAtEnd(use);
    useList.push_back(use);
    useValueList.push_back(value);
}

void Node::addBlock() {
    auto newBlock = new Block();
    if (this->block->endBlock) {
        this->block->endBlock->insertBefore(newBlock);
        return;
    }
    this->block->beginBlock = new Block();
    this->block->endBlock = new Block();
    this->block->beginBlock->setNext(this->block->endBlock);
    this->block->endBlock->setPrev(this->block->beginBlock);
    this->block->endBlock->insertBefore(block);
}

void Node::addBlockWithParam(NodePtr param, GraphPtr graph) {
    auto newBlock = new Block(Block::blockCount++);
    newBlock->paramNode = dynamic_cast<ParamPtr>(param);
    for (auto &innerParam : dynamic_cast<ParamPtr>(param)->getParams()) {
        innerParam->block = newBlock;
    }
    param->block = newBlock;
    if (this->block->endBlock) {
        this->block->endBlock->insertBefore(newBlock);
        return;
    }
    /* for nested blocks */
    this->block->beginBlock = new Block();
    this->block->endBlock = new Block();
    this->block->beginBlock->setNext(this->block->endBlock);
    this->block->endBlock->setPrev(this->block->beginBlock);

    /* insert this new block to graph */
    graph->endBlock->insertBefore(newBlock);
}

void Node::accept(IRVisitor *visitor) { visitor->visit(this); }

Param::Param(std::vector<ValuePtr> params, const TypePtr &type) : Node(type) {
    this->params = std::move(params);
    this->contentType = type;
}

void Param::accept(IRVisitor *visitor) { visitor->visit(this); }

ReturnOp::ReturnOp(const ValuePtr &value) : Node(value->getType()) {
    this->value = value;
}

void ReturnOp::accept(IRVisitor *visitor) { visitor->visit(this); }

// Matmul
Matmul::Matmul(const TypePtr &opType, const ValuePtr &lhs, const ValuePtr &rhs)
    : Node(opType) {
    this->lhs = lhs;
    this->rhs = rhs;
}

Matmul::operator std::string() const {
    return getName() + " = ailang::matmul(" + getLHS()->getName() + ", " +
           getRHS()->getName() + "): " + std::string(*getType());
}

void Matmul::accept(IRVisitor *visitor) { visitor->visit(this); }
// Add
Add::Add(const TypePtr &opType, const ValuePtr &lhs, const ValuePtr &rhs)
    : Node(opType) {
    this->lhs = lhs;
    this->rhs = rhs;
}

Add::operator std::string() const {
    return getName() + " = ailang::add(" + getLHS()->getName() + ", " +
           getRHS()->getName() + "): " + std::string(*getType());
}

void Add::accept(IRVisitor *visitor) { visitor->visit(this); }

// Broadcast
// Broadcast::Broadcast(const TypePtr &opType, const ValuePtr &inValue,
//                      std::vector<ir::ValuePtr>)
//     : Node(opType) {
//     this->inValue = inValue;
// }
// Broadcast::operator std::string() const {
//     return getName() + " = ailang::broadcast(" + getValue()->getName() +
//            "):" + std::string(*getType()) + getArgs()->getName();
// }

// Relu
Relu::Relu(const TypePtr &opType, const ValuePtr &inValue) : Node(opType) {
    this->inValue = inValue;
}
Relu::operator std::string() const {
    return getName() + " = ailang::relu(" + getValue()->getName() +
           "):" + std::string(*getType());
}

// Transpose
Transpose::Transpose(const TypePtr &opType, const ValuePtr &inValue)
    : Node(opType) {
    this->inValue = inValue;
}
Transpose::operator std::string() const {
    return getName() + " = ailang::transpose(" + getValue()->getName() +
           "):" + std::string(*getType());
}

void Transpose::accept(IRVisitor *visitor) { visitor->visit(this); }

std::vector<int> Transpose::getShape() {
    if (auto tensorType =
            dynamic_cast<TensorType *>(inValue->getType().get())) {
        return tensorType->getConcreteShape();
    } else {
        throw std::runtime_error("Transpose input is not a tensor");
    }
}

// Maxpool2d
Maxpool2d::Maxpool2d(const TypePtr &opType, const ValuePtr &inValue)
    : Node(opType) {
    this->inValue = inValue;
}
Maxpool2d::operator std::string() const {
    return getName() + " = ailang::maxpool2d(" + getValue()->getName() +
           "):" + std::string(*getType());
}
// Convolution
Convolution::Convolution(const TypePtr &opType, const ValuePtr &inputValue,
                         const ValuePtr &weightValue)
    : Node(opType) {
    this->inputValue = inputValue;
    this->weightValue = weightValue;
}
Convolution::operator std::string() const {
    return getName() + " = ailang::convolution(" + getInputValue()->getName() +
           "," + getWeightValue()->getName() + "):" + std::string(*getType());
}
void Convolution::accept(IRVisitor *visitor) { visitor->visit(this); }

// BatchNorm2d
BatchNorm2d::BatchNorm2d(const TypePtr &opType, const ValuePtr &inValue)
    : Node(opType) {
    this->inValue = inValue;
}
BatchNorm2d::operator std::string() const {
    return getName() + " = ailang::batchnorm2d(" + getValue()->getName() +
           "):" + std::string(*getType());
}

WhileOp::WhileOp(const TypePtr &nodeType, const ModulePtr &condGraph,
                 const ModulePtr &bodyGraph, const std::vector<ValuePtr> &args)
    : Node(nodeType), cond(condGraph), body(bodyGraph), inits(std::move(args)) {
    if (nodeType->isTupleType()) {
        auto types = asType<TupleType>(nodeType)->getTypes();
        for (const auto &type : types) {
            outs.push_back(Node::create(type));
        }
    } else {
        throw std::runtime_error("WhileOp output type must be a tuple type.");
    }
}

WhileOp::operator std::string() const {
    std::string result;
    std::string lhs;
    std::string indent = "\t\t";

    auto addIndent = [&indent](const std::string &str) {
        std::stringstream ss(str);
        std::string line;
        std::string result;
        while (std::getline(ss, line)) {
            result += indent + line + "\n";
        }
        return result;
    };

    lhs += "(";
    for (size_t i = 0; i < outs.size(); i++) {
        if (i == outs.size() - 1) {
            lhs += outs[i]->getName() + ") ";
        } else {
            lhs += outs[i]->getName() + ", ";
        }
    }
    result += lhs + " = ailang::while (";
    for (size_t i = 0; i < inits.size(); i++) {
        if (i == inits.size() - 1) {
            result += inits[i]->getName() + "): ";
        } else {
            result += inits[i]->getName() + ", ";
        }
    }
    result += getType()->getName();
    result += " {\n" + addIndent(std::string(*cond)) +
              addIndent(std::string(*body)) + "\n\t}";
    return result;
}

CompareOp::CompareOp(const TypePtr &nodeType, const ValuePtr &lhs,
                     const ValuePtr &rhs, CompareOp::CompareType op)
    : Node(nodeType) {
    this->lhs = lhs;
    this->rhs = rhs;
    this->op = op;
}

CompareOp::operator std::string() const {
    return getName() +
           " = ailang::" + compareOpString[static_cast<size_t>(op)] + "(" +
           lhs->getName() + ", " + rhs->getName() +
           "): " + std::string(*getType());
}

void CompareOp::accept(IRVisitor *visitor) { visitor->visit(this); }

IfOp::IfOp(const TypePtr &nodeType, const ModulePtr &trueBranch,
           const ModulePtr &falseBranch, const ValuePtr &cond)
    : Node(nodeType), trueBody(trueBranch), elseBody(falseBranch), cond(cond) {
    if (nodeType->isTupleType()) {
        auto types = asType<TupleType>(nodeType)->getTypes();
        for (const auto &type : types) {
            outs.push_back(Node::create(type));
        }
    } else {
        outs.push_back(Node::create(nodeType));
    }
}

IfOp::operator std::string() const {
    std::string result;
    std::string lhs;
    std::string indent = "\t\t";

    auto addIndent = [&indent](const std::string &str) {
        std::stringstream ss(str);
        std::string line;
        std::string result;
        while (std::getline(ss, line)) {
            result += indent + line + "\n";
        }
        return result;
    };

    for (size_t i = 0; i < outs.size(); i++) {
        if (i == outs.size() - 1) {
            lhs += outs[i]->getName();
        } else {
            lhs += outs[i]->getName() + ", ";
        }
    }
    result += lhs + " = ailang::if (";
    result += cond->getName() + ") : ";
    result += getType()->getName();
    result += " {\n" + addIndent(std::string(*trueBody)) + "\n\t} else {\n" +
              addIndent(std::string(*elseBody)) + "\n\t}";
    return result;
}

void IfOp::accept(IRVisitor *visitor) { visitor->visit(this); }

} // namespace ainl::ir