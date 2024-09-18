#include "ailang/Transforms/Autodiff.h"
#include "ailang/AST/ASTNode.h"
#include "ailang/IR/Container.h"
#include "ailang/IR/Function.h"
#include "ailang/IR/Literal.h"
#include "ailang/IR/Node.h"

using namespace ainl::ir;

void ainl::ir::autodiffOnModule(ModulePtr M) {
  auto Autodiff = std::make_unique<ForwardDiff>(M);
  LOG_DEBUG("%s", "Running autodiff on module");
  Autodiff->run(M);
  LOG_DEBUG("%s", "Autodiff finished");
}

void ForwardDiff::run(ModulePtr M) {
  auto Graph = M->getGraph();
  for (auto *Block : *Graph) {
    for (auto *Node : *Block) {
      setTangent(Node, nullptr);
    }
  }
  for (auto *Block : *Graph) {
    for (auto *Node : *Block) {
      if (isNonLinearNode(Node))
        Node->accept(this);
    }
  }
}

void ForwardDiff::visit(ParamPtr Node) {
  auto Params = Node->getParams();
  auto NumParams = Params.size();
  for (size_t Idx = 0; Idx < NumParams; ++Idx) {
    auto Type = Params[Idx]->getType();
    auto *Tangent = Graph::GraphParam::create(Type, Idx + NumParams);
    setTangent(Params[Idx], Tangent);
    Node->addParam(Tangent, Type, Idx + NumParams);
  }
}

void ForwardDiff::visit(ReturnOpPtr Node) {
  auto *Value = Node->getReturnValue();
  if (asValueType<TupleContainer>(Value)) {
    auto *Tuple = asValueType<TupleContainer>(Value);
    std::vector<ValuePtr> Items;
    std::vector<ValuePtr> Tangents;
    for (auto *Item : Tuple->getValues()) {
      Items.push_back(Item);
      auto *Tangent = getTangent(Item);
      Tangents.push_back(Tangent);
    }
    std::vector<ValuePtr> Values;
    for (auto *Item : Items) {
      Values.push_back(Item);
    }
    for (auto *Item : Tangents) {
      Values.push_back(Item);
    }
    auto *Container = TupleContainer::create(Values);
    Node->setReturnValue(Container);
    Module->setReturnType(Container->getType());
  } else {
    auto *Tangent = getTangent(Value);
    auto *Container = TupleContainer::create({Value, Tangent});
    Node->setReturnValue(Container);
    Module->setReturnType(Container->getType());
  }
}

void ForwardDiff::visit(ExpPtr Node) {
  auto *Value = Node->getOperand(0);
  auto *Tangent = getTangent(Value);
  auto *TangentNode =
      Module->createAfter<Mul>(Node, Node->getType(), Value, Tangent);
  setTangent(Node, TangentNode);
}

void ForwardDiff::visit(AddPtr Node) {
  auto *Left = Node->getOperand(0);
  auto *Right = Node->getOperand(1);
  auto *LeftTangent = getTangent(Left);
  auto *RightTangent = getTangent(Right);
  auto *TangentNode = Module->createAfter<Add>(Node, Node->getType(),
                                               LeftTangent, RightTangent);
  setTangent(Node, TangentNode);
}

void ForwardDiff::visit(NegPtr Node) {
  auto *Value = Node->getOperand(0);
  auto *Tangent = getTangent(Value);
  auto *TangentNode = Module->createAfter<Neg>(Node, Node->getType(), Tangent);
  setTangent(Node, TangentNode);
}

void ForwardDiff::visit(DivPtr Node) {
  auto *Left = Node->getOperand(0);
  auto *Right = Node->getOperand(1);
  auto *LeftTangent = getTangent(Left);
  auto *RightTangent = getTangent(Right);
  auto *UpperTangentNode =
      Module->createAfter<Div>(Node, Left->getType(), LeftTangent, Right);
  auto *SquareNode = Module->createAfter<Mul>(UpperTangentNode,
                                              Right->getType(), Right, Right);
  auto *DivNode = Module->createAfter<Div>(SquareNode, SquareNode->getType(),
                                           Left, SquareNode);
  auto *DivTangentNode = Module->createAfter<Mul>(
      DivNode, DivNode->getType(), RightTangent, DivNode);
  auto *LowerTangentNode =
      Module->createAfter<Neg>(DivTangentNode, DivNode->getType(), DivTangentNode);
  auto *TangentNode = Module->createAfter<Add>(
      LowerTangentNode, Node->getType(), UpperTangentNode, LowerTangentNode);
  setTangent(Node, TangentNode);
}

void ForwardDiff::visit(BroadcastPtr Node) {
  auto *Value = Node->getOperand(0);
  auto *Tangent = getTangent(Value);
  auto InputTensorType = asType<TensorType>(Value->getType());
  auto InputShape = InputTensorType->getConcreteShape();
  auto BroadCastShape = Node->getBroadCastShape();
  int InputSize = 1;
  for (auto Dim : InputShape) {
    InputSize *= Dim;
  }
  float BroadcastSize = 1;
  for (auto Dim : BroadCastShape) {
    BroadcastSize *= Dim;
  }
  float ScalingFactor = BroadcastSize / InputSize;
  std::vector<ValuePtr> ScalingFactors;
  for (size_t Idx = 0; Idx < InputShape.size(); ++Idx) {
    ScalingFactors.push_back(Literal::create(ScalingFactor));
  }
  auto *ConstantTensor = Module->createAfter<ConstantDef>(
      Node, Tangent->getType(), TupleContainer::create(ScalingFactors));
  auto *TangentNode = Module->createAfter<Mul>(
      ConstantTensor, Tangent->getType(), ConstantTensor, Tangent);
  setTangent(Node, TangentNode);
}