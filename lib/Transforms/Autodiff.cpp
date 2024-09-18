#include "ailang/Transforms/Autodiff.h"
#include "ailang/AST/ASTNode.h"
#include "ailang/IR/Container.h"
#include "ailang/IR/Function.h"

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
    TangentMap[Params[Idx]] = Tangent;
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
      auto *Tangent = TangentMap[Item];
      assert(Tangent && "[autodiff] Tangent not found when visiting ReturnOp");
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
    auto *Tangent = TangentMap[Value];
    assert(Tangent && "[autodiff] Tangent not found when visiting ReturnOp");
    auto *Container = TupleContainer::create({Value, Tangent});
    Node->setReturnValue(Container);
    Module->setReturnType(Container->getType());
  }
}