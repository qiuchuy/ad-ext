#include "ailang/Transforms/Autodiff.h"
#include "ailang/IR/Function.h"

using namespace ainl::ir;

void ainl::ir::autodiffOnModule(ModulePtr M) {
  auto Autodiff = std::make_unique<AutodiffPass>();
  Autodiff->run(M);
}

void AutodiffPass::run(ModulePtr M) {
  auto Graph = M->getGraph();
  for (auto *Block : *Graph) {
    for (auto *Node : *Block) {
      Node->accept(this);
    }
  }
}

void AutodiffPass::visit(ParamPtr Node) {
  auto Params = Node->getParams();
  auto NumParams = Params.size();
  for (size_t Idx = 0; Idx < NumParams; ++Idx) {
    auto Type = Params[Idx]->getType();
    auto *Tangent = Graph::GraphParam::create(Type, Idx + NumParams);
    TangentMap[Params[Idx]] = Tangent;
    Node->addParam(Tangent, Type, Idx + NumParams);
  }
}

void AutodiffPass::visit(ReturnOpPtr Node) {
  auto *Value = Node->getReturnValue();
  auto *Tangent = TangentMap[Value];
  assert(Tangent && "[autodiff] Tangent not found when visiting ReturnOp");
  Node->setReturnValue(Tangent);
}