#include "ailang/Transforms/Autodiff.h"

namespace ainl::ir {

AutoDiffPass::AutoDiffPass(ModulePtr Module) : Module(Module) {}

void AutoDiffPass::init() {
  ForwardPattern =
      std::make_shared<ForwardDifferentialPattern>(shared_from_this());
  TransposePattern =
      std::make_shared<TransposeDifferentialPattern>(shared_from_this());
}

bool AutoDiffPass::isLinearizedDAGNode(NodePtr Node) {
  return std::find(LinearizedNodes.begin(), LinearizedNodes.end(), Node) !=
         LinearizedNodes.end();
}

void AutoDiffPass::run(ModulePtr Module) {
  runForwardDiff(Module);
  runTranspose(Module);
}

void AutoDiffPass::runForwardDiff(ModulePtr Module) {
  auto Graph = Module->getGraph();
  for (auto *Block : *Graph) {
    std::vector<NodePtr> Nodes;
    for (auto *Node : *Block) {
      Nodes.push_back(Node);
    }
    for (const auto &Node : Nodes) {
      Node->accept(ForwardPattern.get());
    }
  }
}

void AutoDiffPass::runTranspose(ModulePtr Module) {
  auto Graph = Module->getGraph();
  for (auto *Block : *Graph) {
    std::vector<NodePtr> Nodes;
    for (auto *Node : *Block) {
      if (isLinearizedDAGNode(Node))
        Nodes.push_back(Node);
    }
    std::reverse(Nodes.begin(), Nodes.end());
    for (const auto &Node : Nodes) {
      Node->accept(TransposePattern.get());
    }
  }
}

AutoDiffPass::ForwardDifferentialPattern::ForwardDifferentialPattern(
    std::shared_ptr<AutoDiffPass> Pass)
    : Pass(Pass) {}
AutoDiffPass::TransposeDifferentialPattern::TransposeDifferentialPattern(
    std::shared_ptr<AutoDiffPass> Pass)
    : Pass(Pass) {}

void autodiff(ModulePtr Module) {
  auto Pass = AutoDiffPass::create(Module);
  Pass->run(Module);
}

void AutoDiffPass::ForwardDifferentialPattern::setLinearRelation(
    ValuePtr Node, ValuePtr LinearNode) {
  if (Pass) {
    Pass->TangentTable[Node] = LinearNode;
  } else {
    throw std::runtime_error("AutoDiffPass is not initialized.");
  }
}

void AutoDiffPass::ForwardDifferentialPattern::setTransposeRelation(
    ValuePtr Node, ValuePtr TransposeNode) {
  if (Pass) {
    Pass->AdjointTable[Node] = TransposeNode;
  } else {
    throw std::runtime_error("AutoDiffPass is not initialized.");
  }
}

void AutoDiffPass::ForwardDifferentialPattern::addLinearizedNode(ValuePtr Node) {
  if (Pass) {
    Pass->LinearizedNodes.push_back(Node);
  } else {
    throw std::runtime_error("AutoDiffPass is not initialized.");
  }
}

ValuePtr
AutoDiffPass::ForwardDifferentialPattern::getLinearValue(ValuePtr Node) {
  if (Pass) {
    return Pass->TangentTable[Node];
  }
  throw std::runtime_error("AutoDiffPass is not initialized.");
}

void AutoDiffPass::ForwardDifferentialPattern::visit(NodePtr Node) {}
void AutoDiffPass::ForwardDifferentialPattern::visit(ParamPtr Node) {
  auto Params = Node->getParams();
  for (size_t i = 0; i < Params.size(); i++) {
    auto *LinearParam =
        new Graph::GraphParam(Params[i]->getType(), i + Params.size());
    Node->addParam(LinearParam, LinearParam->getType(), i + Params.size());
    addLinearizedNode(LinearParam);
    setLinearRelation(Params[i], LinearParam);
  }
}

void AutoDiffPass::ForwardDifferentialPattern::visit(ReturnOpPtr Node) {

  auto *Value = Node->getReturnValue();
  auto *Tangent = getLinearValue(Value);
  Node->setReturnValue(Tangent);
  addLinearizedNode(Node);
  setTransposeRelation(Tangent, Tangent);
}

void AutoDiffPass::ForwardDifferentialPattern::visit(TransposePtr Node) {
  auto *Value = Node->getValue();
  auto *Tangent = getLinearValue(Value);
  auto *LinearNode =
      Pass->Module->createAfter<Transpose>(Node, Tangent->getType(), Tangent);
  setLinearRelation(Node, LinearNode);
}

void AutoDiffPass::ForwardDifferentialPattern::visit(MatmulPtr Node) {
  auto *Left = Node->getLHS();
  auto *Right = Node->getRHS();
  auto *LeftLinear = getLinearValue(Left);
  auto *RightLinear = getLinearValue(Right);
  auto *LeftLinearNode = Pass->Module->createAfter<Matmul>(
      Node, LeftLinear->getType(), LeftLinear, Right);
  auto *RightLinearNode = Pass->Module->createAfter<Matmul>(
      LeftLinearNode, RightLinear->getType(), Left, RightLinear);
  auto *LinearNode =
      Pass->Module->createAfter<Add>(RightLinearNode, LeftLinearNode->getType(),
                                     LeftLinearNode, RightLinearNode);
  setLinearRelation(Node, LinearNode);
}
void AutoDiffPass::ForwardDifferentialPattern::visit(CompareOpPtr Node) {}
void AutoDiffPass::ForwardDifferentialPattern::visit(IfOpPtr Node) {}
void AutoDiffPass::ForwardDifferentialPattern::visit(AddPtr Node) {
  auto *Left = Node->getLHS();
  auto *Right = Node->getRHS();
  auto *LeftLinear = getLinearValue(Left);
  auto *RightLinear = getLinearValue(Right);
  auto *LeftLinearNode = Pass->Module->createAfter<Add>(
      Node, LeftLinear->getType(), LeftLinear, Right);
  auto *RightLinearNode = Pass->Module->createAfter<Add>(
      LeftLinearNode, RightLinear->getType(), Left, RightLinear);
  auto *LinearNode =
      Pass->Module->createAfter<Add>(RightLinearNode, LeftLinearNode->getType(),
                                     LeftLinearNode, RightLinearNode);
  setLinearRelation(Node, LinearNode);

}

void AutoDiffPass::TransposeDifferentialPattern::visit(NodePtr Node) {}
void AutoDiffPass::TransposeDifferentialPattern::visit(ParamPtr Node) {
  // nothing to do here
}
void AutoDiffPass::TransposeDifferentialPattern::visit(ReturnOpPtr Node) {
  // nothing to do here
}

void AutoDiffPass::TransposeDifferentialPattern::visit(TransposePtr Node) {
  auto *Value = Node->getValue();
  auto *Adjoint = Pass->AdjointTable[Value];
  auto *LinearTranspose =
      Pass->Module->create<Transpose>(Adjoint->getType(), Adjoint);
  Pass->AdjointTable[Node] = LinearTranspose;
}
void AutoDiffPass::TransposeDifferentialPattern::visit(MatmulPtr Node) {}
void AutoDiffPass::TransposeDifferentialPattern::visit(CompareOpPtr Node) {}
void AutoDiffPass::TransposeDifferentialPattern::visit(IfOpPtr Node) {}
void AutoDiffPass::TransposeDifferentialPattern::visit(AddPtr Node) {}

} // namespace ainl::ir