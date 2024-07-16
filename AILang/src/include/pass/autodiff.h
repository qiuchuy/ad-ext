#pragma once

#include "ir/function.h"
#include "ir/ir_visitor.h"
#include "pass/pass.h"
#include <memory>

namespace ainl::ir {

class AutoDiffPass : public Pass,
                     public std::enable_shared_from_this<AutoDiffPass> {
public:
  AutoDiffPass(ModulePtr Module);
  static std::shared_ptr<AutoDiffPass> create(ModulePtr Module) {
    auto Pass = std::make_shared<AutoDiffPass>(Module);
    Pass->init();
    return Pass;
  }
  void run(ModulePtr Module) override;

private:
  void init();
  bool isLinearizedDAGNode(NodePtr Node);
  class ForwardDifferentialPattern : public IRVisitor {
  public:
    ForwardDifferentialPattern(std::shared_ptr<AutoDiffPass> Pass);
    void visit(NodePtr Node) override;
    void visit(ParamPtr Node) override;
    void visit(ReturnOpPtr Node) override;
    void visit(TransposePtr Node) override;
    void visit(MatmulPtr Node) override;
    void visit(CompareOpPtr Node) override;
    void visit(IfOpPtr Node) override;
    void visit(AddPtr Node) override;

  private:
    void setLinearRelation(ValuePtr Node, ValuePtr LinearNode);
    void setTransposeRelation(ValuePtr Node, ValuePtr TransposeNode);
    void addLinearizedNode(ValuePtr Node);
    ValuePtr getLinearValue(ValuePtr Node);
    std::shared_ptr<AutoDiffPass> Pass;
  };

  class TransposeDifferentialPattern : public IRVisitor {
  public:
    TransposeDifferentialPattern(std::shared_ptr<AutoDiffPass> Pass);
    void visit(NodePtr Node) override;
    void visit(ParamPtr Node) override;
    void visit(ReturnOpPtr Node) override;
    void visit(TransposePtr Node) override;
    void visit(MatmulPtr Node) override;
    void visit(CompareOpPtr Node) override;
    void visit(IfOpPtr Node) override;
    void visit(AddPtr Node) override;

  private:
    std::shared_ptr<AutoDiffPass> Pass;
  };

  void runForwardDiff(ModulePtr Module);
  void runTranspose(ModulePtr Module);
  std::map<ValuePtr, ValuePtr> TangentTable;
  std::map<ValuePtr, ValuePtr> AdjointTable;
  std::vector<ValuePtr> LinearizedNodes;
  std::shared_ptr<ForwardDifferentialPattern> ForwardPattern;
  std::shared_ptr<TransposeDifferentialPattern> TransposePattern;
  ModulePtr Module;
};

void autodiff(ModulePtr Module);

} // namespace ainl::ir