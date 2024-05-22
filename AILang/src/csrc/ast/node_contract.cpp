#include "ast/node_contract.h"

namespace ainl::ir {

ValuePtr reluNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                          const ValuePtr &inValue) {
  if (!inValue->getType()->isTensorType()) {
    throw ainl::core::AINLError("relu operator only applies to tensors.");
  }
  return module->getGraph()->create<Relu>(nodeType, inValue);
}

ValuePtr transposeNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                               const ValuePtr &inValue) {
  if (!inValue->getType()->isTensorType()) {
    throw ainl::core::AINLError("transpose operator only applies to tensors.");
  }
  return module->getGraph()->create<Transpose>(nodeType, inValue);
}
ValuePtr matmulNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                            const ValuePtr &lhs, const ValuePtr &rhs) {
  // Type Checking
  if (!lhs->getType()->isTensorType() || !rhs->getType()->isTensorType()) {
    throw ainl::core::AINLError("matmul operator only applies to two tensors.");
  }
  // Construct the Result Node
  return module->getGraph()->create<Matmul>(nodeType, lhs, rhs);
}

ValuePtr maxpool2dNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                               const ValuePtr &inValue) {
  if (!inValue->getType()->isTensorType()) {
    throw ainl::core::AINLError("maxpool2d operator only applies to tensors.");
  }
  return module->getGraph()->create<Maxpool2d>(nodeType, inValue);
}
ValuePtr convolutionNodeContract(const ModulePtr &module,
                                 const TypePtr &nodeType,
                                 const ValuePtr &inValue) {
  if (!inValue->getType()->isTensorType()) {
    throw ainl::core::AINLError(
        "convolution operator only applies to tensors.");
  }
  return module->getGraph()->create<Convolution>(nodeType, inValue);
}
ValuePtr batchnorm2dNodeContract(const ModulePtr &module,
                                 const TypePtr &nodeType,
                                 const ValuePtr &inValue) {
  if (!inValue->getType()->isTensorType()) {
    throw ainl::core::AINLError(
        "batchnorm2d operator only applies to tensors.");
  }
  return module->getGraph()->create<BatchNorm2d>(nodeType, inValue);
}

NodeContract::NodeContract() {
  registerContract("matmul", [](const ModulePtr &module,
                                const TypePtr &nodeType,
                                std::vector<ValuePtr> args) {
    if (args.size() != 2) {
      throw ainl::core::AINLError(
          "Invalid argument number for operator matmul");
    }
    return matmulNodeContract(module, nodeType, (args[0]), (args[1]));
  });
  registerContract("relu", [](const ModulePtr &module, const TypePtr &nodeType,
                              std::vector<ValuePtr> args) {
    if (args.size() != 1) {
      throw ainl::core::AINLError("Invalid argument number for operator relu");
    }
    return reluNodeContract(module, nodeType, (args[0]));
  });
  registerContract("transpose",
                   [](const ModulePtr &module, const TypePtr &nodeType,
                      std::vector<ValuePtr> args) {
                     if (args.size() != 1) {
                       throw ainl::core::AINLError(
                           "Invalid argument number for operator transpose");
                     }
                     return transposeNodeContract(module, nodeType, (args[0]));
                   });
  registerContract("maxpool2d",
                   [](const ModulePtr &module, const TypePtr &nodeType,
                      std::vector<ValuePtr> args) {
                     if (args.size() != 1) {
                       throw ainl::core::AINLError(
                           "Invalid argument number for operator maxpool2d");
                     }
                     return maxpool2dNodeContract(module, nodeType, (args[0]));
                   });
  registerContract("convolution", [](const ModulePtr &module,
                                     const TypePtr &nodeType,
                                     std::vector<ValuePtr> args) {
    if (args.size() != 1) {
      throw ainl::core::AINLError(
          "Invalid argument number for operator convolution");
    }
    return convolutionNodeContract(module, nodeType, (args[0]));
  });
  registerContract("batchnorm2d", [](const ModulePtr &module,
                                     const TypePtr &nodeType,
                                     std::vector<ValuePtr> args) {
    if (args.size() != 1) {
      throw ainl::core::AINLError(
          "Invalid argument number for operator batchnorm2d");
    }
    return batchnorm2dNodeContract(module, nodeType, (args[0]));
  });
}

NodeContract &getNodeContract() {
  static NodeContract contract;
  return contract;
}

ValuePtr resolveContract(const std::string &name, ModulePtr module,
                         TypePtr nodeType, std::vector<ValuePtr> args) {
  return getNodeContract().resolveContract(name, module, nodeType, args);
}
} // namespace ainl::ir