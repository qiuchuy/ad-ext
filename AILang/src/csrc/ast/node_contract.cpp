#include "ast/node_contract.h"
#include "ast/ast_node.h"
#include "ir/node.h"
#include "ir/value.h"

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
        throw ainl::core::AINLError(
            "transpose operator only applies to tensors.");
    }
    return module->getGraph()->create<Transpose>(nodeType, inValue);
}
ValuePtr matmulNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                            const ValuePtr &lhs, const ValuePtr &rhs) {
    // Type Checking
    if (!lhs->getType()->isTensorType() || !rhs->getType()->isTensorType()) {
        throw ainl::core::AINLError(
            "matmul operator only applies to two tensors.");
    }
    // Construct the Result Node
    return module->getGraph()->create<Matmul>(nodeType, lhs, rhs);
}
ValuePtr addNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                         const ValuePtr &lhs, const ValuePtr &rhs) {
    // Type Checking
    if (!lhs->getType()->isTensorType() || !rhs->getType()->isTensorType()) {
        throw ainl::core::AINLError(
            "add operator only applies to two tensors.");
    }
    // Construct the Result Node
    return module->getGraph()->create<Add>(nodeType, lhs, rhs);
}
ValuePtr maxpool2dNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                               const ValuePtr &inValue) {
    if (!inValue->getType()->isTensorType()) {
        throw ainl::core::AINLError(
            "maxpool2d operator only applies to tensors.");
    }
    return module->getGraph()->create<Maxpool2d>(nodeType, inValue);
}
ValuePtr convolutionNodeContract(const ModulePtr &module,
                                 const TypePtr &nodeType,
                                 const ValuePtr &inputValue,
                                 const ValuePtr &weightValue) {
    if (!inputValue->getType()->isTensorType() ||
        !weightValue->getType()->isTensorType()) {
        throw ainl::core::AINLError(
            "convolution operator only applies to tensors.");
    }
    return module->getGraph()->create<Convolution>(nodeType, inputValue,
                                                   weightValue);
}
ValuePtr batchnorm2dNodeContract(const ModulePtr &module,
                                 const TypePtr &nodeType,
                                 const std::vector<ValuePtr> &inValues) {
    for (auto inValue : inValues) {
        if (!inValue->getType()->isTensorType()) {
            throw ainl::core::AINLError(
                "batchnorm2d operator only applies to special tensors.");
        }
    }

    return module->getGraph()->create<BatchNorm2d>(nodeType, inValues[0],
                                                   inValues[1], inValues[2],
                                                   inValues[3], inValues[4]);
}

ValuePtr whileLoopNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                               const ModulePtr &condGraph,
                               const ModulePtr &bodyGraph,
                               std::vector<ValuePtr> args) {
    return module->getGraph()->create<WhileOp>(nodeType, condGraph, bodyGraph,
                                               args);
}

ValuePtr ifNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                        const ModulePtr &trueModule,
                        const ModulePtr &falseModule, const ValuePtr &cond) {
    return module->getGraph()->create<IfOp>(nodeType, trueModule, falseModule,
                                            cond);
}

ValuePtr compareNodeContract(const ModulePtr &module, const TypePtr &nodeType,
                             const ValuePtr &lhs, const ValuePtr &rhs,
                             CompareOp::CompareType op) {
    return module->getGraph()->create<CompareOp>(nodeType, lhs, rhs, op);
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
    registerContract("add", [](const ModulePtr &module, const TypePtr &nodeType,
                               std::vector<ValuePtr> args) {
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator add");
        }
        return addNodeContract(module, nodeType, (args[0]), (args[1]));
    });
    registerContract("relu",
                     [](const ModulePtr &module, const TypePtr &nodeType,
                        std::vector<ValuePtr> args) {
                         if (args.size() != 1) {
                             throw ainl::core::AINLError(
                                 "Invalid argument number for operator relu");
                         }
                         return reluNodeContract(module, nodeType, (args[0]));
                     });
    registerContract("transpose", [](const ModulePtr &module,
                                     const TypePtr &nodeType,
                                     std::vector<ValuePtr> args) {
        if (args.size() != 1) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator transpose");
        }
        return transposeNodeContract(module, nodeType, (args[0]));
    });

    registerContract("maxpool2d", [](const ModulePtr &module,
                                     const TypePtr &nodeType,
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
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator convolution");
        }
        return convolutionNodeContract(module, nodeType, (args[0]), (args[1]));
    });
    registerContract("batchnorm2d", [](const ModulePtr &module,
                                       const TypePtr &nodeType,
                                       std::vector<ValuePtr> args) {
        if (args.size() != 5) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator batchnorm2d");
        }
        return batchnorm2dNodeContract(module, nodeType, {args[0],args[1],args[2],args[3],args[4]});
    });
    registerContract("loop", [](const ModulePtr &module,
                                const TypePtr &nodeType,
                                std::vector<ValuePtr> args) {
        if (args.size() < 2) {
            throw ainl::core::AINLError(
                "Expect at least cond graph and body graph for loop ir node.");
        }
        auto bodyGraph =
            std::shared_ptr<ALModule>(asValueType<ALModule>(*args.rbegin()));
        args.pop_back();
        auto condGraph =
            std::shared_ptr<ALModule>(asValueType<ALModule>(*args.rbegin()));
        args.pop_back();
        return whileLoopNodeContract(module, nodeType, condGraph, bodyGraph,
                                     args);
    });
    registerContract("ifop", [](const ModulePtr &module,
                                const TypePtr &nodeType,
                                std::vector<ValuePtr> args) {
        if (args.size() != 3) {
            throw ainl::core::AINLError("Expect three arguments for if "
                                        "operator: true branch, false branch, "
                                        "and condition.");
        }
        auto falseModule =
            std::shared_ptr<ALModule>(asValueType<ALModule>(*args.rbegin()));
        args.pop_back();
        auto trueModule =
            std::shared_ptr<ALModule>(asValueType<ALModule>(*args.rbegin()));
        args.pop_back();
        auto ifCond = *args.rbegin();
        args.pop_back();
        return ifNodeContract(module, nodeType, trueModule, falseModule,
                              ifCond);
    });
    registerContract("eq", [](const ModulePtr &module, const TypePtr &nodeType,
                              std::vector<ValuePtr> args) {
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator eq");
        }
        return compareNodeContract(module, nodeType, (args[0]), (args[1]),
                                   CompareOp::CompareType::EQ);
    });
    registerContract("ne", [](const ModulePtr &module, const TypePtr &nodeType,
                              std::vector<ValuePtr> args) {
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator ne");
        }
        return compareNodeContract(module, nodeType, (args[0]), (args[1]),
                                   CompareOp::CompareType::NE);
    });
    registerContract("lt", [](const ModulePtr &module, const TypePtr &nodeType,
                              std::vector<ValuePtr> args) {
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator lt");
        }
        return compareNodeContract(module, nodeType, (args[0]), (args[1]),
                                   CompareOp::CompareType::LT);
    });
    registerContract("le", [](const ModulePtr &module, const TypePtr &nodeType,
                              std::vector<ValuePtr> args) {
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator le");
        }
        return compareNodeContract(module, nodeType, (args[0]), (args[1]),
                                   CompareOp::CompareType::LE);
    });
    registerContract("gt", [](const ModulePtr &module, const TypePtr &nodeType,
                              std::vector<ValuePtr> args) {
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator gt");
        }
        return compareNodeContract(module, nodeType, (args[0]), (args[1]),
                                   CompareOp::CompareType::GT);
    });
    registerContract("ge", [](const ModulePtr &module, const TypePtr &nodeType,
                              std::vector<ValuePtr> args) {
        if (args.size() != 2) {
            throw ainl::core::AINLError(
                "Invalid argument number for operator ge");
        }
        return compareNodeContract(module, nodeType, (args[0]), (args[1]),
                                   CompareOp::CompareType::GE);
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