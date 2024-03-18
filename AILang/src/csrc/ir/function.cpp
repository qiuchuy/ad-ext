#include "ir/function.h"
#include <utility>

namespace ainl::ir {

ALModule::ALModule(std::string name, const TypePtr &inputType,
                   const TypePtr &returnType) {
    this->graph = std::make_shared<Graph>(name);
    std::vector<ValuePtr> params;
    if (inputType->isTupleType()) {
        std::vector<TypePtr> paramTypes =
            SAFE_TYPE_DOWNCAST(inputType, TupleType)->getTypes();
        for (int idx = 0; (size_t)idx < paramTypes.size(); idx++) {
            auto param = new Graph::GraphParam(paramTypes[idx], idx);
            params.push_back(param);
        }
    } else {
        params.push_back(new Graph::GraphParam(inputType, 0));
    }
    for (auto &param : params) {
        param->graph = graph;
    }
    ParamPtr paramNode = Param::create(params, inputType);
    paramNode->graph = this->graph;
    paramNode->addBlockWithParam(paramNode);
    this->signature = new Signature(inputType, returnType);
    this->name = std::move(name);
}

std::vector<ValuePtr> ALModule::getParams() { return graph->getParams(); }

std::string ALModule::str() {
    std::stringstream ss;
    for (size_t i = 0; i < getParams().size(); ++i) {
        ss << std::string(*getParams()[i]);
        if (i != getParams().size() - 1) {
            ss << ", ";
        }
    }
    std::string paramList = ss.str();
    std::string str;
    str.append("define @graph.")
        .append(name)
        .append("(")
        .append(paramList)
        .append(") : ")
        .append(std::string(*signature))
        .append(" {\n");
    str.append(graph->str());
    str.append("}\n");
    return str;
}
} // namespace ainl::ir