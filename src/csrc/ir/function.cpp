#include "function.h"

#include <utility>

ALModule::ALModule(std::string name, const TypePtr &inputType,
                   const TypePtr &returnType) {
    this->graph = std::make_shared<Graph>(name, inputType);
    for (size_t i = 0; i < graph->getParams().size(); i++) {
        graph->getParams()[i]->graph = graph;
    }
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
