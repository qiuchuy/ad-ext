#include "ir/function.h"

#include <utility>

namespace ainl::ir {

ALModule::ALModule(std::string name, const TypePtr &inputType,
                   const TypePtr &returnType) {
  this->graph = Graph::create(inputType, returnType);
  this->name = name;
  this->signature = new Signature(inputType, returnType);
}

std::vector<ValuePtr> ALModule::getParams() { return graph->getParams(); }

std::vector<TypePtr> ALModule::getParamTypes() {
  std::vector<TypePtr> paramTypes;
  for (auto &param : getParams()) {
    paramTypes.push_back(param->getType());
  }
  return paramTypes;
}

TypePtr ALModule::getReturnType() { return signature->returnType; }

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
      .append(std::string(*signature) + " ");
  str.append(graph->str());
  return str;
}
} // namespace ainl::ir