#pragma once

#include "ailang/IR/Value.h"

namespace ainl::ir {

class TupleContainer : public Value {
public:
  explicit TupleContainer(const std::vector<ValuePtr> &values,
                          const TypePtr &type)
      : Value(type), values(values) {}
  static TupleContainer *create(const std::vector<ValuePtr> &values) {
    std::vector<TypePtr> types;
    TypePtr type;
    for (const auto &value : values) {
      types.push_back(value->getType());
    }
    type = TupleType::createUnnamedTuple(types);
    return new TupleContainer(values, type);
  }
  std::vector<ValuePtr> getValues() { return values; }
  Value::ValueKind getValueKind() const override { return ValueKind::Tuple; }
  std::string getName() const override {
    std::string name = "(";
    for (size_t i = 0; i < values.size(); i++) {
      name += values[i]->getName();
      if (i != values.size() - 1) {
        name += ", ";
      }
    }
    name += ")";
    return name;
  }

private:
  std::vector<ValuePtr> values;
};

using TupleContainerPtr = TupleContainer *;

} // namespace ainl::ir