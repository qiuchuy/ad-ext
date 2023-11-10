#include "value.h"

#include <utility>

#include "use.h"

Value::Value() {
    beginUse = new Use();
    endUse = new Use();
    beginUse->setNext(endUse);
    endUse->setPrev(beginUse);
}

Value::Value(const TypePtr &type) {
    beginUse = new Use();
    endUse = new Use();
    this->type = type;
    beginUse->setNext(endUse);
    endUse->setPrev(beginUse);
}

Value::Value(const std::vector<TypePtr> &types) {
    this->type = TupleType::createUnnamedTuple(types);
    for (const auto &type : types) {
        auto value = new Value(type);
        values.push_back(value);
    }
}

void Value::insertUseAtEnd(UsePtr use) { endUse->insertBefore(use); }

std::string Value::getName() const {
    return std::string(*type) + " " + prefix + name;
}

TypePtr createTypePtrForValues(const std::vector<ValuePtr> &values) {
    if (values.size() > 1) {
        std::vector<std::string> names;
        std::vector<TypePtr> types;
        for (const ValuePtr &value : values) {
            names.push_back(value->getName());
            types.push_back(value->getType());
        }
        return TupleType::createNamedTuple(types, names);
    }
    if (values.size() == 1)
        return values[0]->getType();
    return VoidTypePtr::get();
}
