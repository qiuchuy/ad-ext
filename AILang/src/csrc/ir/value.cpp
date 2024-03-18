
#include <utility>

#include "ir/value.h"
#include "ir/use.h"

namespace ainl::ir {

int Value::valueNum = 0;
std::string Value::LOCAL_PREFIX = "%";
std::string Value::LOCAL_NAME_PREFIX = "v";
std::string Value::FPARAM_NAME_PREFIX = "f";

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
    beginUse = new Use();
    endUse = new Use();
    this->type = TupleType::createUnnamedTuple(types);
    for (const auto &inType : types) {
        auto value = new Value(inType);
        values.push_back(value);
    }
}

bool Value::operator==(const Value &other) const {
    if (!this->isLiteral() || !other.isLiteral())
        return false;
    if (this->getType()->kind() != other.getType()->kind())
        return false;
    return this->getName() == other.getName();
}

bool Value::operator!=(const Value &other) const { return !(*this == other); }

void Value::insertUseAtEnd(UsePtr use) { endUse->insertBefore(use); }

std::string Value::getName() const { return prefix + name; }

TypePtr createTypePtrForValues(const std::vector<Value *> &values) {
    if (values.size() > 1) {
        std::vector<std::string> names;
        std::vector<TypePtr> types;
        for (const Value *value : values) {
            names.push_back(value->getName());
            types.push_back(value->getType());
        }
        return TupleType::createNamedTuple(types, names);
    }
    if (values.size() == 1)
        return values[0]->getType();
    return VoidTypePtr::get();
}

} // namespace ainl::ir