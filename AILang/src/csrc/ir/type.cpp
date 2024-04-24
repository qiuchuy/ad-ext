#include "ir/type.h"

#include "dtype.h"
#include "ir/literal.h"

namespace ainl::ir {

template <typename T> std::shared_ptr<T> SingletonTypePtr<T>::ptr = nullptr;

std::string TensorType::str() {
  std::stringstream ssm;
  size_t size = shape.size();
  ssm << "tensor<";
  for (size_t i = 0; i < size; i++) {
    ssm << shape[i]->getName() << "x";
  }
  ssm << elementType->getName() << ">";
  return ssm.str();
}

// true: <, false: >=
bool Type::compare(Type &other) {
  if (this->kind() > TypeKind::TensorType ||
      other.kind() > TypeKind::TensorType)
    // throw AINLError("Illegal type comparison.");
    ;
  TypePtr rhsPtr = other.getTypePtr();
  if (this->isTensorType() && rhsPtr->isTensorType()) {
    TypePtr thisBaseType =
        SAFE_TYPE_DOWNCAST(shared_from_this(), TensorType)->getElementType();
    TypePtr rhsBaseType =
        SAFE_TYPE_DOWNCAST(rhsPtr, TensorType)->getElementType();
    if (thisBaseType->kind() < rhsBaseType->kind()) {
      return true;
    } else {
      return false;
    }
  }
  if (this->isTensorType())
    return false;
  if (rhsPtr->isTensorType())
    return true;
  return this->kind() < rhsPtr->kind();
}

std::vector<int> TensorType::getConcreteShape() {
  auto shape = getShape();
  if (std::all_of(shape.begin(), shape.end(),
                  [](ValuePtr value) { return value->isLiteral(); })) {
    std::vector<int> concreteShape;
    for (const auto &value : shape) {
      assert(value->getType()->isIntType());
      concreteShape.push_back(
          (SAFE_VALUE_DOWNCAST(value, Literal)->getIntConcreteValue()));
    }
    return concreteShape;
  } else {
    // throw AINLError(
    // "Attempting to get concrete shape of a fully symbolic tensor.");
  }
}

DependentTupleType::DependentTupleType(const std::vector<ValuePtr> &values) {
  std::vector<TypePtr> valueTypes;
  for (const auto &value : values) {
    valueTypes.push_back(value->getType());
  }
  this->types = valueTypes;
  this->values = values;
}

bool LiteralType::equals(const Type &rhs) {
  return value->getType()->equals(rhs);
}

LiteralType::LiteralType(const ValuePtr &value) { this->value = value; }

std::string LiteralType::str() { return value->getType()->getName(); }

ValuePtr LiteralType::getValue() { return value; }

TypePtr DtypeToTypePtr(core::Dtype dtype) {
  switch (dtype.type) {
  case core::Dtype::DataType::Int32Type:
    return IntTypePtr::get();
  case core::Dtype::DataType::Float32Type:
    return FloatTypePtr::get();
  default:
      // throw AINLError("Unsupported dtype.");
      ;
  }
}

} // namespace ainl::ir