#include "type.h"

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
bool Type::compare(const Type &other) {
    if (this->kind() > TypeKind::TensorType ||
        other.kind() > TypeKind::TensorType)
        throw AINLError("Illegal type comparison.");

    Type rhsType = other;
    TypePtr rhsPtr = rhsType.getTypePtr();
    if (this->isTensorType() && rhsPtr->isTensorType()) {
        TypePtr thisBaseType =
            SAFE_TYPE_DOWNCAST(shared_from_this(), TensorType)
                ->getElementType();
        TypePtr rhsBaseType =
            SAFE_TYPE_DOWNCAST(rhsPtr, TensorType)->getElementType();
        if (thisBaseType < rhsBaseType) {
            return true;
        } else {
            return false;
        }
    }
    if (this->isTensorType())
        return false;
    if (rhsPtr->isTensorType())
        return true;
    return (*this < rhsType);
}
