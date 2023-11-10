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
