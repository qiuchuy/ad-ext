#include "dtype.h"

namespace ainl::core {

bool Dtype::operator<(const Dtype &other) const {
    return static_cast<int>(type) < static_cast<int>(other.type);
}

template <> TypeToDtype<bool>::operator Dtype() {
    return Dtype(Dtype::DataType::BoolType);
}

template <> TypeToDtype<int>::operator Dtype() {
    return Dtype(Dtype::DataType::IntType);
}

template <> TypeToDtype<float>::operator Dtype() {
    return Dtype(Dtype::DataType::FloatType);
}

template <> TypeToDtype<void>::operator Dtype() {
    return Dtype(Dtype::DataType::VoidType);
}

} // namespace ainl::core