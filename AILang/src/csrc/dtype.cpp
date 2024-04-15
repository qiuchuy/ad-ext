#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "dtype.h"

namespace ainl::core {

bool Dtype::operator<(const Dtype &other) const {
  return static_cast<int>(type) < static_cast<int>(other.type);
}

template <> TypeToDtype<bool>::operator Dtype() {
  return Dtype(Dtype::DataType::BoolType);
}

template <> TypeToDtype<int>::operator Dtype() {
  return Dtype(Dtype::DataType::Int32Type);
}

template <> TypeToDtype<float>::operator Dtype() {
  return Dtype(Dtype::DataType::Float32Type);
}

template <> TypeToDtype<void>::operator Dtype() {
  return Dtype(Dtype::DataType::VoidType);
}

size_t dtypeSize(Dtype dtype) {

  switch (dtype.type) {
  case Dtype::DataType::BoolType:
    return sizeof(bool);
  case Dtype::DataType::Int8Type:
    return sizeof(int8_t);
  case Dtype::DataType::Int16Type:
    return sizeof(int16_t);
  case Dtype::DataType::Int32Type:
    return sizeof(int32_t);
  case Dtype::DataType::Int64Type:
    return sizeof(int64_t);
  case Dtype::DataType::Float32Type:
    return sizeof(float);
  case Dtype::DataType::Float64Type:
    return sizeof(double);
  default:
    throw std::invalid_argument("Invalid dtype.");
  }
}

std::string Dtype::toString() const {
  switch (type) {
  case DataType::BoolType:
    return "bool";
  case DataType::Int8Type:
    return "int8";
  case DataType::Int16Type:
    return "int16";
  case DataType::Int32Type:
    return "int32";
  case DataType::Int64Type:
    return "int64";
  case DataType::Float32Type:
    return "float32";
  case DataType::Float64Type:
    return "float64";
  default:
    return "unknown";
  }
}

Dtype getDtypeFromFormat(const std::string &formatStr) {
  if (formatStr == "b" || formatStr == "B") {
    return Int8;
  } else if (formatStr == "h" || formatStr == "H") {
    return Int16;
  } else if (formatStr == "i" || formatStr == "I") {
    return Int32;
  } else if (formatStr == "l" || formatStr == "L") {
    return Int64;
  } else if (formatStr == "f") {
    return Float32;
  } else if (formatStr == "d") {
    return Float64;
  } else {
    throw std::runtime_error("Unsupported data type: " + formatStr);
  }
}

std::ostream &operator<<(std::ostream &os, const Dtype &dtype) {
#include <ostream> // Include the <ostream> header

  switch (dtype.type) {
  case Dtype::DataType::BoolType:
    os << "bool";
    break;
  case Dtype::DataType::Int8Type:
    os << "int8";
    break;
  case Dtype::DataType::Int16Type:
    os << "int16";
    break;
  case Dtype::DataType::Int32Type:
    os << "int32";
    break;
  case Dtype::DataType::Int64Type:
    os << "int64";
    break;
  case Dtype::DataType::Float32Type:
    os << "float32";
    break;
  case Dtype::DataType::Float64Type:
    os << "float64";
    break;
  default:
    os << "unknown";
    break;
  }
  return os;
}

} // namespace ainl::core