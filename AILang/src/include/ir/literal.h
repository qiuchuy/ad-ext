#pragma once

#include <variant>

#include "ir/value.h"
namespace ainl::ir {

class Literal;
using LiteralPtr = Literal *;
class Literal : public Value {
public:
  enum class LiteralType {
    Int,
    Float,
    Bool,
  };

  explicit Literal(int intValue) : Value(IntTypePtr::get()) {
    std::get<(size_t)LiteralType::Int>(value) = intValue;
  }
  explicit Literal(float floatValue) : Value(FloatTypePtr::get()) {
    std::get<(size_t)LiteralType::Float>(value) = floatValue;
  }
  explicit Literal(bool boolValue) : Value(BoolTypePtr::get()) {
    std::get<(size_t)LiteralType::Bool>(value) = boolValue;
  }

  static LiteralPtr create(int value) { return new Literal(value); }

  static LiteralPtr create(float value) { return new Literal(value); }

  static LiteralPtr create(bool value) { return new Literal(value); }

  int getIntConcreteValue() {
    assert(type->isIntType());
    return std::get<(size_t)LiteralType::Int>(value);
  }
  float getFloatConcreteValue() { assert(type->isFloatType()); }
  bool getBoolConcreteValue() {
    assert(type->isBoolType());
    return std::get<(size_t)LiteralType::Bool>(value);
  }
  std::string getName() const override {
    if (type->isIntType())
      return std::to_string(std::get<(size_t)LiteralType::Int>(value));
    if (type->isFloatType())
      return std::to_string(std::get<(size_t)LiteralType::Float>(value));
    if (type->isBoolType())
      return std::to_string(std::get<(size_t)LiteralType::Bool>(value));
  }

  Value::ValueKind getValueKind() const override {
    return Value::ValueKind::Literal;
  }

private:
  std::variant<int, float, bool> value;
};
} // namespace ainl::ir
