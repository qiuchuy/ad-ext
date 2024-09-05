#pragma once

#include <variant>

#include "ailang/IR/Value.h"
namespace ainl::ir {

class Literal;
using LiteralPtr = Literal *;
class Literal : public Value {
public:
  enum class LiteralType { Int = 0, Float = 1, Bool = 2 };

  explicit Literal(int intValue) : Value(IntTypePtr::get()), value(intValue) {}

  explicit Literal(float floatValue)
      : Value(FloatTypePtr::get()), value(floatValue) {}

  explicit Literal(bool boolValue)
      : Value(BoolTypePtr::get()), value(boolValue) {}

  static LiteralPtr create(int value) { return new Literal(value); }
  static LiteralPtr create(float value) { return new Literal(value); }
  static LiteralPtr create(bool value) { return new Literal(value); }

  int getIntConcreteValue() {
    assert(type->isIntType());
    return std::get<int>(value);
  }

  float getFloatConcreteValue() {
    assert(type->isFloatType());
    return std::get<float>(value);
  }

  bool getBoolConcreteValue() {
    assert(type->isBoolType());
    return std::get<bool>(value);
  }

  std::string getName() const override {
    if (type->isIntType())
      return std::to_string(std::get<int>(value));
    if (type->isFloatType())
      return std::to_string(std::get<float>(value));
    if (type->isBoolType())
      return std::to_string(std::get<bool>(value));
    return ""; // Add a default return
  }

  Value::ValueKind getValueKind() const override {
    return Value::ValueKind::Literal;
  }

private:
  std::variant<int, float, bool> value;
};
} // namespace ainl::ir
