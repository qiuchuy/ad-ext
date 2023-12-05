#ifndef AINL_SRC_INCLUDE_LITERAL_H
#define AINL_SRC_INCLUDE_LITERAL_H

#include <variant>

#include "value.h"

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
    float getFloatConcreteValue() {
        assert(type->isFloatType());
        throw AINLError(
            "Attempting to get a concrete float value from an int Literal.");
    }
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
        throw AINLError("Unknown literal type.");
    }

  private:
    std::variant<int, float, bool> value;
};

#endif // AINL_SRC_INCLUDE_LITERAL_H
