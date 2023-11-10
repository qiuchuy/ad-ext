#ifndef AINL_SRC_INCLUDE_TYPE_H
#define AINL_SRC_INCLUDE_TYPE_H

#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "logger.h"
#include "value.h"

class Value;
using ValuePtr = Value *;

#define SAFE_TYPE_DOWNCAST(shared_ptr, derived_type)                           \
    std::dynamic_pointer_cast<derived_type>(shared_ptr)

template <typename T> class SingletonTypePtr {
  public:
    static std::shared_ptr<T> get() {
        if (SingletonTypePtr::ptr == nullptr) {
            SingletonTypePtr::ptr = std::make_shared<T>();
            return SingletonTypePtr::ptr;
        } else {
            return SingletonTypePtr::ptr;
        }
    }

  public:
    SingletonTypePtr(SingletonTypePtr const &) = delete;
    void operator=(SingletonTypePtr const &) = delete;
    static std::shared_ptr<T> ptr;
};

// Type: Basic class in Type hierarchy
class Type;
using TypePtr = std::shared_ptr<Type>;
class Type : public std::enable_shared_from_this<Type> {
  public:
    enum class TypeKind {
        // Basic Types, they are ranked in partial order
        VoidType = 0,
        BoolType,
        IntType,
        FloatType,
        // Derived Types
        TensorType,
        FunctionType,
        TupleType,
        PointerType,
        SumType,
        AnyType,
        // Analysis-Dependent Types
        LinearType,
        // ----
        NumTypes,
    };

    ~Type() = default;
    std::string getName() { return str(); }
    explicit operator std::string() { return str(); }

    virtual TypeKind kind() const { return TypeKind::VoidType; }
    virtual bool equals(const Type &rhs) { return true; }
    virtual std::string str() { return "void"; }
    virtual bool isVoidType() { return true; }
    virtual bool isIntType() { return false; }
    virtual bool isFloatType() { return false; }
    virtual bool isBoolType() { return false; }
    virtual bool isFunctionType() { return false; }
    virtual bool isTensorType() { return false; }
    virtual bool isTupleType() { return false; }
    virtual bool isPointerType() { return false; }
    virtual bool isSumType() { return false; }
    virtual bool isAnyType() { return false; }
    virtual bool isLinearType() { return false; }
    virtual TypePtr getTypePtr() { return nullptr; }

    bool operator==(const Type &other) { return equals(other); }
    bool operator!=(const Type &other) { return !equals(other); }
    bool operator<(const Type &other) const { return kind() < other.kind(); }
    bool compare(const Type &other);
};

class VoidType : public Type {
  public:
    TypeKind kind() const override { return Type::TypeKind::IntType; }

    bool equals(const Type &rhs) override { return kind() == rhs.kind(); }

    std::string str() override { return "void"; }

    bool isVoidType() override { return true; }

    TypePtr getTypePtr() override { return SingletonTypePtr<VoidType>::get(); }
};
using VoidTypePtr = SingletonTypePtr<VoidType>;

class IntType : public Type {
  public:
    TypeKind kind() const override { return Type::TypeKind::IntType; }

    bool equals(const Type &rhs) override { return kind() == rhs.kind(); }

    std::string str() override { return "i32"; }

    bool isIntType() override { return true; }

    TypePtr getTypePtr() override { return SingletonTypePtr<IntType>::get(); }
};
using IntTypePtr = SingletonTypePtr<IntType>;

class FloatType : public Type {
  public:
    TypeKind kind() const override { return Type::TypeKind::FloatType; }

    bool equals(const Type &rhs) override { return kind() == rhs.kind(); }

    std::string str() override { return "f32"; }

    bool isFloatType() override { return true; }

    TypePtr getTypePtr() override { return SingletonTypePtr<FloatType>::get(); }
};
using FloatTypePtr = SingletonTypePtr<FloatType>;

class BoolType : public Type {
  public:
    TypeKind kind() const override { return Type::TypeKind::BoolType; }

    bool equals(const Type &rhs) override { return kind() == rhs.kind(); }

    std::string str() override { return "bool"; }

    bool isBoolType() override { return true; }

    TypePtr getTypePtr() override { return SingletonTypePtr<BoolType>::get(); }
};
using BoolTypePtr = SingletonTypePtr<BoolType>;

class FunctionType;
using FunctionTypePtr = std::shared_ptr<FunctionType>;
class FunctionType : public Type {
  public:
    TypeKind kind() const override { return Type::TypeKind::FunctionType; }
    FunctionType(TypePtr inputType, TypePtr returnType)
        : argType(std::move(inputType)), returnType(std::move(returnType)) {}
    static FunctionTypePtr create(const TypePtr &inType,
                                  const TypePtr &retType) {
        return std::make_shared<FunctionType>(inType, retType);
    }
    bool equals(const Type &rhs) override {
        if (kind() != rhs.kind()) {
            return false;
        } else {
            Type rhsType = rhs;
            FunctionTypePtr rhsPtr =
                std::dynamic_pointer_cast<FunctionType>(rhsType.getTypePtr());
            return argType->equals(*rhsPtr->argType) &&
                   returnType->equals(*rhsPtr->returnType);
        }
    }
    std::string str() override {
        return argType->getName() + " -> " + returnType->getName();
    }
    bool isFunctionType() override { return true; }
    TypePtr getTypePtr() override { return shared_from_this(); }
    TypePtr getReturnType() { return returnType; }

  private:
    TypePtr argType;
    TypePtr returnType;
};

class TupleType;
using TupleTypePtr = std::shared_ptr<TupleType>;
class TupleType : public Type {
  public:
    explicit TupleType(const std::vector<TypePtr> &types) : types(types) {}

    TupleType(const std::vector<TypePtr> &types,
              const std::vector<std::string> &names)
        : types(types), names(names) {}

    static TupleTypePtr
    createNamedTuple(const std::vector<TypePtr> &types,
                     const std::vector<std::string> &names) {
        assert(types.size() == names.size());
        return std::make_shared<TupleType>(types, names);
    }

    static TupleTypePtr createUnnamedTuple(const std::vector<TypePtr> &types) {
        return std::make_shared<TupleType>(types);
    }

    TypeKind kind() const override { return TypeKind::TupleType; }
    bool equals(const Type &rhs) override {
        if (kind() != rhs.kind()) {
            return false;
        } else {
            Type rhsType = rhs;
            TupleTypePtr rhsPtr =
                std::dynamic_pointer_cast<TupleType>(rhsType.getTypePtr());
            if (types.size() != rhsPtr->types.size()) {
                return false;
            }
            for (size_t i = 0; i < types.size(); i++) {
                if (!types[i]->equals(*(rhsPtr->types[i])))
                    return false;
            }
            return true;
        }
    }

    std::string str() override {
        std::stringstream ssm;
        size_t size = types.size();
        ssm << "tuple(";
        for (size_t i = 0; i < size; i++) {
            if (i == size - 1) {
                ssm << types[i]->str() << ")";
            } else {
                ssm << types[i]->str() << ", ";
            }
        }
        return ssm.str();
    }

    bool isTupleType() override { return true; }

    TypePtr getTypePtr() override { return shared_from_this(); }

    std::vector<TypePtr> getTypes() { return types; }

  private:
    std::vector<TypePtr> types;
    std::vector<std::string> names;
};

class PointerType;
using PointerTypePtr = std::shared_ptr<PointerType>;
class PointerType : public Type {
  public:
    static PointerTypePtr createPointerType(const TypePtr &pointeeType) {
        return std::make_shared<PointerType>(pointeeType);
    }
    explicit PointerType(TypePtr pointeeType)
        : pointeeType(std::move(pointeeType)) {}

  public:
    TypeKind kind() const override { return Type::TypeKind::PointerType; }
    bool equals(const Type &rhs) override {
        if (kind() != rhs.kind()) {
            return false;
        } else {
            Type rhsType = rhs;
            PointerTypePtr rhsPtr =
                std::dynamic_pointer_cast<PointerType>(rhsType.getTypePtr());
            return (*pointeeType).equals(*rhsPtr->getPointeeType());
        }
    }
    std::string str() override {
        std::stringstream ssm;
        ssm << pointeeType->str() << " *";
        ssm << " *";
        return ssm.str();
    }
    bool isPointerType() override { return true; }
    TypePtr getTypePtr() override { return shared_from_this(); }
    TypePtr getPointeeType() { return pointeeType; }

  private:
    TypePtr pointeeType;
};

class TensorType;
using TensorTypePtr = std::shared_ptr<TensorType>;
class TensorType : public Type {
  public:
    static TensorTypePtr create(const TypePtr &elementType,
                                const std::vector<ValuePtr> &shape) {
        // std::vector<int> stride(1, (int)shape.size());
        return std::make_shared<TensorType>(elementType, shape);
    }

    TensorType(TypePtr elementType, const std::vector<ValuePtr> &shape)
        // const std::vector<ValuePtr> &stride)
        : elementType(std::move(elementType)), shape(shape) {}

  public:
    TypeKind kind() const override { return Type::TypeKind::TensorType; }
    bool equals(const Type &rhs) override {
        if (kind() != rhs.kind()) {
            return false;
        } else {
            Type rhsType = rhs;
            TensorTypePtr rhsPtr =
                std::dynamic_pointer_cast<TensorType>(rhsType.getTypePtr());
            if (shape.size() != rhsPtr->shape.size()) {
                return false;
            }
            for (size_t i = 0; i < shape.size(); i++) {
                if (shape[i] != rhsPtr->shape[i])
                    return false;
            }
            return true;
        }
    }
    std::string str() override;
    bool isTensorType() override { return true; }
    TypePtr getTypePtr() override { return shared_from_this(); }
    std::vector<ValuePtr> getShape() { return shape; }
    TypePtr getElementType() { return elementType; }

  private:
    TypePtr elementType;
    std::vector<ValuePtr> shape;
    // std::vector<ValuePtr> stride;
};

class LinearType : public Type {
  public:
    TypeKind kind() const override { return Type::TypeKind::LinearType; }

    bool equals(const Type &rhs) override { return kind() == rhs.kind(); }

    std::string str() override { return "linear"; }

    bool isLinearType() override { return true; }

    TypePtr getTypePtr() override {
        return SingletonTypePtr<LinearType>::get();
    }
};
using LinearTypePtr = SingletonTypePtr<LinearType>;

#endif // AINL_SRC_INCLUDE_TYPE_H
