//
// Created by kom on 23-10-30.
//

#ifndef AINL_SRC_INCLUDE_TYPE_H
#define AINL_SRC_INCLUDE_TYPE_H

#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define SAFE_DOWNCAST(shared_ptr, derived_type)                                \
    std::dynamic_pointer_cast<derived_type>(shared_ptr)

template <typename T> class SingletonTypePtr {
  public:
    static std::shared_ptr<T> get() {
        static std::shared_ptr<T> ptr;
        return ptr;
    }

  public:
    SingletonTypePtr(SingletonTypePtr const &) = delete;
    void operator=(SingletonTypePtr const &) = delete;
};

// Type: Basic class in Type hirachy
class Type;
using TypePtr = std::shared_ptr<Type>;
class Type : public std::enable_shared_from_this<Type> {
  public:
    enum class TypeKind {
        // Basic Types
        VoidType = 0,
        IntType,
        FloatType,
        BoolType,
        TensorType,
        // Derived Types
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
    virtual bool isTensorType() { return false; }
    virtual bool isTupleType() { return false; }
    virtual bool isPointerType() { return false; }
    virtual bool isSumType() { return false; }
    virtual bool isAnyType() { return false; }
    virtual bool isLinearType() { return false; }
    virtual TypePtr getTypePtr() { return nullptr; }

    bool operator==(const Type &other) { return equals(other); }
    bool operator!=(const Type &other) { return !equals(other); }
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
    TypeKind kind() const override { return Type::TypeKind::VoidType; }

    bool equals(const Type &rhs) override { return kind() == rhs.kind(); }

    std::string str() override { return "i32"; }

    bool isIntType() override { return true; }

    TypePtr getTypePtr() override { return SingletonTypePtr<IntType>::get(); }
};
using IntTypePtr = SingletonTypePtr<IntType>;

class FloatType : public Type {
  public:
    TypeKind kind() const override { return Type::TypeKind::IntType; }

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
    static TensorTypePtr createDense(const TypePtr &elementType,
                                     const std::vector<int> &shape) {
        std::vector<int> stride(1, (int)shape.size());
        return std::make_shared<TensorType>(elementType, shape, stride);
    }
    TensorType(TypePtr elementType, const std::vector<int> &shape,
               const std::vector<int> &stride)
        : elementType(std::move(elementType)), shape(shape), stride(stride) {}

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
    std::string str() override {
        std::stringstream ssm;
        size_t size = shape.size();
        ssm << "tensor<";
        for (size_t i = 0; i < size; i++) {
            if (i == size - 1) {
                ssm << shape[i] << ">";
            } else {
                ssm << shape[i] << "x";
            }
        }
        return ssm.str();
    }
    bool isTensorType() override { return true; }
    TypePtr getTypePtr() override { return shared_from_this(); }

  private:
    TypePtr elementType;
    std::vector<int> shape;
    std::vector<int> stride;
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
