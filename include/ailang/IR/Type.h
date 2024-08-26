#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ailang/Core/Dtype.h"
#include "ailang/IR/Value.h"

namespace ainl::ir {

class Value;
class Type;
using ValuePtr = Value *;
using TypePtr = std::shared_ptr<Type>;

#define SAFE_TYPE_DOWNCAST(shared_ptr, derived_type)                           \
  std::dynamic_pointer_cast<derived_type>(shared_ptr)

template <typename T> std::shared_ptr<T> asType(std::shared_ptr<Type> type) {
  return std::dynamic_pointer_cast<T>(type);
}

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
    DoubleType,
    // Derived Types
    FunctionType,
    TupleType,
    PointerType,
    SumType,
    AnyType,
    // Dependent Types
    TensorType,
    DependentTupleType,
    LiteralType,
    // Analysis-Ralted Types
    LinearType,
    // ----
    NumTypes,
  };

  ~Type() = default;
  std::string getName() { return str(); }
  explicit operator std::string() { return str(); }

  virtual TypeKind kind() const { return TypeKind::VoidType; }
  bool equals(const TypePtr &rhs) { return getName() == rhs->getName(); }
  virtual std::string str() { return "void"; }
  virtual bool isVoidType() { return true; }
  virtual bool isIntType() { return false; }
  virtual bool isFloatType() { return false; }
  virtual bool isDoubleType() { return false; }
  virtual bool isBoolType() { return false; }
  virtual bool isFunctionType() { return false; }
  virtual bool isTensorType() { return false; }
  virtual bool isTupleType() { return false; }
  virtual bool isPointerType() { return false; }
  virtual bool isDependentTupleType() { return false; }
  virtual bool isLiteralType() { return false; }
  virtual bool isLinearType() { return false; }
  virtual TypePtr getTypePtr() { return shared_from_this(); }

  bool operator==(const TypePtr &other) { return equals(other); }
  bool operator!=(const TypePtr &other) { return !equals(other); }
  bool operator<(const TypePtr &other) const { return kind() < other->kind(); }
  bool compare(Type &other);
};

class VoidType : public Type {
public:
  TypeKind kind() const override { return Type::TypeKind::IntType; }

  std::string str() override { return "void"; }

  bool isVoidType() override { return true; }

  TypePtr getTypePtr() override { return SingletonTypePtr<VoidType>::get(); }
};
using VoidTypePtr = SingletonTypePtr<VoidType>;

class IntType : public Type {
public:
  TypeKind kind() const override { return Type::TypeKind::IntType; }

  std::string str() override { return "i32"; }

  bool isIntType() override { return true; }

  TypePtr getTypePtr() override { return SingletonTypePtr<IntType>::get(); }
};
using IntTypePtr = SingletonTypePtr<IntType>;

class FloatType : public Type {
public:
  TypeKind kind() const override { return Type::TypeKind::FloatType; }

  std::string str() override { return "f32"; }

  bool isFloatType() override { return true; }

  TypePtr getTypePtr() override { return SingletonTypePtr<FloatType>::get(); }
};
using FloatTypePtr = SingletonTypePtr<FloatType>;

class DoubleType : public Type {
public:
  TypeKind kind() const override { return Type::TypeKind::DoubleType; }

  std::string str() override { return "f64"; }

  bool isDoubleType() override { return true; }

  TypePtr getTypePtr() override { return SingletonTypePtr<DoubleType>::get(); }
};
using DoubleTypePtr = SingletonTypePtr<DoubleType>;

class BoolType : public Type {
public:
  TypeKind kind() const override { return Type::TypeKind::BoolType; }

  std::string str() override { return "i1"; }

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
  static FunctionTypePtr create(const TypePtr &inType, const TypePtr &retType) {
    return std::make_shared<FunctionType>(inType, retType);
  }
  std::string str() override {
    return argType->getName() + " -> " + returnType->getName();
  }
  bool isFunctionType() override { return true; }
  TypePtr getTypePtr() override { return shared_from_this(); }
  TypePtr getReturnType() { return returnType; }
  TypePtr getArgType() { return argType; }

private:
  TypePtr argType;
  TypePtr returnType;
};

class TupleType;
using TupleTypePtr = std::shared_ptr<TupleType>;
class TupleType : public Type {
public:
  explicit TupleType(const std::vector<TypePtr> &types) : types(types) {}

  TupleType() = default;
  TupleType(const std::vector<TypePtr> &types,
            const std::vector<std::string> &names)
      : types(types), names(names) {}

  static TupleTypePtr createNamedTuple(const std::vector<TypePtr> &types,
                                       const std::vector<std::string> &names) {
    assert(types.size() == names.size());
    return std::make_shared<TupleType>(types, names);
  }

  static TupleTypePtr createUnnamedTuple(const std::vector<TypePtr> &types) {
    return std::make_shared<TupleType>(types);
  }

  TypeKind kind() const override { return TypeKind::TupleType; }

  std::string str() override {
    std::stringstream ssm;
    size_t size = types.size();
    ssm << "(";
    for (size_t i = 0; i < size; i++) {
      if (i == size - 1) {
        ssm << types[i]->str() << " ";
      } else {
        ssm << types[i]->str() << ", ";
      }
    }
    ssm << ")";
    return ssm.str();
  }

  bool isTupleType() override { return true; }

  TypePtr getTypePtr() override { return shared_from_this(); }

  std::vector<TypePtr> getTypes() { return types; }

protected:
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
  std::string str() override;
  bool isTensorType() override { return true; }
  TypePtr getTypePtr() override { return shared_from_this(); }
  std::vector<int> getConcreteShape();
  std::vector<ValuePtr> getShape() { return shape; }
  TypePtr getElementType() { return elementType; }

private:
  TypePtr elementType;
  std::vector<ValuePtr> shape;
  // std::vector<ValuePtr> stride;
};

class DependentTupleType;
using DependentTupleTypePtr = std::shared_ptr<DependentTupleType>;
class DependentTupleType : public TupleType {
public:
  static DependentTupleTypePtr create(const std::vector<ValuePtr> &values) {
    // std::vector<int> stride(1, (int)shape.size());
    return std::make_shared<DependentTupleType>(values);
  }

  DependentTupleType(const std::vector<ValuePtr> &values);

public:
  TypeKind kind() const override { return Type::TypeKind::DependentTupleType; }
  bool isDependentTupleType() override { return true; }

private:
  std::vector<ValuePtr> values;
};

class LiteralType;
using LiteralTypePtr = std::shared_ptr<LiteralType>;
class LiteralType : public Type {
public:
  static LiteralTypePtr create(const ValuePtr &value) {
    // std::vector<int> stride(1, (int)shape.size());
    return std::make_shared<LiteralType>(value);
  }

  LiteralType(const ValuePtr &value);

public:
  TypeKind kind() const override { return Type::TypeKind::LiteralType; }
  bool isLiteralType() override { return true; }
  std::string str() override;
  TypePtr getTypePtr() override { return shared_from_this(); }

  ValuePtr getValue();

private:
  ValuePtr value;
};

TypePtr DtypeToTypePtr(core::Dtype dtype);

} // namespace ainl::ir
