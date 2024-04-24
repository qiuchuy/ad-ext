#pragma once

#include <memory>
#include <sstream>
#include <tuple>
#include <utility>

#include "ir/linklist.h"
#include "ir/type.h"
#include "ir/use.h"
#include "utils/logger.h"

namespace ainl::ir {

#define SAFE_VALUE_DOWNCAST(value, derived_type)                               \
  dynamic_cast<derived_type *>(value)

class Attribute;
class Value;
class Use;
class Type;
class Block;
class Graph;
using AttributePtr = Attribute *;
using ValuePtr = Value *;
using UsePtr = Use *;
using TypePtr = std::shared_ptr<Type>;
using BlockPtr = Block *;
using GraphPtr = std::shared_ptr<Graph>;

class Attribute {
public:
  enum class AttributeKind : size_t {
    ConstantValue = 0,
    Linearity,
    TangentValue,
    // ---
    NumAttributes,
  };

  Attribute(ValuePtr constant, TypePtr linear, ValuePtr tangent) {
    std::get<(size_t)AttributeKind::ConstantValue>(attributes_) = constant;
    std::get<(size_t)AttributeKind::ConstantValue>(names_) = "constant";
    std::get<(size_t)AttributeKind::Linearity>(attributes_) = std::move(linear);
    std::get<(size_t)AttributeKind::Linearity>(names_) = "linear";
    std::get<(size_t)AttributeKind::TangentValue>(attributes_) = tangent;
    std::get<(size_t)AttributeKind::TangentValue>(names_) = "tangent";
  }

  ~Attribute() = default;

  static AttributePtr createConstantAttributeValue(ValuePtr constant) {
    return new Attribute(constant, nullptr, nullptr);
  }

  static AttributePtr createLinearAttributeValue(TypePtr linear) {
    return new Attribute(nullptr, std::move(linear), nullptr);
  }

  static AttributePtr createTangentAttributeValue(ValuePtr tangent) {
    return new Attribute(nullptr, nullptr, tangent);
  }

  void setConstantAttributeValue(ValuePtr constant) {
    std::get<(size_t)AttributeKind::ConstantValue>(attributes_) = constant;
    std::get<(size_t)AttributeKind::ConstantValue>(names_) = "constant";
  }

  void setLinearAttributeValue(TypePtr linear) {
    std::get<(size_t)AttributeKind::Linearity>(attributes_) = std::move(linear);
    std::get<(size_t)AttributeKind::Linearity>(names_) = "linear";
  }

  void setTangentAttributeValue(ValuePtr tangent) {
    std::get<(size_t)AttributeKind::TangentValue>(attributes_) = tangent;
    std::get<(size_t)AttributeKind::TangentValue>(names_) = "tangent";
  }

  explicit operator std::string() { return str(attributes_, names_); }

private:
  template <size_t Index = 0, typename... Types1, typename... Types2>
  std::string str(const std::tuple<Types1...> &tup1,
                  const std::tuple<Types2...> &tup2) {
    if constexpr (Index < sizeof...(Types1)) {
      if (std::get<Index>(tup1)) {
        std::string element =
            std::get<Index>(tup2) + "=" + std::string(*std::get<Index>(tup1));
        if constexpr (Index + 1 < sizeof...(Types1)) {
          element += ", ";
        }
        return element + str<Index + 1>(tup1, tup2);
      } else {
        return str<Index + 1>(tup1, tup2);
      }
    } else {
      return "";
    }
  }

  std::tuple<ValuePtr, TypePtr, ValuePtr> attributes_;
  std::tuple<std::string, std::string, std::string> names_;
};

class Value : public ILinkNode {
public:
  enum class ValueKind {
    Value = 0,
    Node,
    Graph,
    Block,
    Literal,
  };

  std::string prefix;
  std::string name;

  // static std::string GLOBAL_PREFIX;
  static std::string LOCAL_PREFIX;
  // static std::string GLOBAL_NAME_PREFIX;
  static std::string LOCAL_NAME_PREFIX;
  static std::string FPARAM_NAME_PREFIX;

  static int valueNum;

public:
  Value();
  explicit Value(const TypePtr &type);
  Value(const std::vector<TypePtr> &types);
  ~Value() override = default;
  virtual std::string getName() const;
  TypePtr getType() const { return type; }

  virtual bool isLiteral() const { return "false"; }

  bool operator==(const Value &other) const;

  bool operator!=(const Value &other) const;
  explicit operator std::string() const override { return "!value!"; }
  friend class Block;
  friend class ALModule;
  friend class Node;

public:
  void insertUseAtEnd(UsePtr use);

protected:
  // Type of this value
  TypePtr type;

  // attribute will be created by nodes/analysis passes
  AttributePtr attribute;

  // Use-def chain
  UsePtr beginUse;
  UsePtr endUse;

  // This value could be a tuple which stores other values
  std::vector<ValuePtr> values;

  // the Graph & Block which this value belongs to
  GraphPtr graph;
  BlockPtr block;
};

TypePtr createTypePtrForValues(const std::vector<ValuePtr> &values);
} // namespace ainl::ir