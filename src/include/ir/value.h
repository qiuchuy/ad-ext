#ifndef AINL_SRC_INCLUDE_VALUE_H
#define AINL_SRC_INCLUDE_VALUE_H

#include <memory>
#include <sstream>
#include <tuple>

#include "linklist.h"
#include "type.h"
#include "use.h"

class Attribute;
class Value;
class Use;
using AttributePtr = Attribute *;
using ValuePtr = Value *;
using UsePtr = Use *;

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
        std::get<(size_t)AttributeKind::Linearity>(attributes_) = linear;
        std::get<(size_t)AttributeKind::Linearity>(names_) = "linear";
        std::get<(size_t)AttributeKind::TangentValue>(attributes_) = tangent;
        std::get<(size_t)AttributeKind::TangentValue>(names_) = "tangent";
    }

    ~Attribute() {}

    static AttributePtr createConstantAttributeValue(ValuePtr constant) {
        return new Attribute(constant, nullptr, nullptr);
    }

    static AttributePtr createLinearAttributeValue(TypePtr linear) {
        return new Attribute(nullptr, linear, nullptr);
    }

    static AttributePtr createTangentAttributeValue(ValuePtr tangent) {
        return new Attribute(nullptr, nullptr, tangent);
    }

    void setConstantAttributeValue(ValuePtr constant) {
        std::get<(size_t)AttributeKind::ConstantValue>(attributes_) = constant;
        std::get<(size_t)AttributeKind::ConstantValue>(names_) = "constant";
    }

    void setLinearAttributeValue(TypePtr linear) {
        std::get<(size_t)AttributeKind::Linearity>(attributes_) = linear;
        std::get<(size_t)AttributeKind::Linearity>(names_) = "linear";
    }

    void setTangentAttributeValue(ValuePtr tangent) {
        std::get<(size_t)AttributeKind::TangentValue>(attributes_) = tangent;
        std::get<(size_t)AttributeKind::TangentValue>(names_) = "tangent";
    }

    operator std::string() { return str(attributes_, names_); }

  private:
    template <size_t Index = 0, typename... Types1, typename... Types2>
    std::string str(const std::tuple<Types1...> &tup1,
                    const std::tuple<Types2...> &tup2) {
        if constexpr (Index < sizeof...(Types1)) {
            if (std::get<Index>(tup1)) {
                std::string element = std::get<Index>(tup2) + "=" +
                                      std::string(*std::get<Index>(tup1));
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
    };

    std::string prefix;
    std::string name;

    static std::string GLOBAL_PREFIX;
    static std::string LOCAL_PREFIX;
    static std::string GLOBAL_NAME_PREFIX;
    static std::string LOCAL_NAME_PREFIX;
    static std::string FPARAM_NAME_PREFIX;

  public:
    Value();
    Value(TypePtr type);
    Value(const std::vector<TypePtr> &types);
    ~Value() override = default;
    operator std::string() const { return ""; }
    virtual std::string getName() const;
    TypePtr getType() { return type; }

  public:
    void insertUseAtEnd(UsePtr use);

  protected:
    TypePtr type;
    AttributePtr attribute;
    UsePtr beginUse;
    UsePtr endUse;
    ValuePtr beginValue;
    ValuePtr endValue;
};

class Constant : public Value {};

TypePtr createTypePtrForValues(const std::vector<ValuePtr> &values);

#endif // AINL_SRC_INCLUDE_VALUE_H