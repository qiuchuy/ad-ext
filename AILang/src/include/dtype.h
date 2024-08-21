#pragma once

#include <cstddef>
#include <ostream>
#include <string>

namespace ainl::core {

struct Dtype {
    enum class DataType {
        Any = 0,
        VoidType,
        BoolType,
        Int8Type,
        Int16Type,
        Int32Type,
        Int64Type,
        Float32Type,
        Float64Type,
    };
    DataType type;
    Dtype() = default;
    constexpr explicit Dtype(DataType type) : type(type) {}
    bool operator<(const Dtype &other) const;
    std::string toString() const;
    size_t hash() const;
    friend std::ostream &operator<<(std::ostream &os, const Dtype &dtype);
    bool operator==(const Dtype &other) const;
};

template <typename T> struct TypeToDtype {
    operator Dtype();
};

static constexpr Dtype Any = Dtype(Dtype::DataType::Any);
static constexpr Dtype Bool = Dtype(Dtype::DataType::BoolType);
static constexpr Dtype Int8 = Dtype(Dtype::DataType::Int8Type);
static constexpr Dtype Int16 = Dtype(Dtype::DataType::Int16Type);
static constexpr Dtype Int32 = Dtype(Dtype::DataType::Int32Type);
static constexpr Dtype Int64 = Dtype(Dtype::DataType::Int64Type);
static constexpr Dtype Float32 = Dtype(Dtype::DataType::Float32Type);
static constexpr Dtype Float64 = Dtype(Dtype::DataType::Float64Type);

size_t dtypeSize(Dtype dtype);

Dtype getDtypeFromFormat(const std::string &formatStr);

} // namespace ainl::core
