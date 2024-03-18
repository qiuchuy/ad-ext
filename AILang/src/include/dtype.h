#pragma once

namespace ainl::core {

struct Dtype {
    enum class DataType {
        Any = 0,
        VoidType,
        BoolType,
        IntType,
        FloatType,
    };
    DataType type;
    constexpr explicit Dtype(DataType type) : type(type) {}
    bool operator<(const Dtype &other) const;
};

template <typename T> struct TypeToDtype { operator Dtype(); };

static constexpr Dtype Any = Dtype(Dtype::DataType::Any);
static constexpr Dtype Bool = Dtype(Dtype::DataType::BoolType);
static constexpr Dtype Int = Dtype(Dtype::DataType::IntType);
static constexpr Dtype Float = Dtype(Dtype::DataType::FloatType);

} // namespace ainl::core
