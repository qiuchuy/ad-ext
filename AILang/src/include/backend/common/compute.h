#pragma once

#include <cmath>
namespace ainl::core::detail {

struct Abs {
    template <typename T> T operator()(T x) { return std::fabs(x); };

    uint8_t operator()(uint8_t x) { return x; };
};

struct ArcCos {
    template <typename T>

    T operator()(T x) {
        return std::acos(x);
    };
};
struct ArcCosh {
    template <typename T>

    T operator()(T x) {
        return std::acosh(x);
    };
};

struct ArcSin {
    template <typename T>

    T operator()(T x) {
        return std::asin(x);
    };
};

struct ArcSinh {
    template <typename T>

    T operator()(T x) {
        return std::asinh(x);
    };
};

struct ArcTan {
    template <typename T>

    T operator()(T x) {
        return std::atan(x);
    };
};
struct ArcTanh {
    template <typename T>

    T operator()(T x) {
        return std::atanh(x);
    };
};
struct Sin {
    template <typename T>

    T operator()(T x) {
        return std::sin(x);
    };
};

struct Cos {
    template <typename T>

    T operator()(T x) {
        return std::cos(x);
    };
};

struct Tan {
    template <typename T>

    T operator()(T x) {
        return std::tan(x);
    };
};
struct Sinh {
    template <typename T>

    T operator()(T x) {
        return std::sinh(x);
    };
};

struct Cosh {
    template <typename T>

    T operator()(T x) {
        return std::cosh(x);
    };
};

struct Tanh {
    template <typename T>

    T operator()(T x) {
        return std::tanh(x);
    };
};

struct Exp {
    // TODO fast exp
    template <typename T> T operator()(T x) { return std ::exp(x); }
};

struct Log {
    template <typename T> T operator()(T x) { return std::log(x); }
};

struct Log2 {
    template <typename T> T operator()(T x) { return std::log2(x); }
};

struct Log10 {
    template <typename T> T operator()(T x) { return std::log10(x); }
};

// TODO fast Sigmoid
struct Sigmoid {
    template <typename T> T operator()(T x) {
        auto one = static_cast<decltype(x)>(1.0);
        return one / (one + exp(-x));
    }
};

struct Add {
    template <typename T> T operator()(T x, T y) { return x + y; }
};

struct Multiply {
    template <typename T> T operator()(T x, T y) { return x * y; }
};

}; // namespace ainl::core::detail