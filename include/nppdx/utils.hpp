/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NPPDX_UTILS_HPP
#define NPPDX_UTILS_HPP

#include "nppdx/detail/config.hpp"
#include "nppdx/detail/decl.hpp"

#include NPPDX_STD_INCLUDE_TYPE_TRAITS
#include NPPDX_STD_INCLUDE_CSTDIO

#if !defined(__CUDACC_RTC__) && !(defined(NPPDX_CLANG_CUDA_COMPAT) && defined(__CUDA_ARCH__))
#    define NPPDX_PRIVATE_HAVE_HOST_EXCEPTIONS 1
#endif

#if NPPDX_PRIVATE_HAVE_HOST_EXCEPTIONS
#    include <stdexcept>
#endif
namespace nppdx {

    template<typename T, typename U>
    NPPDX_DECL_NCFHD auto nppdx_min(T a, U b) {
#if defined(NPPDX_CLANG_CUDA_COMPAT)
        return (a) < (b) ? (a) : (b);
#elif defined(__CUDA_ARCH__)
        return min(a, b);
#else
        return (a < b) ? a : b;
#endif
    }

    template<typename T, typename U>
    NPPDX_DECL_NCFHD auto nppdx_max(T a, U b) {
#if defined(NPPDX_CLANG_CUDA_COMPAT)
        return (a) > (b) ? (a) : (b);
#elif defined(__CUDA_ARCH__)
        return max(a, b);
#else
        return (a > b) ? a : b;
#endif
    }

    template<typename T>
    NPPDX_DECL_NCFHD T ceil_div(T a, T b) {
        // Note: Intended to be used with non-negative a and positive b.
        // The expression below is more verbose than the common (a + b - 1) / b, but enables better compiler
        // optimization in the common case that a % b is also needed, as it can be derived cheaply from a / b (but not
        // from (a + b - 1) / b).
        return a / b + T(a % b != 0);
    }

    // "Directional" quotient and remainder. Denoting these q and r,
    // respectively, these are given for a dividend x and divisor d as the
    // integral solutions to x = q * d + s * r with 0 <= r < d. d is assumed to
    // be positive. s represents the direction and is +1 for the positive
    // direction and -1 for the negative direction. The positive directional
    // quotient and remainder are the mathematical quotient and remainder. This
    // coincides with the usual C++ quotient and remainder for a positive x, but
    // already needs a custom implementation for negative x. The negative
    // directional quotient and remainder are somewhat idiosyncratic but
    // simplify a number of expressions.

    template<int Dir = +1, typename T>
    NPPDX_DECL_NCFHD T dir_pos_div(T a, T b) {
        static_assert(Dir == +1 || Dir == -1);
        if constexpr (Dir == +1) {
            return a / b;
        } else {
            return ceil_div(a, b);
        }
    }

    template<typename T>
    NPPDX_DECL_NCFHD T anti_rem(T rem, T d) {
        return (rem == 0) ? 0 : d - rem;
    }

    template<int Dir, typename T>
    NPPDX_DECL_NCFHD T dir_pos_rem(T a, T b) {
        static_assert(Dir == +1 || Dir == -1);
        if constexpr (Dir == +1) {
            return a % b;
        } else {
            return anti_rem(a % b, b);
        }
    }

    // Helper to suppress warnings about pointlesss unsigned comparisons.
    template<typename T>
    NPPDX_DECL_NCFHD bool is_negative([[maybe_unused]] T val) {
        if constexpr (NPPDX_STD::is_unsigned_v<T>) {
            return false;
        } else {
            return val < 0;
        }
    }

    template<int Dir, typename T>
    NPPDX_DECL_NCFHD T dir_quot(T dividend, T divisor) {
        static_assert(Dir == +1 || Dir == -1);
        if (is_negative(dividend)) {
            return -dir_pos_div<-Dir>(-dividend, divisor);
        } else {
            return dir_pos_div<Dir>(dividend, divisor);
        }
    }

    template<int Dir, typename T>
    NPPDX_DECL_NCFHD T dir_rem(T dividend, T divisor) {
        static_assert(Dir == +1 || Dir == -1);
        if (is_negative(dividend)) {
            return dir_pos_rem<-Dir>(-dividend, divisor);
        } else {
            return dir_pos_rem<Dir>(dividend, divisor);
        }
    }

    template<typename T>
    __host__ __device__ inline constexpr T clamp_value(T x, T lo, T hi) {
        return x < lo ? lo : (x > hi ? hi : x);
    }

    // The utilities below allow uniform handling of compile-time and runtime
    // values. CVal<Val>{} is the compile-time equivalent of the runtime Val.
    // Any type that is not a specialization of CVal is considered to "have a
    // runtime type". In both cases, Val is referred to as the "carried value"
    // and either decltype(Val) in the runtime case or CVal<Val> in the
    // compile-time case as the "carrier type".
    // Functions that generically handle both runtime and compile-time
    // parameters must be templated on those parameters' types. Often, they will
    // still expect them to have a given carried type. In C++20, this expectation
    // could be expressed through concepts. Pre-C++20, one can use the
    // has_carried_type variable template in a static_assert.
    // For convenience, the variable template C<Val> is defined to be a
    // default-constructed CVal<Val>. This can be used at call sites when
    // explicitly passing a compile-time value.

    template<auto Val>
    struct CVal {
        NPPDX_DECL_SC auto value = Val;
    };

    template<typename T>
    NPPDX_DECL_CI bool is_cval = false;

    template<auto Val>
    NPPDX_DECL_CI bool is_cval<CVal<Val>> = true;

    namespace detail {
        template<typename T>
        auto get_carried_type([[maybe_unused]] T val) {
            if constexpr (is_cval<T>) {
                return T::value;
            } else {
                return val;
            }
        };
    } // namespace detail

    template<typename T>
    using carried_type = decltype(detail::get_carried_type(NPPDX_STD::declval<T>()));

    template<typename DesiredCarriedT, typename TestedT>
    NPPDX_DECL_CI bool has_carried_type = NPPDX_STD::is_same_v<DesiredCarriedT, carried_type<TestedT>>;

    template<auto Val>
    NPPDX_DECL_CI auto C = CVal<Val> {};

    template<typename T>
    NPPDX_DECL_NCHD auto get_carried_val(const T& carrier) {
        if constexpr (is_cval<T>) {
            return carrier.value;
        } else {
            return carrier;
        }
    }

    [[noreturn]] NPPDX_DECL_IHD void fatal_error(char const* message = "") {
        printf("Fatal error: %s\n", message);
#ifdef __CUDA_ARCH__
        __trap();
#    ifdef __clang__
        __builtin_unreachable(); // Suppress warnings about noreturn function returning.
#    endif
#elif NPPDX_PRIVATE_HAVE_HOST_EXCEPTIONS
        throw std::logic_error(message);
#else
        NPPDX_STD::terminate();
#endif
    }

    struct simple_nullopt_t {
    };

    constexpr simple_nullopt_t simple_nullopt = simple_nullopt_t {};

    template<typename T>
    class simple_optional
    {
        static_assert(NPPDX_STD::is_default_constructible_v<T>,
                      "simple_optional requires a default-constructible type");

        T    value_;
        bool has_value_;

    public:
        NPPDX_DECL_CHD simple_optional(T value): value_(value), has_value_(true) {}
        NPPDX_DECL_CHD simple_optional(): value_(), has_value_(false) {}
        NPPDX_DECL_CHD simple_optional(simple_nullopt_t): value_(), has_value_(false) {}

        NPPDX_DECL_CHD simple_optional& operator=(T value) {
            value_     = value;
            has_value_ = true;
            return *this;
        }
        NPPDX_DECL_CHD simple_optional& operator=(simple_nullopt_t) {
            has_value_ = false;
            return *this;
        }

        NPPDX_DECL_NCHD bool has_value() const { return has_value_; }
        NPPDX_DECL_NCHD T&   value() {
            if (!has_value_) {
                fatal_error("simple_optional: no value");
            }
            return value_;
        }
        NPPDX_DECL_NCHD const T& value() const {
            if (!has_value_) {
                fatal_error("simple_optional: no value");
            }
            return value_;
        }

        NPPDX_DECL_NCHD T&       operator*() { return value_; }
        NPPDX_DECL_NCHD const T& operator*() const { return value_; }

        NPPDX_DECL_NCHD T*       operator->() { return &value_; }
        NPPDX_DECL_NCHD const T* operator->() const { return &value_; }
    };

    struct Empty {
    };

    // Trivial copy function largely intended to work around compiler quirks, non-conformities, and bugs.
    // Examples:
    // * GCC versions 9.4-13.4 sometimes wrongly complain that variables from which non-type template arguments are
    //   derived should not only be constexpr but also static. Wrapping the arguments in a copy() suppresses this
    //   behavior.
    // * if static constexpr values defined outside a __device__ function are accessed by reference within such a
    //   function, the access fails as no device symbol is typically generated. Wrapping the generated name in a copy()
    //   suppresses this behavior.
    template<typename T>
    NPPDX_DECL_NCFHD T copy(T val) {
        return val;
    }

    // Tagged-type infrastructure.
    // When passing multiple parameters of the same type to a function, it is sometimes easy to mix up their order.
    // This can be mitigated by letting the function accept distinct thin wrappers around the type. These have to have
    // an explicit constructor from the underlying type and, for convenience, an implicit conversion to the underlying
    // type.
    template<typename T, typename TagT>
    class tagged_type
    {
        T value_;

    public:
        NPPDX_DECL_CEFHD         tagged_type(T value): value_(value) {}
        NPPDX_DECL_NCHD          operator T() const { return value_; }
        NPPDX_DECL_NCHD T&       operator*() { return value_; }
        NPPDX_DECL_NCHD const T& operator*() const { return value_; }
        NPPDX_DECL_NCHD T*       operator->() { return &value_; }
        NPPDX_DECL_NCHD const T* operator->() const { return &value_; }
    };

#define NPPDX_DECLARE_TAGGED_TYPE(Name, WrappedT) \
    namespace detail {                            \
        struct Name##Tag {                        \
        };                                        \
    }                                             \
    using Name = tagged_type<WrappedT, detail::Name##Tag>;

    // Helpers for sets of static assertions. Such a set can be put in the body of a struct template A with a single
    // type template parameter T where A inherits from checker_base<T>. When a specific type U needs to be checked, it
    // should be passed to check_t<A, U> which evaluates to U if all checks pass and static_asserts otherwise.

    template<typename T>
    struct checker_base {
        using type = T;
    };

    template<template<typename> typename Checker, typename T>
    using check_t = typename Checker<T>::type;

} // namespace nppdx

#endif // NPPDX_DETAIL_UTILS_HPP
