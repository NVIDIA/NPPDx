/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMONDX_COMPLEX_TYPES_CUTLASS_HPP
#define COMMONDX_COMPLEX_TYPES_CUTLASS_HPP

// Include cutlass numeric and complex types
#include <cutlass/numeric_types.h>
#include <cutlass/complex.h>

// Include base complex types
#include "commondx/complex_types.hpp"
#include "commondx/detail/stl/type_traits.hpp"

#include "commondx/traits/numeric_traits.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    // Type aliases for cutlass types
    using half_t       = typename cutlass::half_t;
    using tfloat32_t   = typename cutlass::tfloat32_t;
    using bfloat16_t   = typename cutlass::bfloat16_t;
    using float_e5m2_t = typename cutlass::float_e5m2_t;
    using float_e4m3_t = typename cutlass::float_e4m3_t;

    // Specializations of is_floating_point (numeric_traits.hpp) for cutlass types
    template<> struct is_floating_point<half_t>       : COMMONDX_STL_NAMESPACE::true_type {};
    template<> struct is_floating_point<tfloat32_t>   : COMMONDX_STL_NAMESPACE::true_type {};
    template<> struct is_floating_point<bfloat16_t>   : COMMONDX_STL_NAMESPACE::true_type {};
    template<> struct is_floating_point<float_e5m2_t> : COMMONDX_STL_NAMESPACE::true_type {};
    template<> struct is_floating_point<float_e4m3_t> : COMMONDX_STL_NAMESPACE::true_type {};

    namespace detail {
        // Extend use_default_operator to include cutlass types
        template<typename T>
        struct use_default_operator_cutlass {
            static constexpr bool value = use_default_operator<T>::value
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, half_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, bfloat16_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, float_e5m2_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, float_e4m3_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, tfloat32_t>
                                       ;
        };

        // Operators for cutlass types
        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<
            use_default_operator_cutlass<T>::value && !use_default_operator<T>::value, int> = 0>
        __device__ __forceinline__ __host__ complex<T> operator*(const complex<T>& a, const complex<T>& b) {
            return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
        }

        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<
            use_default_operator_cutlass<T>::value && !use_default_operator<T>::value, int> = 0>
        __device__ __forceinline__ __host__ complex<T> operator+(const complex<T>& a, const complex<T>& b) {
            return {a.x + b.x, a.y + b.y};
        }

        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<
            use_default_operator_cutlass<T>::value && !use_default_operator<T>::value, int> = 0>
        __device__ __forceinline__ __host__ complex<T> operator-(const complex<T>& a, const complex<T>& b) {
            return {a.x - b.x, a.y - b.y};
        }

        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<
            use_default_operator_cutlass<T>::value && !use_default_operator<T>::value, int> = 0>
        __device__ __forceinline__ __host__ bool operator==(const complex<T>& a, const complex<T>& b) {
            return (a.x == b.x) && (a.y == b.y);
        }

        // tfloat32_t conversions only work with float/double
        template<typename K>
        struct is_valid_conversion_to_from<tfloat32_t, K> : COMMONDX_STL_NAMESPACE::bool_constant<
            COMMONDX_STL_NAMESPACE::is_same_v<K, float> ||
            COMMONDX_STL_NAMESPACE::is_same_v<K, double>> {};

        template<typename T>
        struct is_valid_conversion_to_from<T, tfloat32_t> : COMMONDX_STL_NAMESPACE::bool_constant<
            COMMONDX_STL_NAMESPACE::is_same_v<T, float> ||
            COMMONDX_STL_NAMESPACE::is_same_v<T, double>> {};

        // bfloat16_t conversions only work with float/double
        template<typename K>
        struct is_valid_conversion_to_from<bfloat16_t, K> : COMMONDX_STL_NAMESPACE::bool_constant<
            COMMONDX_STL_NAMESPACE::is_same_v<K, float> ||
            COMMONDX_STL_NAMESPACE::is_same_v<K, double>> {};

        template<typename T>
        struct is_valid_conversion_to_from<T, bfloat16_t> : COMMONDX_STL_NAMESPACE::bool_constant<
            COMMONDX_STL_NAMESPACE::is_same_v<T, float> ||
            COMMONDX_STL_NAMESPACE::is_same_v<T, double>> {};

        // __half
        // Invalid to convert from tfloat32_t or bfloat16_t to __half
        template<>
        struct is_valid_conversion_to_from<__half, tfloat32_t> : COMMONDX_STL_NAMESPACE::false_type {};
        template<>
        struct is_valid_conversion_to_from<__half, bfloat16_t> : COMMONDX_STL_NAMESPACE::false_type {};

        // __nv_bfloat16
        // Invalid to convert from cutlass types to __nv_bfloat16
        template<>
        struct is_valid_conversion_to_from<__nv_bfloat16, half_t>       : COMMONDX_STL_NAMESPACE::false_type {};
        template<>
        struct is_valid_conversion_to_from<__nv_bfloat16, bfloat16_t>   : COMMONDX_STL_NAMESPACE::false_type {};
        template<>
        struct is_valid_conversion_to_from<__nv_bfloat16, tfloat32_t>   : COMMONDX_STL_NAMESPACE::false_type {};
        template<>
        struct is_valid_conversion_to_from<__nv_bfloat16, float_e5m2_t> : COMMONDX_STL_NAMESPACE::false_type {};
        template<>
        struct is_valid_conversion_to_from<__nv_bfloat16, float_e4m3_t> : COMMONDX_STL_NAMESPACE::false_type {};

        // Add static assertions to the generic complex template for cutlass types
        // These will be checked at compile time when conversions involving cutlass types are attempted

        // Specialization that adds validation for template constructor with cutlass types
        template<>
        template<typename K>
        __device__ __forceinline__ __host__
        constexpr complex<tfloat32_t>::complex(const complex<K>& other)
            : complex(tfloat32_t(other.real()), tfloat32_t(other.imag())) {
            static_assert(is_valid_conversion_to_from<tfloat32_t, K>::value,
                "For T = tfloat32_t, only supports K = float/double");
        }

        template<>
        template<typename K>
        __device__ __forceinline__ __host__
        constexpr complex<bfloat16_t>::complex(const complex<K>& other)
            : complex(bfloat16_t(other.real()), bfloat16_t(other.imag())) {
            static_assert(is_valid_conversion_to_from<bfloat16_t, K>::value,
                "For T = bfloat16_t, only supports K = float/double");
        }

        // Specialization for assignment operators with validation
        template<>
        template<typename K>
        __device__ __forceinline__ __host__
        complex<tfloat32_t>& complex<tfloat32_t>::operator=(const complex<K>& other) {
            static_assert(is_valid_conversion_to_from<tfloat32_t, K>::value,
                "For T = tfloat32_t, only supports K = float/double");
            this->x = static_cast<tfloat32_t>(other.real());
            this->y = static_cast<tfloat32_t>(other.imag());
            return *this;
        }

        template<>
        template<typename K>
        __device__ __forceinline__ __host__
        complex<bfloat16_t>& complex<bfloat16_t>::operator=(const complex<K>& other) {
            static_assert(is_valid_conversion_to_from<bfloat16_t, K>::value,
                "For T = bfloat16_t, only supports K = float/double");
            this->x = static_cast<bfloat16_t>(other.real());
            this->y = static_cast<bfloat16_t>(other.imag());
            return *this;
        }

        // Conversion constructors for complex<__half> from cutlass types
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
        template<>
        __device__ __forceinline__ __host__
        complex<__half>::complex(const complex<half_t>& other):
            complex<__half>(other.real().to_half(), other.imag().to_half()) {}

        template<>
        __device__ __forceinline__ __host__
        complex<__half>::complex(const complex<float_e5m2_t>& other):
            complex<__half>(float_e5m2_t::to_half(other.real()), float_e5m2_t::to_half(other.imag())) {}

        template<>
        __device__ __forceinline__ __host__
        complex<__half>::complex(const complex<float_e4m3_t>& other):
            complex<__half>(float_e4m3_t::to_half(other.real()), float_e4m3_t::to_half(other.imag())) {}
#endif

    } // namespace detail
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_COMPLEX_TYPES_CUTLASS_HPP
