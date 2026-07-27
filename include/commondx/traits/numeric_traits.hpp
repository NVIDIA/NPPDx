/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMONDX_TRAITS_NUMERIC_TRAITS_HPP
#define COMMONDX_TRAITS_NUMERIC_TRAITS_HPP

#include "commondx/detail/stl/cstdint.hpp"

namespace commondx {
        template<typename T>
        struct is_floating_point {
            static constexpr bool value =
                COMMONDX_STL_NAMESPACE::is_floating_point_v<T> or
#if COMMONDX_DETAIL_CUDA_FP8_ENABLED
                COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_fp8_e4m3> or
                COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_fp8_e5m2> or
#endif
                COMMONDX_STL_NAMESPACE::is_same_v<T, __half> or
                COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_bfloat16> or
                COMMONDX_STL_NAMESPACE::is_same_v<T, float> or
                COMMONDX_STL_NAMESPACE::is_same_v<T, double>;
        };

        template<template<class> class Complex, class T>
        struct is_floating_point<Complex<T>> : is_floating_point<T> {};

        template<typename T>
        static constexpr bool is_floating_point_v = is_floating_point<T>::value;

        template<typename T>
        struct is_signed_integral {
            static constexpr bool value =
                (COMMONDX_STL_NAMESPACE::is_integral_v<T> and COMMONDX_STL_NAMESPACE::is_signed_v<T>) or
                COMMONDX_STL_NAMESPACE::is_same_v<T, int8_t>   or
                COMMONDX_STL_NAMESPACE::is_same_v<T, int16_t>  or
                COMMONDX_STL_NAMESPACE::is_same_v<T, int32_t>  or
                COMMONDX_STL_NAMESPACE::is_same_v<T, int64_t>;
        };

        template<template<class> class Complex, class T>
        struct is_signed_integral<Complex<T>> : is_signed_integral<T> {};

        template<typename T>
        static constexpr bool is_signed_integral_v = is_signed_integral<T>::value;

        template<typename T>
        struct is_unsigned_integral {
            static constexpr bool value =
                (COMMONDX_STL_NAMESPACE::is_integral_v<T> and COMMONDX_STL_NAMESPACE::is_unsigned_v<T>) or
                COMMONDX_STL_NAMESPACE::is_same_v<T, uint8_t>   or
                COMMONDX_STL_NAMESPACE::is_same_v<T, uint16_t>  or
                COMMONDX_STL_NAMESPACE::is_same_v<T, uint32_t>  or
                COMMONDX_STL_NAMESPACE::is_same_v<T, uint64_t>;
        };

        template<template<class> class Complex, class T>
        struct is_unsigned_integral<Complex<T>> : is_unsigned_integral<T> {};

        template<typename T>
        static constexpr bool is_unsigned_integral_v = is_unsigned_integral<T>::value;

        template<typename T>
        struct is_integral {
            static constexpr bool value = is_signed_integral_v<T> or
                                          is_unsigned_integral_v<T>;
        };

        template<typename T>
        static constexpr bool is_integral_v = is_integral<T>::value;
}

#endif // COMMONDX_TRAITS_NUMERIC_TRAITS_HPP
