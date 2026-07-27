/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMONDX_TRAITS_DETAIL_MAKE_CUDAVALUE_TYPE_HPP
#define COMMONDX_TRAITS_DETAIL_MAKE_CUDAVALUE_TYPE_HPP

#include <cuComplex.h>

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/operators/precision.hpp"
#include "commondx/operators/type.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        template<data_type DataType, class Precision>
        struct make_cudavalue_type {
            static_assert(is_supported_type<Precision, __half, float, double>::value, "Precision must be __half, double, or float");
            using type = void;
        };

        template<class Precision>
        struct make_cudavalue_type<data_type::real, Precision> {
            using type = Precision;
        };

        template<>
        struct make_cudavalue_type<data_type::complex, __half> {
            using type = __half2;
        };

        template<>
        struct make_cudavalue_type<data_type::complex, float> {
            using type = cuComplex;
        };

        template<>
        struct make_cudavalue_type<data_type::complex, double> {
            using type = cuDoubleComplex;
        };

        template<data_type DataType, class Precision>
        using make_cudavalue_type_t = typename make_cudavalue_type<DataType, Precision>::type;
    } // namespace detail
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TRAITS_DETAIL_MAKE_CUDAVALUE_TYPE_HPP
