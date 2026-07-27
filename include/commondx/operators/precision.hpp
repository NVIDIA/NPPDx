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

#ifndef COMMONDX_OPERATORS_PRECISION_HPP
#define COMMONDX_OPERATORS_PRECISION_HPP

#include <cuda_fp16.h>

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/expressions.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        template<class T, class... SupportedTypes>
        struct is_supported_type:
            COMMONDX_STL_NAMESPACE::integral_constant<bool,
                                      (COMMONDX_STL_NAMESPACE::is_same<SupportedTypes, typename COMMONDX_STL_NAMESPACE::remove_cv<T>::type>::value ||
                                       ...)> {};
    } // namespace detail

    template<class T, class... SupportedPrecisions>
    struct PrecisionBase: detail::operator_expression {
        using type = typename COMMONDX_STL_NAMESPACE::remove_cv<T>::type;
        static_assert(detail::is_supported_type<type, SupportedPrecisions...>::value, "Unsupported precision type.");
    };
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_TYPE_HPP
