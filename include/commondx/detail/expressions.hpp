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

#ifndef COMMONDX_DETAIL_EXPRESSIONS_HPP
#define COMMONDX_DETAIL_EXPRESSIONS_HPP

#include "commondx/detail/stl/type_traits.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        struct expression {};
        struct operator_expression: expression {};
        struct block_operator_expression: operator_expression {};
        struct device_operator_expression: operator_expression {};

        struct description_expression: expression {};
        struct execution_description_expression: description_expression {};

        template<class ValueType, ValueType Value>
        struct constant_operator_expression:
            public operator_expression,
            public COMMONDX_STL_NAMESPACE::integral_constant<ValueType, Value> {};

        template<class ValueType, ValueType Value>
        struct constant_block_operator_expression:
            public block_operator_expression,
            public COMMONDX_STL_NAMESPACE::integral_constant<ValueType, Value> {};
    } // namespace detail
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_DETAIL_EXPRESSIONS_HPP
