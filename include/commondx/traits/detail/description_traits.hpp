/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMONDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
#define COMMONDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        /// is_expression

        template<class T>
        struct is_expression: public COMMONDX_STL_NAMESPACE::is_base_of<expression, T> {};

        template<class T, class U>
        struct are_expressions:
            public COMMONDX_STL_NAMESPACE::integral_constant<bool, is_expression<T>::value && is_expression<U>::value> {};

        /// is_operator_expression

        template<class T>
        struct is_operator_expression: public COMMONDX_STL_NAMESPACE::is_base_of<operator_expression, T> {};

        template<class T, class U>
        struct are_operator_expressions:
            public COMMONDX_STL_NAMESPACE::integral_constant<bool, is_operator_expression<T>::value && is_operator_expression<U>::value> {
        };

        /// is_description_expression

        template<class T>
        struct is_description_expression: public COMMONDX_STL_NAMESPACE::is_base_of<description_expression, T> {};

        /// is_operator definition

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class T>
        struct is_operator: COMMONDX_STL_NAMESPACE::false_type {};

        /// has_n_of

        namespace has_n_of_impl {
            template<unsigned int Counter, class OperatorTypeClass, OperatorTypeClass OperatorType, class Head, class... Types>
            struct counter_helper {
                static constexpr unsigned int value = is_operator<OperatorTypeClass, OperatorType, Head>::value
                                                          ? counter_helper<(Counter + 1), OperatorTypeClass, OperatorType, Types...>::value
                                                          : counter_helper<Counter, OperatorTypeClass, OperatorType, Types...>::value;
            };

            template<unsigned int Counter, class OperatorTypeClass, OperatorTypeClass OperatorType, class Head>
            struct counter_helper<Counter, OperatorTypeClass, OperatorType, Head> {
                static constexpr unsigned int value = is_operator<OperatorTypeClass, OperatorType, Head>::value ? Counter + 1 : Counter;
            };

            template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Operator>
            struct counter: COMMONDX_STL_NAMESPACE::integral_constant<unsigned int, is_operator<OperatorTypeClass, OperatorType, Operator>::value> {};

            template<class OperatorTypeClass, OperatorTypeClass OperatorType, template<class...> class Description, class... Types>
            struct counter<OperatorTypeClass, OperatorType, Description<Types...>>:
                COMMONDX_STL_NAMESPACE::integral_constant<unsigned int, counter_helper<0, OperatorTypeClass,OperatorType, Types...>::value> {};


            template<class OperatorTypeClass, OperatorTypeClass OperatorType, class T>
            struct counter_removed_cvref: counter<OperatorTypeClass, OperatorType, COMMONDX_STL_NAMESPACE::remove_cv_t<COMMONDX_STL_NAMESPACE::remove_reference_t<T>>> {};

        } // namespace has_n_of_impl

        template<unsigned int N, class OperatorTypeClass, OperatorTypeClass OperatorType, class Description>
        struct has_n_of: COMMONDX_STL_NAMESPACE::integral_constant<bool, has_n_of_impl::counter_removed_cvref<OperatorTypeClass, OperatorType, Description>::value == N> {};

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description>
        struct has_at_most_one_of:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, (has_n_of_impl::counter_removed_cvref<OperatorTypeClass, OperatorType, Description>::value <= 1)> {};

        /// has_operator

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description>
        struct has_operator:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, (has_n_of_impl::counter_removed_cvref<OperatorTypeClass, OperatorType, Description>::value > 0)> {};
    } // namespace detail
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
