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

#ifndef COMMONDX_TRAITS_DETAIL_GET_HPP
#define COMMONDX_TRAITS_DETAIL_GET_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        namespace get_impl {
            template<class OperatorTypeClass, OperatorTypeClass OperatorType, class T>
            struct helper {
                using type = typename COMMONDX_STL_NAMESPACE::
                    conditional<is_operator<OperatorTypeClass, OperatorType, T>::value, T, void>::type;
            };

            // clang-format off
            template<class OperatorTypeClass, OperatorTypeClass OperatorType,
                     template<class...> class DescriptionType,
                     class TypeHead,
                     class... TailTypes>
            struct helper<OperatorTypeClass, OperatorType, DescriptionType<TypeHead, TailTypes...>> {

                using type = typename COMMONDX_STL_NAMESPACE::conditional<
                    is_operator<OperatorTypeClass, OperatorType, TypeHead>::value,
                    TypeHead,
                    typename helper<OperatorTypeClass, OperatorType, DescriptionType<TailTypes...>>::type>::type;
            };
            // clang-format on
        } // namespace get_impl

        /// get

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description>
        struct get {
            using type = typename get_impl::helper<OperatorTypeClass, OperatorType, COMMONDX_STL_NAMESPACE::remove_cv_t<COMMONDX_STL_NAMESPACE::remove_reference_t<Description>>>::type;
        };

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description>
        using get_t = typename get<OperatorTypeClass, OperatorType, Description>::type;

        /// get_or_default

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description, class Default = void>
        struct get_or_default {
        private:
            using get_type = get_t<OperatorTypeClass, OperatorType, Description>;

        public:
            using type = typename COMMONDX_STL_NAMESPACE::
                conditional<COMMONDX_STL_NAMESPACE::is_void<get_type>::value, Default, get_type>::type;
        };

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description, class Default = void>
        using get_or_default_t = typename get_or_default<OperatorTypeClass, OperatorType, Description, Default>::type;
    } // namespace detail
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TRAITS_DETAIL_GET_HPP
