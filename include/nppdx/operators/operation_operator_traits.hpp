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

#ifndef NPPDX_OPERATORS_OPERATION_OPERATOR_TRAITS_HPP
#define NPPDX_OPERATORS_OPERATION_OPERATOR_TRAITS_HPP

#include "nppdx/operators/operator_type.hpp"
#include "nppdx/operators/halo.hpp"
#include "nppdx/operators/tile_layout.hpp"
#include "nppdx/detail/config.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/detail/database/storage_config.hpp"

#include NPPDX_STD_INCLUDE_TYPE_TRAITS

namespace nppdx {

    // A structure containing typedefs and helper functions depending only on the operation type and not the specific
    // operation parameters.
    // Note: Tag types are used instead of raw enums due to gcc7's defective handling of enum template parameters.
    template<typename TagType>
    struct operation_operator_traits {
        static_assert(!NPPDX_STD::is_same_v<TagType, TagType>,
                      "Unspecialized operation_operator_traits template should not be instantiated");
    };

    namespace detail {
        template<typename T>
        using get_value_type = typename T::value_type;

        template<typename T, typename = void>
        NPPDX_DECL_CI bool has_op_type_val = false;
        template<typename T>
        NPPDX_DECL_CI bool has_op_type_val<
            T, NPPDX_STD::enable_if_t<
                   NPPDX_STD::is_same_v<NPPDX_STD::remove_const_t<decltype(T::op_type_val)>, operator_type>>> = true;

        template<typename T, typename = void>
        NPPDX_DECL_CI bool has_default_op = false;
        template<typename T>
        NPPDX_DECL_CI bool has_default_op<T, NPPDX_STD::void_t<typename T::default_op>> = true;

        template<typename T, typename = void>
        NPPDX_DECL_CI bool has_get_local_halo = false;
        template<typename T>
        NPPDX_DECL_CI bool has_get_local_halo<
            T, NPPDX_STD::enable_if_t<
                   NPPDX_STD::is_same_v<decltype(T::get_local_halo(NPPDX_STD::declval<get_value_type<T>>())), Halo4>>> =
            true;

        template<typename T, typename = void>
        NPPDX_DECL_CI bool has_get_storage_config = false;
        template<typename T>
        NPPDX_DECL_CI bool has_get_storage_config<
            T, NPPDX_STD::enable_if_t<
                   NPPDX_STD::is_same_v<decltype(T::get_storage_config(NPPDX_STD::declval<get_value_type<T>>(),
                                                                       NPPDX_STD::declval<const layout_props&>())),
                                        storage_config>>> = true;

        template<typename T, typename = void>
        NPPDX_DECL_CI bool has_get_output_nominal_tile = false;
        template<typename T>
        NPPDX_DECL_CI bool has_get_output_nominal_tile<
            T, NPPDX_STD::enable_if_t<
                   NPPDX_STD::is_same_v<decltype(T::get_output_nominal_tile(NPPDX_STD::declval<get_value_type<T>>(),
                                                                            NPPDX_STD::declval<uint2>())),
                                        uint2>>> = true;
    } // namespace detail

    template<typename T>
    struct operation_operator_traits_checker: public checker_base<T> {
        static_assert(detail::has_op_type_val<T>);
        static_assert(detail::has_default_op<T>);
        static_assert(detail::has_get_local_halo<T>);
        static_assert(detail::has_get_storage_config<T>);
        static_assert(detail::has_get_output_nominal_tile<T>);
    };

} // namespace nppdx

#endif // NPPDX_OPERATORS_OPERATION_OPERATOR_TRAITS_HPP
