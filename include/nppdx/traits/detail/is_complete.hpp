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

#ifndef NPPDX_TRAITS_DETAIL_IS_COMPLETE_HPP
#define NPPDX_TRAITS_DETAIL_IS_COMPLETE_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/description_traits.hpp"

#include "nppdx/detail/config.hpp"
#include "nppdx/operators/operator_type.hpp"
#include "nppdx/operators/input_output.hpp"
#include "nppdx/traits/detail/description_traits.hpp"


namespace nppdx {
    namespace detail {
        // is_complete_description
        namespace is_complete_description_impl {
            template<class Description, class Enable = void>
            struct helper: NPPDX_STD::false_type {
            };

            template<template<class...> class Description, class... Types>
            struct helper<
                Description<Types...>,
                NPPDX_STD::enable_if_t<commondx::detail::is_description_expression<Description<Types...>>::value>> {
                using description_type = Description<Types...>;

                // ---------
                // Mandatory for NPPDx
                // ---------
                // SM
                static constexpr bool has_this_sm = has_operator_v<operator_type::sm, description_type>;

                // Check if this is a function operation
                static constexpr bool has_this_function = has_operator_v<operator_type::function, description_type>;

                static constexpr bool has_this_block = has_operator_v<operator_type::block, description_type>;

                // InputOutput (required for NPPDx input output operations, but NOT for function operations)
                static constexpr bool has_this_input_output =
                    has_operator_v<operator_type::input_output, description_type>;
                // If it has it get the direction
                static constexpr bool has_this_input_output_direction =
                    has_operator_v<operator_type::input_output, description_type>;
                using this_input_output_direction = get_or_default_t<operator_type::input_output, description_type,
                                                                     InputOutput<input_output_direction::ingest>>;

                // Input or output format depending on InputOutput direction
                static constexpr bool has_this_input_output_format =
                    this_input_output_direction::value == input_output_direction::ingest
                        ? has_operator_v<operator_type::input_format, description_type>
                        : has_operator_v<operator_type::output_format, description_type>;

                // Complete if:
                // - Has SM (always required)
                // - Has an execution-kind operator (currently only Block() is supported)
                // - AND either:
                //   - Is a function operation (has function operator)
                //   - OR is an input/output operation (has input_output and appropriate format)
                static constexpr bool value =
                    has_this_sm && has_this_block &&
                    (has_this_function || (has_this_input_output && has_this_input_output_format));
            };
        } // namespace is_complete_description_impl

        template<class Description>
        struct is_complete_description:
            NPPDX_STD::integral_constant<bool, is_complete_description_impl::helper<Description>::value> {
        };
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_TRAITS_DETAIL_IS_COMPLETE_HPP
