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

#ifndef NPPDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
#define NPPDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/description_traits.hpp"

#include "nppdx/operators/operator_type.hpp"

namespace nppdx {
    namespace detail {
        // Extract operator from Description type
        // Void if absent
        template<operator_type OperatorType, class Description>
        using get_t = commondx::detail::get_t<operator_type, OperatorType, Description>;

        // Extract operator from Description type
        // Default if absent
        template<operator_type OperatorType, class Description, class Default = void>
        using get_or_default_t = commondx::detail::get_or_default_t<operator_type, OperatorType, Description, Default>;

        // Check if Description type contains operator
        template<operator_type OperatorType, class Description>
        using has_operator = commondx::detail::has_operator<operator_type, OperatorType, Description>;

        template<operator_type OperatorType, class Description>
        inline constexpr bool has_operator_v = has_operator<OperatorType, Description>::value;

        // Check if Description type contains at most 1 operator
        template<operator_type OperatorType, class Description>
        using has_at_most_one_of = commondx::detail::has_at_most_one_of<operator_type, OperatorType, Description>;

        template<operator_type OperatorType, class Description>
        inline constexpr bool has_at_most_one_of_v = has_at_most_one_of<OperatorType, Description>::value;
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
