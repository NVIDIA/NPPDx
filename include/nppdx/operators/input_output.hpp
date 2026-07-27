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

#ifndef NPPDX_OPERATORS_INPUT_OUTPUT_HPP
#define NPPDX_OPERATORS_INPUT_OUTPUT_HPP


#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"
#include "nppdx/operators/operator_type.hpp"
#include "nppdx/operators/formats.hpp"
#include "nppdx/operators/operation_operator_traits.hpp"
#include "nppdx/detail/config.hpp"
#include "nppdx/detail/database/storage_config.hpp"

namespace nppdx {
    // Direction enums
    enum class input_output_direction
    {
        ingest,
        exgest
    };

    template<input_output_direction TagValue>
    struct input_output_direction_tag {
        using type                                    = input_output_direction;
        static constexpr input_output_direction value = TagValue;
    };

    // Simplified operators using enums
    template<input_output_direction Value>
    struct InputOutput: public commondx::detail::constant_operator_expression<input_output_direction, Value> {
    };

    template<packing_format Value>
    struct InputFormat: public commondx::detail::constant_operator_expression<packing_format, Value> {
        using tag_type = input_output_direction_tag<input_output_direction::ingest>;
    };

    // Used by both InputFormat and OutputFormat.
    NPPDX_DECL_NCFHD Halo4 compute_local_halo(const packing_format&) {
        return Halo4::make_empty();
    }

    template<>
    struct operation_operator_traits<input_output_direction_tag<input_output_direction::ingest>> {
        using value_type                        = packing_format;
        NPPDX_DECL_SC operator_type op_type_val = operator_type::input_format;
        NPPDX_DECL_NSCHD Halo4      get_local_halo(const value_type& val) { return compute_local_halo(val); }
        NPPDX_DECL_NSCHD detail::storage_config get_storage_config(const value_type&, const layout_props& layout) {
            return detail::make_inplace_storage_config(layout);
        }
        NPPDX_DECL_NSCHD uint2 get_output_nominal_tile(const value_type&, uint2 input_nominal) { return input_nominal; }
        using default_op = InputFormat<packing_format::rgb24>;
    };

    template<packing_format Value>
    struct OutputFormat: public commondx::detail::constant_operator_expression<packing_format, Value> {
        using tag_type = input_output_direction_tag<input_output_direction::exgest>;
    };

    template<>
    struct operation_operator_traits<input_output_direction_tag<input_output_direction::exgest>> {
        using value_type                        = packing_format;
        NPPDX_DECL_SC operator_type op_type_val = operator_type::output_format;
        NPPDX_DECL_NSCHD Halo4      get_local_halo(const value_type&) { return Halo4::make_empty(); }
        NPPDX_DECL_NSCHD detail::storage_config get_storage_config(const value_type&, const layout_props& layout) {
            return detail::make_inplace_storage_config(layout);
        }
        NPPDX_DECL_NSCHD uint2 get_output_nominal_tile(const value_type&, uint2 input_nominal) { return input_nominal; }
        using default_op = OutputFormat<packing_format::rgb24>;
    };

} // namespace nppdx

namespace commondx::detail {
    // InputOutput operator specializations
    template<nppdx::input_output_direction Direction>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::input_output, nppdx::InputOutput<Direction>>:
        NPPDX_STD::true_type {
    };

    template<nppdx::input_output_direction Direction>
    struct get_operator_type<nppdx::operator_type, nppdx::InputOutput<Direction>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::input_output;
    };

    // InputFormat operator specializations
    template<nppdx::packing_format Format>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::input_format, nppdx::InputFormat<Format>>:
        NPPDX_STD::true_type {
    };

    template<nppdx::packing_format Format>
    struct get_operator_type<nppdx::operator_type, nppdx::InputFormat<Format>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::input_format;
    };

    // OutputFormat operator specializations
    template<nppdx::packing_format Format>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::output_format, nppdx::OutputFormat<Format>>:
        NPPDX_STD::true_type {
    };

    template<nppdx::packing_format Format>
    struct get_operator_type<nppdx::operator_type, nppdx::OutputFormat<Format>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::output_format;
    };

} // namespace commondx::detail

#endif // NPPDX_OPERATORS_INPUT_OUTPUT_HPP
