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

#ifndef NPPDX_DETAIL_NPPDX_DESCRIPTION_HPP
#define NPPDX_DETAIL_NPPDX_DESCRIPTION_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"
#include "commondx/detail/expressions.hpp"

#include "nppdx/operators/operator_type.hpp"
#include "nppdx/operators/tile_size.hpp"
#include "nppdx/operators/sm.hpp"
#include "nppdx/operators/block_dim.hpp"
#include "nppdx/operators/input_output.hpp"
#include "nppdx/operators/function.hpp"
#include "nppdx/traits/detail/description_traits.hpp"
#include "nppdx/traits/detail/is_complete.hpp"
namespace nppdx {
    namespace detail {
        template<class... Operators>
        class nppdx_operator_wrapper: public commondx::detail::description_expression
        {
        };
        template<class... Operators>
        class nppdx_description: public commondx::detail::description_expression
        {

            using description_type = nppdx_operator_wrapper<Operators...>;


        protected:
            // SM (compute capability)
            static constexpr bool has_sm    = has_operator<operator_type::sm, description_type>::value;
            using dummy_default_sm          = SM<700>;
            using this_sm                   = get_or_default_t<operator_type::sm, description_type, dummy_default_sm>;
            static constexpr auto this_sm_v = this_sm::value;

            // Block execution
            static constexpr bool has_block = has_operator_v<operator_type::block, description_type>;

            static constexpr bool has_block_dim = has_operator_v<operator_type::block_dim, description_type>;
            using dummy_default_block_dim       = BlockDim<256, 1, 1>;

            using this_block_dim =
                get_or_default_t<operator_type::block_dim, description_type, dummy_default_block_dim>;

            // Template member access - should be safe with default
            static constexpr unsigned int this_block_dim_x = this_block_dim::x;
            static constexpr unsigned int this_block_dim_y = this_block_dim::y;
            static constexpr unsigned int this_block_dim_z = this_block_dim::z;

            // Provide dim3 for test compatibility
            static constexpr dim3 this_block_dim_v = {this_block_dim_x, this_block_dim_y, this_block_dim_z};

            // InputOutput
            static constexpr bool has_input_output = has_operator_v<operator_type::input_output, description_type>;
            static constexpr auto this_input_output =
                get_or_default_t<operator_type::input_output, description_type,
                                 InputOutput<input_output_direction::ingest>>::value;

            // InputFormat - only valid for ingest operations
            static constexpr bool has_input_format  = has_operator_v<operator_type::input_format, description_type>;
            static constexpr auto this_input_format = get_or_default_t<operator_type::input_format, description_type,
                                                                       InputFormat<packing_format::none>>::value;

            // OutputFormat - only valid for exgest operations
            static constexpr bool has_output_format  = has_operator_v<operator_type::output_format, description_type>;
            static constexpr auto this_output_format = get_or_default_t<operator_type::output_format, description_type,
                                                                        OutputFormat<packing_format::none>>::value;

            // Format validation based on direction
            static constexpr bool is_ingest = has_input_output && (this_input_output == input_output_direction::ingest);
            static constexpr bool is_exgest = has_input_output && (this_input_output == input_output_direction::exgest);
            // If does not have inputoutput always true, if it does, check the correct format
            static constexpr bool has_correct_input_output_format =
                !has_input_output || (is_ingest && has_input_format && !has_output_format) ||
                (is_exgest && has_output_format && !has_input_format);

            // Function operations validation
            static constexpr bool has_function_operator = has_operator_v<operator_type::function, description_type>;
            static constexpr auto this_function_operator =
                get_or_default_t<operator_type::function, description_type, Function<function::none>>::value;
            static constexpr bool is_function = has_function_operator && (this_function_operator != function::none);
            //Color conversion validation for function operations
            static constexpr bool color_convert_operator =
                has_operator_v<operator_type::color_convert, description_type>;
            static constexpr bool has_correct_color_conversion =
                (this_function_operator != function::color_convert) || color_convert_operator;
            // TileSize
            static constexpr unsigned int default_tile_x = 16;
            static constexpr unsigned int default_tile_y = 16;
            static constexpr bool         has_tile_size  = has_operator_v<operator_type::tile_size, description_type>;
            using dummy_default_tile_size                = TileSize<default_tile_x, default_tile_y>;
            using this_tile_size =
                get_or_default_t<operator_type::tile_size, description_type, dummy_default_tile_size>;
            static constexpr auto this_tile_size_x = this_tile_size::value.x;
            static constexpr auto this_tile_size_y = this_tile_size::value.y;


            // Suggested tile size - description trait
            static constexpr dim3 tile_size = dim3 {this_tile_size_x, this_tile_size_y, 1};

            // Constraints - ensure only one of each operator type
            static constexpr bool has_one_block_dim = has_at_most_one_of_v<operator_type::block_dim, description_type>;
            static constexpr bool has_one_block     = has_at_most_one_of_v<operator_type::block, description_type>;
            static constexpr bool has_one_sm        = has_at_most_one_of_v<operator_type::sm, description_type>;
            static constexpr bool has_one_input_format =
                has_at_most_one_of_v<operator_type::input_format, description_type>;
            static constexpr bool has_one_output_format =
                has_at_most_one_of_v<operator_type::output_format, description_type>;
            static constexpr bool has_one_tile_size = has_at_most_one_of_v<operator_type::tile_size, description_type>;
            //Can only have one function expression or inputoutput expression exclusively
            static constexpr bool has_one_function =
                !has_function_operator ||
                (has_at_most_one_of_v<operator_type::function, description_type> && !has_input_output);
            static constexpr bool has_one_input_output =
                !has_input_output ||
                (has_at_most_one_of_v<operator_type::input_output, description_type> && !has_function_operator);

            // Halo validation, only one halo type can be present
            static constexpr bool has_one_memory_halo =
                has_at_most_one_of_v<operator_type::memory_halo, description_type>;
            static constexpr bool has_one_cumulative_halo =
                has_at_most_one_of_v<operator_type::cumulative_halo, description_type>;

            // Static assertions
            static_assert(has_one_block_dim, "Can't create nppdx function with two BlockDim<> expressions");
            static_assert(has_one_block, "Can't create nppdx function with two Block expressions");
            static_assert(has_one_sm, "Can't create nppdx function with two SM expressions");
            static_assert(has_one_input_format, "Can't create nppdx function with two InputFormat expressions");
            static_assert(has_one_output_format, "Can't create nppdx function with two OutputFormat expressions");
            static_assert(has_one_tile_size, "Can't create nppdx function with two TileSize expressions");
            static_assert(has_one_function,
                          "Can't create nppdx function with two Function expressions or with additional InputOutput "
                          "expressions.");
            static_assert(has_one_input_output,
                          "Can't create nppdx function with two InputOutput expressions or with additional Function "
                          "expressions.");
            static_assert(has_one_memory_halo, "Can't create nppdx function with two or more MemoryHalo expressions");
            static_assert(has_one_cumulative_halo,
                          "Can't create nppdx function with two or more CumulativeHalo expressions");

            static_assert(has_correct_input_output_format && has_correct_color_conversion,
                          "Ingest operations require InputFormat (not OutputFormat), "
                          "Exgest operations require OutputFormat (not InputFormat), "
                          "Color conversion operations require ColorConvert<> operator");

            // Completion check - simplified (Block() added by execution framework)
            static constexpr bool is_complete_v = is_complete_description<description_type>::value;
        };

        template<>
        class nppdx_description<>: public commondx::detail::description_expression
        {
        };

    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_NPPDX_DESCRIPTION_HPP