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

#ifndef NPPDX_TRAITS_NPPDX_TRAITS_HPP
#define NPPDX_TRAITS_NPPDX_TRAITS_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"
#include "commondx/traits/detail/get.hpp"
#include "commondx/traits/dx_traits.hpp"

#include "nppdx/detail/backend/resize_operations.hpp"
#include "nppdx/detail/nppdx_description_fd.hpp"
#include "nppdx/detail/decl.hpp"

#include "nppdx/operators.hpp"
#include "nppdx/operators/input_output.hpp"
#include "nppdx/types.hpp"

#include "nppdx/traits/detail/description_traits.hpp"
#include "nppdx/traits/detail/is_complete.hpp"
#include "nppdx/detail/config.hpp"

#include NPPDX_STD_INCLUDE_TUPLE

namespace nppdx {
    namespace detail {
        // NPPDx-specific detail implementations can go here
    } // namespace detail

    // ------------------
    // Execution checkers
    // ------------------

    // is_block
    template<class Description>
    struct is_block {
    public:
        static constexpr bool value = detail::has_operator_v<operator_type::block, Description>;
    };

    // ----------------
    // Operator getters
    // ----------------


    // type_of
    template<class Description>
    using type_of = commondx::data_type_of<operator_type, Description, detail::default_type_operator>;

    template<class Description>
    inline constexpr type type_of_v = type_of<Description>::value;

    // sm_of
    template<class Description>
    using sm_of = commondx::sm_of<operator_type, Description>;
    template<class Description>
    inline constexpr unsigned int sm_of_v = sm_of<Description>::value;

    // block_dim_of
    template<class Description>
    using block_dim_of = commondx::block_dim_of<operator_type, Description>;
    template<class Description>
    inline constexpr dim3 block_dim_of_v = block_dim_of<Description>::value;

    // processing_type_of
    template<class Description>
    struct processing_type_of {
        using type = typename Description::processing_type;
    };
    template<class Description>
    using processing_type_of_t = typename processing_type_of<Description>::type;

    // elements_per_thread_of
    template<class Description>
    struct elements_per_thread_of {
        static constexpr auto value = Description::elements_per_thread;
    };
    template<class Description>
    inline constexpr auto elements_per_thread_of_v = elements_per_thread_of<Description>::value;

    // input_output_direction_of
    template<class Description>
    struct input_output_direction_of {
    private:
        using this_input_output_direction = detail::get_or_default_t<operator_type::input_output, Description,
                                                                     InputOutput<input_output_direction::ingest>>;

    public:
        using value_type                  = input_output_direction;
        static constexpr value_type value = this_input_output_direction::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };
    template<class Description>
    inline constexpr auto input_output_direction_of_v = input_output_direction_of<Description>::value;

    // input_format_of - only valid for ingest operations
    template<class Description>
    struct input_format_of {
    private:
        using this_packing_format =
            detail::get_or_default_t<operator_type::input_format, Description, InputFormat<packing_format::rgb24>>;

    public:
        using value_type                  = packing_format;
        static constexpr value_type value = this_packing_format::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };
    template<class Description>
    inline constexpr auto input_format_of_v = input_format_of<Description>::value;

    // output_format_of - only valid for exgest operations
    template<class Description>
    struct output_format_of {
    private:
        using this_packing_format =
            detail::get_or_default_t<operator_type::output_format, Description, OutputFormat<packing_format::rgb24>>;

    public:
        using value_type                  = packing_format;
        static constexpr value_type value = this_packing_format::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };
    template<class Description>
    inline constexpr auto output_format_of_v = output_format_of<Description>::value;

    // packing_format_of - returns the appropriate format based on direction
    template<class Description>
    struct packing_format_of {
    public:
        using value_type = packing_format;
        static constexpr packing_format value =
            input_output_direction_of_v<Description> == input_output_direction::ingest
                ? input_format_of<Description>::value
                : output_format_of<Description>::value;
        constexpr operator value_type() const noexcept { return value; }
    };
    template<class Description>
    inline constexpr auto packing_format_of_v = packing_format_of<Description>::value;


    // tile_size_of - returns tile size as dim3
    template<class Description>
    struct tile_size_of {
    private:
        using this_tile_size = detail::get_or_default_t<operator_type::tile_size, Description, TileSize<16, 16>>;

    public:
        static constexpr uint2 value = this_tile_size::value;
        constexpr              operator uint2() const noexcept { return value; }
    };

    template<class Description>
    inline constexpr auto tile_size_of_v = tile_size_of<Description>::value;

    // memory_halo_of - returns tile halo capacity
    template<class Description>
    struct memory_halo_of {
    private:
        using this_memory_halo = detail::get_or_default_t<operator_type::memory_halo, Description, EmptyMemoryHalo>;

    public:
        static constexpr Halo4 value = this_memory_halo::value;
    };

    template<class Description>
    inline constexpr Halo4 memory_halo_of_v = memory_halo_of<Description>::value;

    // cumulative_halo_of - returns cumulative halo (overlap between tiles)
    template<class Description>
    struct cumulative_halo_of {
    private:
        using this_cumulative_halo =
            detail::get_or_default_t<operator_type::cumulative_halo, Description, EmptyCumulativeHalo>;

    public:
        static constexpr Halo4 value = this_cumulative_halo::value;
    };

    template<class Description>
    inline constexpr Halo4 cumulative_halo_of_v = cumulative_halo_of<Description>::value;

    // function_of - returns the function type for function operations
    template<class Description>
    struct function_of {
    private:
        using this_function =
            detail::get_or_default_t<operator_type::function, Description, Function<function::color_convert>>;

    public:
        using value_type                  = function;
        static constexpr value_type value = this_function::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };
    template<class Description>
    inline constexpr auto function_of_v = function_of<Description>::value;

    template<class Description>
    NPPDX_DECL_C auto get_operation_tag_type() {
        if constexpr (detail::has_operator_v<operator_type::input_output, Description>) {
            return input_output_direction_tag<input_output_direction_of_v<Description>> {};
        } else {
            return function_tag<function_of_v<Description>> {};
        }
    }

    template<class Description>
    struct operation_operator_of {
        using tag_type = decltype(get_operation_tag_type<Description>());
        using traits   = operation_operator_traits<tag_type>;
        using type     = detail::get_or_default_t<traits::op_type_val, Description, typename traits::default_op>;
    };

    template<class Description>
    using operation_operator_of_t = typename operation_operator_of<Description>::type;

    // local_halo_of - returns operation halo requirements
    template<class Description>
    struct local_halo_of {
    public:
        using op_op_data = operation_operator_of<Description>;
        static constexpr Halo4 value =
            operation_operator_traits<typename op_op_data::tag_type>::get_local_halo(op_op_data::type::value);
    };

    template<class Description>
    inline constexpr Halo4 local_halo_of_v = local_halo_of<Description>::value;


    // --------------------------
    // General Description traits
    // --------------------------

    template<class Description>
    using is_npp = commondx::is_dx_expression<Description>;

    template<class Description>
    inline constexpr bool is_npp_v = is_npp<Description>::value;

    template<class Description>
    using is_nppdx_execution = commondx::is_dx_execution_expression<operator_type, Description>;

    template<class Description>
    inline constexpr bool is_nppdx_execution_v = is_nppdx_execution<Description>::value;

    template<class Description>
    using is_complete_npp = commondx::is_complete_dx_expression<Description, detail::is_complete_description>;

    template<class Description>
    inline constexpr bool is_complete_npp_v = is_complete_npp<Description>::value;

    template<class Description>
    using is_complete_nppdx_execution =
        commondx::is_complete_dx_execution_expression<operator_type, Description, detail::is_complete_description>;

    template<class Description>
    inline constexpr bool is_complete_nppdx_execution_v = is_complete_nppdx_execution<Description>::value;

    template<class Description>
    using extract_nppdx_description =
        commondx::extract_dx_description<detail::nppdx_description, Description, operator_type>;

    template<class Description>
    using extract_nppdx_description_t = typename extract_nppdx_description<Description>::type;

    // ========================================================================
    // Shared memory storage traits
    // ========================================================================

    // inputs_of - returns Description::inputs
    template<class Description>
    struct inputs_of {
        using type = typename Description::inputs;
    };

    template<class Description>
    using inputs_of_t = typename inputs_of<Description>::type;

    // outputs_of - returns Description::outputs
    template<class Description>
    struct outputs_of {
        using type = typename Description::outputs;
    };

    template<class Description>
    using outputs_of_t = typename outputs_of<Description>::type;

    // temp_of - returns Description::temp
    template<class Description>
    struct temp_of {
        using type = typename Description::temp;
    };

    template<class Description>
    using temp_of_t = typename temp_of<Description>::type;

    template<class Description, NPPDX_STD::size_t Index = 0>
    using input_storage_of_t = NPPDX_STD::tuple_element_t<Index, inputs_of_t<Description>>;

    template<class Description, NPPDX_STD::size_t Index = 0>
    using output_storage_of_t = NPPDX_STD::tuple_element_t<Index, outputs_of_t<Description>>;

    template<class Description, NPPDX_STD::size_t Index = 0>
    using temp_storage_of_t = NPPDX_STD::tuple_element_t<Index, temp_of_t<Description>>;

} // namespace nppdx

#endif // NPPDX_NPPDX_TRAITS_TRAITS_HPP
