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

#ifndef NPPDX_DETAIL_NPPDX_EXECUTION_HPP
#define NPPDX_DETAIL_NPPDX_EXECUTION_HPP

#include "nppdx/detail/nppdx_is_supported.hpp"
#include "nppdx/detail/nppdx_description.hpp"
#include "nppdx/detail/nppdx_exec_op.hpp"
#include "nppdx/detail/database/operation_database.hpp"
#include "nppdx/detail/database/storage_config.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/operators/tile_layout.hpp"
#include "nppdx/shared_memory.hpp"
#include "nppdx/types.hpp"
#include "nppdx/utils.hpp"

namespace nppdx {

    namespace detail {

        template<class... Operators>
        class nppdx_execution_partial:
            public nppdx_description<Operators...>,
            public commondx::detail::execution_description_expression
        {
            using base_type = nppdx_description<Operators...>;
        };

        template<class... Operators>
        class nppdx_execution:
            public nppdx_description<Operators...>,
            public commondx::detail::execution_description_expression
        {
        private:
            using base_type      = nppdx_description<Operators...>;
            using execution_type = nppdx_execution<Operators...>;

            // Note: This is essentially a placeholder, the support requirements are not yet defined.
            NPPDX_DECL_SC bool is_supported_configuration = is_supported_v<base_type, base_type::this_sm_v>;
            static_assert(is_supported_configuration, "This operation configuration is not supported");


            NPPDX_DECL_SC auto get_bare_exec_operator() {
                using op_op_data = operation_operator_of<base_type>;
                using tag_type   = typename op_op_data::tag_type;

                simple_optional<dim3> a_block_dim = simple_nullopt;
                if constexpr (has_operator_v<operator_type::block_dim, base_type>) {
                    a_block_dim = block_dim_of_v<base_type>;
                }

                // As a convenient shortcut, an ingest operation is allowed to
                // not specify a cumulative halo. In that case, it will be set
                // equal to the tile halo.
                Halo4 a_memory_halo     = memory_halo_of_v<base_type>;
                Halo4 a_cumulative_halo = cumulative_halo_of_v<base_type>;
                if constexpr (has_operator_v<operator_type::input_output, base_type>) {
                    if constexpr (input_output_direction_of_v<base_type> == input_output_direction::ingest &&
                                  !has_operator_v<operator_type::cumulative_halo, base_type>) {
                        a_cumulative_halo = a_memory_halo;
                    }
                }

                return ExecOperator<tag_type> {
                    /*sm=*/int(sm_of_v<base_type>),
                    /*block_dim=*/a_block_dim,
                    /*layout=*/ {tile_size_of_v<base_type>, a_memory_halo, a_cumulative_halo},
                    /*op_data=*/op_op_data::type::value,
                };
            }

            NPPDX_DECL_SC auto bare_value = get_bare_exec_operator();

        public:
            NPPDX_DECL_SC auto value = normalize_exec_operator(bare_value);

            NPPDX_DECL_SC auto recommended_op_config = get_recommended_op_config(value);

            using impl =
                op_impl<operation_operator_of_t<base_type>, NPPDX_MAKE_LAYOUT(value.layout),
                        NPPDX_MAKE_CELL_CONFIG(recommended_op_config.cell_cfg), value.sm, value.block_threads()>;


            static constexpr bool has_function_operator = has_operator<operator_type::function, base_type>::value;
            static constexpr bool has_input_output_operator =
                has_operator<operator_type::input_output, base_type>::value;

            // Expose local halo for the database entry
            static constexpr Halo4 local_halo = value.local_halo();

            // Processing type - always 3-channel float for NPPDx internal processing
            using processing_type                   = float;
            static constexpr int channel_count      = 3;
            static constexpr int number_of_channels = channel_count;

            // Tile and block dimensions
            static constexpr unsigned int tile_size_x = value.layout.nominal_size.x;
            static constexpr unsigned int tile_size_y = value.layout.nominal_size.y;
            // Effective tile size includes halo (for shared memory allocation)
            static constexpr unsigned int effective_tile_x = value.layout.memory_size().x;
            static constexpr unsigned int effective_tile_y = value.layout.memory_size().y;
            static constexpr dim3         block_dim        = value.block_dim();

            // Suggested tile size from database
            static constexpr dim3 suggested_tile_size = dim3(tile_size_x, tile_size_y, 1);

            // Grid calculation helper
            static constexpr dim3 calculate_grid_dim(int width, int height) {
                return dim3(ceil_div(uint(width), tile_size_x), ceil_div(uint(height), tile_size_y), 1);
            }

            static constexpr unsigned int elements_per_thread = recommended_op_config.cell_cfg.elements_per_cell();
            static constexpr unsigned int block_threads       = value.block_threads();

            // Input, output and temp memory.
            using inputs  = NPPDX_MAKE_SLOT_STORAGE(processing_type, value.storage_config().input);
            using outputs = NPPDX_MAKE_SLOT_STORAGE(processing_type, value.storage_config().output);
            using temp    = NPPDX_MAKE_SLOT_STORAGE(processing_type, value.storage_config().temp);

            static constexpr tile_shape input_storage  = value.storage_config().input;
            static constexpr tile_shape output_storage = value.storage_config().output;
            static constexpr tile_shape temp_storage   = value.storage_config().temp;

            static constexpr unsigned int shared_memory_size =
                shared_memory::detail::compute_operation_storage_from_tuples<inputs, outputs, temp>();

            // ========================================================================
            // Register-based execution methods
            //

            // Public execute method for function operations (register-based)
            NPPDX_DECL_D void execute(processing_type* data, int width, int height) const {
                static_assert(has_function_operator, "This execute method is only available for function operations");
                return impl::execute(data, width, height);
            }

            // Public execute method for ingest/exgest operations (register-based)
            template<typename InputType, typename OutputType>
            NPPDX_DECL_D void execute(const InputType* input, OutputType* output, int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                return impl::execute(input, output, width, height);
            }

            template<typename InputType, int PlaneCount, typename OutputType>
            __device__ void execute(const ImageLayout<InputType, PlaneCount>& input, OutputType* output, int width,
                                    int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::ingest,
                              "This execute signature is for ingest operations only");
                return impl::execute(input, output, width, height);
            }

            template<typename InputType, typename OutputType, int PlaneCount>
            __device__ void execute(InputType* input, const ImageLayout<OutputType, PlaneCount>& output, int width,
                                    int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::exgest,
                              "This execute signature is for exgest operations only");
                return impl::execute(input, output, width, height);
            }

            // Texture ingest: TexObjT -> float registers
            __device__ void execute(TexObjT input, processing_type* output, int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::ingest,
                              "TexObjT input is only valid for ingest operations");
                return impl::execute(input, output, width, height);
            }

            // Surface exgest: float registers -> SurfObjT
            __device__ void execute(processing_type* input, SurfObjT output, int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::exgest,
                              "SurfObjT output is only valid for exgest operations");
                return impl::execute(input, output, width, height);
            }

            // Multi-plane texture ingest (e.g. NV12): TexObjPlanes -> float registers
            template<int NumPlanes>
            __device__ void execute(const TexObjPlanes<NumPlanes>& input, processing_type* output, int width,
                                    int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::ingest,
                              "TexObjPlanes input is only valid for ingest operations");
                return impl::execute(input, output, width, height);
            }

            // Multi-plane surface exgest (e.g. NV12): float registers -> SurfObjPlanes
            template<int NumPlanes>
            __device__ void execute(processing_type* input, const SurfObjPlanes<NumPlanes>& output, int width,
                                    int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::exgest,
                              "SurfObjPlanes output is only valid for exgest operations");
                return impl::execute(input, output, width, height);
            }

            // ========================================================================
            // Shared memory-based execution methods
            // ========================================================================

            // Ingest with shared memory: global memory -> Channels
            template<typename InputType, typename ChannelSliceType, unsigned int NumChannels>
            NPPDX_DECL_D void execute(const InputType*                                           input,
                                      shared_memory::TileStorage<ChannelSliceType, NumChannels>& output_channels,
                                      int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::ingest,
                              "This execute signature is for ingest operations only");

                return impl::execute(input, output_channels, width, height);
            }

            // Ingest with shared memory: multi-plane global memory -> Channels
            template<typename InputType, int PlaneCount, typename ChannelSliceType, unsigned int NumChannels>
            __device__ void execute(const ImageLayout<InputType, PlaneCount>&                  input,
                                    shared_memory::TileStorage<ChannelSliceType, NumChannels>& output_channels,
                                    int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::ingest,
                              "This execute signature is for ingest operations only");

                return impl::execute(input, output_channels, width, height);
            }

            // Exgest with shared memory: Channels -> global memory
            template<typename OutputType, typename ChannelSliceType, unsigned int NumChannels>
            NPPDX_DECL_D void execute(const shared_memory::TileStorage<ChannelSliceType, NumChannels>& input_channels,
                                      OutputType* output, int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::exgest,
                              "This execute signature is for exgest operations only");

                return impl::execute(input_channels, output, width, height);
            }

            // Exgest with shared memory: Channels -> multi-plane global memory
            template<typename OutputType, int PlaneCount, typename ChannelSliceType, unsigned int NumChannels>
            __device__ void execute(const shared_memory::TileStorage<ChannelSliceType, NumChannels>& input_channels,
                                    const ImageLayout<OutputType, PlaneCount>& output, int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::exgest,
                              "This execute signature is for exgest operations only");

                return impl::execute(input_channels, output, width, height);
            }

            // Ingest with shared memory: texture -> Channels
            template<typename ChannelSliceType, unsigned int NumChannels>
            __device__ void execute(TexObjT                                                    input,
                                    shared_memory::TileStorage<ChannelSliceType, NumChannels>& output_channels,
                                    int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::ingest,
                              "TexObjT input is only valid for ingest operations");
                return impl::execute(input, output_channels, width, height);
            }

            // Ingest with shared memory: multi-plane texture -> Channels
            template<int NumPlanes, typename ChannelSliceType, unsigned int NumChannels>
            __device__ void execute(const TexObjPlanes<NumPlanes>&                             input,
                                    shared_memory::TileStorage<ChannelSliceType, NumChannels>& output_channels,
                                    int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::ingest,
                              "TexObjPlanes input is only valid for ingest operations");
                return impl::execute(input, output_channels, width, height);
            }

            // Exgest with shared memory: Channels -> surface
            template<typename ChannelSliceType, unsigned int NumChannels>
            __device__ void execute(const shared_memory::TileStorage<ChannelSliceType, NumChannels>& input_channels,
                                    SurfObjT output, int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::exgest,
                              "SurfObjT output is only valid for exgest operations");
                return impl::execute(input_channels, output, width, height);
            }

            // Exgest with shared memory: Channels -> multi-plane surface
            template<int NumPlanes, typename ChannelSliceType, unsigned int NumChannels>
            __device__ void execute(const shared_memory::TileStorage<ChannelSliceType, NumChannels>& input_channels,
                                    const SurfObjPlanes<NumPlanes>& output, int width, int height) const {
                static_assert(has_input_output_operator,
                              "This execute method is only available for ingest/exgest operations");
                static_assert(input_output_direction_of_v<base_type> == input_output_direction::exgest,
                              "SurfObjPlanes output is only valid for exgest operations");
                return impl::execute(input_channels, output, width, height);
            }

            // Function operation with shared memory (in-place)
            template<typename ChannelSliceType, unsigned int NumChannels>
            NPPDX_DECL_D void execute(shared_memory::TileStorage<ChannelSliceType, NumChannels>& channels, int width,
                                      int height) const {
                static_assert(has_function_operator, "This execute method is only available for function operations");

                return impl::execute(channels, width, height);
            }

            // Function operation with shared memory
            template<typename ChannelsSliceInputType, typename ChannelsSliceOutputType, unsigned int InputNumChannels,
                     unsigned int OutputNumChannels>
            NPPDX_DECL_D void execute(
                const shared_memory::TileStorage<ChannelsSliceInputType, InputNumChannels>& input_channels,
                shared_memory::TileStorage<ChannelsSliceOutputType, OutputNumChannels>& output_channels, int width,
                int height) const {
                static_assert(has_function_operator, "This execute method is only available for function operations");

                return impl::execute(input_channels, output_channels, width, height);
            }

            // Function operation with shared memory and intermediate buffer
            template<typename ChannelsSliceInputType, typename ChannelsSliceIntermediateType,
                     typename ChannelsSliceOutputType, unsigned int InputNumChannels,
                     unsigned int IntermediateNumChannels, unsigned int OutputNumChannels>
            NPPDX_DECL_D void execute(
                const shared_memory::TileStorage<ChannelsSliceInputType, InputNumChannels>& input_channels,
                shared_memory::TileStorage<ChannelsSliceIntermediateType, IntermediateNumChannels>&
                                                                                        intermediate_channels,
                shared_memory::TileStorage<ChannelsSliceOutputType, OutputNumChannels>& output_channels, int width,
                int height) const {
                static_assert(has_function_operator, "This execute method is only available for function operations");

                return impl::execute(input_channels, intermediate_channels, output_channels, width, height);
            }
        };

    } // namespace detail

} // namespace nppdx

#endif // NPPDX_DETAIL_NPPDX_EXECUTION_HPP
