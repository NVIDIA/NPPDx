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

#ifndef NPPDX_DETAIL_DATABASE_INPUT_OUTPUT_OPERATION_IMPLEMENTATION_HPP
#define NPPDX_DETAIL_DATABASE_INPUT_OUTPUT_OPERATION_IMPLEMENTATION_HPP

#include "nppdx/detail/config.hpp"
#include "nppdx/detail/backend/texture/texture_operations.hpp"
#include "nppdx/detail/database/region.hpp"
#include "nppdx/detail/database/cell_executor.hpp"
#include "nppdx/detail/database/input_output/npp_adapter.hpp"
#include "nppdx/detail/database/operation_common.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/types.hpp"

#include NPPDX_STD_INCLUDE_TYPE_TRAITS
#include NPPDX_STD_INCLUDE_NUMERIC
#include NPPDX_STD_INCLUDE_TUPLE
#include NPPDX_STD_INCLUDE_UTILITY

namespace nppdx {
    namespace detail {

        template<bool Ingest, packing_format Format, typename T>
        NPPDX_DECL_D auto get_minimum_image_layout(T* gmem, uint2 img_size) {
            using GMemT    = NPPDX_STD::conditional_t<Ingest, NPPDX_STD::add_const_t<T>, T>;
            using GMemPtrT = GMemT*;

            return packing_format_helper<Format>::minimum_image_layout(GMemPtrT(gmem), size2(img_size));
        }

        template<bool Ingest, packing_format Format, typename CellCfg, typename T, int N, typename U>
        NPPDX_DECL_D auto process_gmem_cell(const ImageLayout<T, N>& gmem_layout, U* regs, uint2 img_size,
                                            int2 cell_pos) {
            static_assert(packing_format_helper<Format>::props.planes == N, "Number of planes must match the format");
            using GMemT = NPPDX_STD::conditional_t<Ingest, NPPDX_STD::add_const_t<T>, T>;
            using RegsT = NPPDX_STD::conditional_t<Ingest, U, NPPDX_STD::add_const_t<U>>;
            if constexpr (Ingest) {
                static_assert(NPPDX_STD::is_same_v<GMemT, const uint8_t>,
                              "Global memory pointer must be (possibly const) uint8_t for ingest");
            } else {
                static_assert(NPPDX_STD::is_same_v<GMemT, uint8_t>, "Global memory pointer must be uint8_t for exgest");
            }

            using npp_cell_data_type = NPPCellData<CellCfg, RegsT>;
            using npp_funcs          = nppdx_safe_functions<Format>;
            npp_cell_data_type cell_data(regs);

            constexpr uint2 cell_size = CellCfg::value.size;
            int2            cell_idx  = cell_pos / int2(cell_size);
            auto            strides   = gmem_layout.byte_strides.template cast<size_t>();
            auto            ptrs      = gmem_layout.ptrs;

            if constexpr (Ingest) {
                npp_funcs::safe_load_cell(ptrs.value, strides.value, cell_data, cell_idx, img_size);
            } else {
                npp_funcs::safe_save_cell(ptrs.value, strides.value, cell_data, cell_idx, img_size);
            }
            return cell_data;
        }

        template<typename GridT>
        struct grid_helper {
            GridT grid;
            int2  padding_offset;

            int2 image_offset;

            NPPDX_DECL_NCFHD OuterPos smem_pos(InnerCellIdx cell_idx) const {
                return get_smem_pos(grid, cell_idx, padding_offset);
            }

            NPPDX_DECL_NCFHD int2 global_pos(InnerCellIdx cell_idx) const {
                return get_global_pos(grid, cell_idx, image_offset);
            }
        };

        template<typename Operation, typename Layout, typename CellCfg>
        struct IngestExgestOpImplHelper: public OpImplHelper<Operation, Layout, CellCfg> {
        public:
            NPPDX_DECL_SC layout_props layout         = Layout::value;
            NPPDX_DECL_SC uint2        cell_size      = CellCfg::CellSize::value;
            NPPDX_DECL_SC int2         padding_offset = layout.padding_halo().left_top;

            NPPDX_DECL_NSC bool is_ingest() { return Operation::tag_type::value == input_output_direction::ingest; }

            // Workaround for incorrectly deduced void type.
            using grid_helper_t =
                NPPDX_STD::conditional_t<is_ingest(), grid_helper<generic_cell_grid>, grid_helper<aligned_cell_grid>>;

            NPPDX_DECL_NSFD grid_helper_t get_grid_helper() {
                const int2     tile_idx        = thread_utils::get_tile_idx();
                constexpr auto layout_         = copy(layout);
                constexpr auto cell_size_      = copy(cell_size);
                constexpr auto padding_offset_ = copy(padding_offset);

                if constexpr (is_ingest()) {
                    const region            nominal_tile {tile_idx * int2(layout_.nominal_size), layout_.nominal_size};
                    const region            extended_tile = nominal_tile + layout_.cumulative_halo;
                    const generic_cell_grid grid {extended_tile, cell_size_};
                    return grid_helper<generic_cell_grid> {grid, padding_offset_, int2::zero()};
                } else {
                    constexpr aligned_cell_grid grid {layout_.nominal_size, cell_size_};
                    const int2                  image_offset = -tile_idx * int2(layout_.nominal_size);
                    return grid_helper<aligned_cell_grid> {grid, padding_offset_, image_offset};
                }
            }

            NPPDX_DECL_NSC auto get_first_image_offset() {
                return int2::zero();
            }

            // Only with ingest.
            NPPDX_DECL_NSC uint2 extra_max_cell_count() {
                return vec2_apply(
                    [](uint nominal_extent, uint extended_extent, uint cell_extent, int first_buf_pos) {
                        const uint g     = NPPDX_STD::gcd(nominal_extent, cell_extent);
                        const uint delta = dir_rem<-1>(extended_extent, cell_extent);
                        if ((cell_extent - delta) > g) {
                            return 1u;
                        }

                        int xi = dir_rem<-1>(first_buf_pos, int(g));
                        return (xi != 0 && xi < int(cell_extent - delta)) ? 1u : 0u;
                    },
                    layout.nominal_size, layout.extended_size(), cell_size,
                    get_first_image_offset() - layout.cumulative_halo.left_top);
            }
            //Move to public to avoid clang compilation errors. Alias was accessing a private member,
            //clang checks accessibility at caller side which means that classes outside this class cannot access the member.
            template<typename T>
            NPPDX_DECL_NSC auto get_executor_type() {
                OpImplHelperBase::check_data_type<T>();
                if constexpr (is_ingest()) {
                    constexpr uint2 max_cell_count = layout.memory_size().ceil_div(cell_size) + extra_max_cell_count();
                    return cell_executor<NPPDX_MAKE_UINT2D(max_cell_count)> {};
                } else {
                    static_assert((layout.nominal_size % cell_size).is_zero(),
                                  "Tile size must be a multiple of the cell size for exgest");
                    constexpr uint2 cell_count = layout.nominal_size.ceil_div(cell_size);
                    return cell_executor<NPPDX_MAKE_UINT2D(cell_count)> {};
                }
            }

        public:
            template<typename T>
            using executor_type = decltype(get_executor_type<T>());
        };

        template<packing_format Format, typename Layout, typename CellCfg, int SM, int BlockThreads>
        class generic_ingest
        {
        public:
            using traits = IngestExgestOpImplHelper<InputFormat<Format>, Layout, CellCfg>;

            // Register-based (global memory -> registers)
            template<typename T>
            NPPDX_DECL_SD void execute(const uint8_t* input, T* output, int width, int height) {
                // Calling the register-based ingest function implies that the
                // initial cumulative halo is zero and that there are no area
                // operations following it. Shared memory will not be accessed
                // for data exchange between threads, but this also means that
                // the ingest cell grid must exactly coincide with the exgest
                // cell grid. This can only happen when every tile is left-top
                // aligned with a cell, which also implies that the tile size is
                // an integer multiple of the cell size in all dimensions.

                const auto img_size = uint2 {uint(width), uint(height)};
                const auto layout   = get_minimum_image_layout</*Ingest=*/true, Format>(input, img_size);
                execute(layout, output, width, height);
            }

            template<typename T, int N>
            NPPDX_DECL_SD void execute(const ImageLayout<const uint8_t, N>& input, T* output, int width, int height) {
                static_assert(traits::layout.cumulative_halo.empty(),
                              "Initial cumulative halo must be zero for register-based ingest");
                static_assert((traits::get_first_image_offset() % int2(traits::cell_size)).is_zero(),
                              "Image offset must be a multiple of the cell size for register-based ingest");
                static_assert((traits::layout.nominal_size % traits::cell_size).is_zero(),
                              "Tile size must be a multiple of the cell size for register-based ingest");
                static_assert(traits::extra_max_cell_count().is_zero(),
                              "Extra max cell count must be zero for register-based ingest");

                using executor = typename traits::template executor_type<T>;
                executor::template for_each_cell_idx</*Strided=*/false>([&](InnerCellIdx cell_idx) {
                    const auto grid_helper = traits::get_grid_helper();
                    (void)process_gmem_cell</*Ingest=*/true, Format, CellCfg>(
                        input, output, uint2 {uint(width), uint(height)}, grid_helper.global_pos(cell_idx));
                });
            }

            // Shared memory variant (global memory -> shared memory tile)
            NPPDX_DECL_SD void execute(const uint8_t* input, typename traits::TileStorageType& output_tile, int width,
                                       int height) {
                const auto img_size = uint2 {uint(width), uint(height)};
                const auto layout   = get_minimum_image_layout</*Ingest=*/true, Format>(input, img_size);
                execute(layout, output_tile, width, height);
            }

            // Shared memory variant (multi-plane global memory -> shared memory tile)
            template<int N>
            NPPDX_DECL_SD void execute(const ImageLayout<const uint8_t, N>& input,
                                       typename traits::TileStorageType& output_tile, int width, int height) {
                using internal_processing_type = typename traits::processing_type;
                using executor                 = typename traits::template executor_type<internal_processing_type>;

                executor::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    internal_processing_type cell_registers[traits::cell_cfg.elements_per_cell()];
                    const auto               grid_helper = traits::get_grid_helper();
                    auto                     cell_data   = process_gmem_cell</*Ingest=*/true, Format, CellCfg>(
                        input, cell_registers, uint2 {uint(width), uint(height)}, grid_helper.global_pos(cell_idx));
                    uint2 clip_size = Layout::value.memory_size();
                    store_cell_to_tile_clipped(cell_data, output_tile, grid_helper.smem_pos(cell_idx), clip_size);
                });

                __syncthreads();
            }

            // Texture ingest: N cuArray handles -> registers. texture_handle_backend maps the handle
            // count to the topology (1 = packed/stacked single array, N = one array per plane), so a
            // single code path serves both.
            template<int NumHandles, typename T>
            NPPDX_DECL_SD void execute(const TexObjPlanes<NumHandles>& input, T* output, int width, int height) {
                using Backend = backend::texture_handle_backend<Format, NumHandles>;
                static_assert(Backend::is_implemented,
                              "Texture ingest not implemented for this format/topology");
                static_assert(NumHandles == packing_format_helper<Format>::props.planes || NumHandles == 1,
                              "Handle count must equal the plane count (per-plane) or 1 (stacked/packed)");
                using executor = typename traits::template executor_type<T>;
                TexObjT tex[NumHandles];
#pragma unroll
                for (int p = 0; p < NumHandles; ++p) {
                    tex[p] = input[p];
                }
                executor::template for_each_cell_idx</*Strided=*/false>([&](InnerCellIdx cell_idx) {
                    constexpr uint2         cell_size   = CellCfg::value.size;
                    const auto              grid_helper = traits::get_grid_helper();
                    const int2              cell_pos    = grid_helper.global_pos(cell_idx) / int2(cell_size);
                    NPPCellData<CellCfg, T> cell_data(output);
                    Backend::tex_load(tex, cell_data, cell_pos.x, cell_pos.y, uint(width), uint(height));
                });
            }

            // Shared memory variant: N cuArray handles -> shared memory tile.
            template<int NumHandles>
            NPPDX_DECL_SD void execute(const TexObjPlanes<NumHandles>& input,
                                       typename traits::TileStorageType& output_tile, int width, int height) {
                using Backend = backend::texture_handle_backend<Format, NumHandles>;
                static_assert(Backend::is_implemented,
                              "Texture ingest not implemented for this format/topology");
                static_assert(NumHandles == packing_format_helper<Format>::props.planes || NumHandles == 1,
                              "Handle count must equal the plane count (per-plane) or 1 (stacked/packed)");
                using PT       = typename traits::processing_type;
                using executor = typename traits::template executor_type<PT>;
                TexObjT tex[NumHandles];
#pragma unroll
                for (int p = 0; p < NumHandles; ++p) {
                    tex[p] = input[p];
                }
                executor::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    PT                       cell_registers[traits::cell_cfg.elements_per_cell()];
                    NPPCellData<CellCfg, PT> cell_data(cell_registers);
                    constexpr uint2          cell_size   = CellCfg::value.size;
                    const auto               grid_helper = traits::get_grid_helper();
                    const int2               cell_pos    = grid_helper.global_pos(cell_idx) / int2(cell_size);
                    Backend::tex_load(tex, cell_data, cell_pos.x, cell_pos.y, uint(width), uint(height));
                    const uint2 clip_size = Layout::value.memory_size();
                    store_cell_to_tile_clipped(cell_data, output_tile, grid_helper.smem_pos(cell_idx), clip_size);
                });
                __syncthreads();
            }

            // Convenience: a bare single handle is TexObjPlanes<1> (packed single-plane, or a
            // multi-plane format stacked in one cuArray).
            template<typename T>
            NPPDX_DECL_SD void execute(TexObjT input, T* output, int width, int height) {
                execute(TexObjPlanes<1> {input}, output, width, height);
            }
            NPPDX_DECL_SD void execute(TexObjT input, typename traits::TileStorageType& output_tile, int width,
                                       int height) {
                execute(TexObjPlanes<1> {input}, output_tile, width, height);
            }
        };

        template<packing_format Format, typename Layout, typename CellCfg, int SM, int BlockThreads>
        class generic_exgest
        {
        public:
            using traits = IngestExgestOpImplHelper<OutputFormat<Format>, Layout, CellCfg>;

            // Register-based (registers -> global memory)
            template<typename T>
            NPPDX_DECL_SD void execute(T* input, uint8_t* output, int width, int height) {
                const auto img_size = uint2 {uint(width), uint(height)};
                const auto layout   = get_minimum_image_layout</*Ingest=*/false, Format>(output, img_size);
                execute(input, layout, width, height);
            }

            // Register-based (registers -> multi-plane global memory)
            template<typename T, int N>
            NPPDX_DECL_SD void execute(T* input, const ImageLayout<uint8_t, N>& output, int width, int height) {
                using executor = typename traits::template executor_type<T>;
                executor::template for_each_cell_idx</*Strided=*/false>([&](InnerCellIdx cell_idx) {
                    const auto grid_helper = traits::get_grid_helper();
                    (void)process_gmem_cell</*Ingest=*/false, Format, CellCfg>(
                        output, input, uint2 {uint(width), uint(height)}, grid_helper.global_pos(cell_idx));
                });
            }

            // Shared memory variant (shared memory tile -> global memory)
            NPPDX_DECL_SD void execute(const typename traits::TileStorageType& input_tile, uint8_t* output, int width,
                                       int height) {
                const auto img_size = uint2 {uint(width), uint(height)};
                const auto layout   = get_minimum_image_layout</*Ingest=*/false, Format>(output, img_size);
                execute(input_tile, layout, width, height);
            }

            // Shared memory variant (shared memory tile -> multi-plane global memory)
            template<int N>
            NPPDX_DECL_SD void execute(const typename traits::TileStorageType& input_tile,
                                       const ImageLayout<uint8_t, N>& output, int width, int height) {
                using internal_processing_type = typename traits::processing_type;
                using executor                 = typename traits::template executor_type<internal_processing_type>;

                executor::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    internal_processing_type cell_registers_data[traits::cell_cfg.elements_per_cell()];
                    NPPCellData<CellCfg, internal_processing_type> cell_registers(cell_registers_data);
                    const auto                                     grid_helper = traits::get_grid_helper();
                    load_cell_from_tile(input_tile, cell_registers, cell_idx, int2(grid_helper.padding_offset));
                    (void)process_gmem_cell</*Ingest=*/false, Format, CellCfg>(output, cell_registers_data,
                                                                               uint2 {uint(width), uint(height)},
                                                                               grid_helper.global_pos(cell_idx));
                });
            }

            // Surface exgest: registers -> N cuArray handles. texture_handle_backend maps the handle
            // count to the topology (1 = packed/stacked single array, N = one array per plane), so a
            // single code path serves both.
            template<typename T, int NumHandles>
            NPPDX_DECL_SD void execute(T* input, const SurfObjPlanes<NumHandles>& output, int width, int height) {
                using Backend = backend::texture_handle_backend<Format, NumHandles>;
                static_assert(Backend::is_implemented,
                              "Surface exgest not implemented for this format/topology");
                static_assert(NumHandles == packing_format_helper<Format>::props.planes || NumHandles == 1,
                              "Handle count must equal the plane count (per-plane) or 1 (stacked/packed)");
                using executor = typename traits::template executor_type<T>;
                SurfObjT surf[NumHandles];
#pragma unroll
                for (int p = 0; p < NumHandles; ++p) {
                    surf[p] = output[p];
                }
                executor::template for_each_cell_idx</*Strided=*/false>([&](InnerCellIdx cell_idx) {
                    constexpr uint2         cell_size   = CellCfg::value.size;
                    const auto              grid_helper = traits::get_grid_helper();
                    const int2              cell_pos    = grid_helper.global_pos(cell_idx) / int2(cell_size);
                    NPPCellData<CellCfg, T> cell_data(input);
                    Backend::surf_save(surf, cell_data, cell_pos.x, cell_pos.y, uint(width), uint(height));
                });
            }

            // Shared memory variant: shared memory tile -> N cuArray handles.
            template<int NumHandles>
            NPPDX_DECL_SD void execute(const typename traits::TileStorageType& input_tile,
                                       const SurfObjPlanes<NumHandles>& output, int width, int height) {
                using Backend = backend::texture_handle_backend<Format, NumHandles>;
                static_assert(Backend::is_implemented,
                              "Surface exgest not implemented for this format/topology");
                static_assert(NumHandles == packing_format_helper<Format>::props.planes || NumHandles == 1,
                              "Handle count must equal the plane count (per-plane) or 1 (stacked/packed)");
                using PT       = typename traits::processing_type;
                using executor = typename traits::template executor_type<PT>;
                SurfObjT surf[NumHandles];
#pragma unroll
                for (int p = 0; p < NumHandles; ++p) {
                    surf[p] = output[p];
                }
                executor::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    PT                       cell_registers_data[traits::cell_cfg.elements_per_cell()];
                    NPPCellData<CellCfg, PT> cell_registers(cell_registers_data);
                    const auto               grid_helper = traits::get_grid_helper();
                    load_cell_from_tile(input_tile, cell_registers, cell_idx, int2(grid_helper.padding_offset));
                    constexpr uint2 cell_size = CellCfg::value.size;
                    const int2      cell_pos  = grid_helper.global_pos(cell_idx) / int2(cell_size);
                    Backend::surf_save(surf, cell_registers, cell_pos.x, cell_pos.y, uint(width), uint(height));
                });
            }

            // Convenience: a bare single handle is SurfObjPlanes<1> (packed single-plane, or a
            // multi-plane format stacked in one cuArray).
            template<typename T>
            NPPDX_DECL_SD void execute(T* input, SurfObjT output, int width, int height) {
                execute(input, SurfObjPlanes<1> {output}, width, height);
            }
            NPPDX_DECL_SD void execute(const typename traits::TileStorageType& input_tile, SurfObjT output, int width,
                                       int height) {
                execute(input_tile, SurfObjPlanes<1> {output}, width, height);
            }
        };

        //==========================================================================
        // Macro to instantiate operation implementations for all formats
        //==========================================================================

#define NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(FORMAT)                                \
    template<typename Layout, typename CellCfg, int SM, int BlockThreads>                   \
    class op_impl<InputFormat<packing_format::FORMAT>, Layout, CellCfg, SM, BlockThreads>:  \
        public generic_ingest<packing_format::FORMAT, Layout, CellCfg, SM, BlockThreads>    \
    {                                                                                       \
    };                                                                                      \
    template<typename Layout, typename CellCfg, int SM, int BlockThreads>                   \
    class op_impl<OutputFormat<packing_format::FORMAT>, Layout, CellCfg, SM, BlockThreads>: \
        public generic_exgest<packing_format::FORMAT, Layout, CellCfg, SM, BlockThreads>    \
    {                                                                                       \
    };

        //==========================================================================
        // Format specific operations
        //==========================================================================

        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(yuv420p);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(yuv420p10);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(yuv422p);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(yuv422p10);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(nv12);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(p010);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(nv16);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(p216);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(rgb24);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(rgb10);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(rgb16);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(y210);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(uyvp);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(v210);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(yuv2);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(rgbp);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(bgrp);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(yuv444p);
        NPPDX_INSTANTIATE_GENERIC_INGEST_EXGEST_IMPL(yuv444p10);
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_DATABASE_INPUT_OUTPUT_OPERATION_IMPLEMENTATION_HPP
