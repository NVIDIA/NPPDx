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

#ifndef NPPDX_DETAIL_DATABASE_CELL_EXECUTOR_HPP
#define NPPDX_DETAIL_DATABASE_CELL_EXECUTOR_HPP

#include "nppdx/detail/database/region.hpp"
#include "nppdx/detail/database/cell_data.hpp"
#include "nppdx/detail/database/thread_utils.hpp"
#include "nppdx/detail/decl.hpp"

namespace nppdx {
    namespace detail {

        NPPDX_DECLARE_TAGGED_TYPE(OuterCellIdx, int2);
        NPPDX_DECLARE_TAGGED_TYPE(InnerCellIdx, int2);

        // The left-top corner of the first cell is aligned with the left-top corner of the considered region. Special
        // case of the more general generic_cell_grid. Inner and outer positions and cell indices are the same.
        struct aligned_cell_grid {
            uint2 region_size;
            uint2 cell_size;

            NPPDX_DECL_NCHD InnerPos inner_cell_pos(InnerCellIdx cell_idx) const {
                return InnerPos(cell_idx * int2(cell_size));
            }

            NPPDX_DECL_NCHD OuterPos outer_cell_pos(InnerCellIdx cell_idx) const {
                return OuterPos(inner_cell_pos(cell_idx));
            }

            NPPDX_DECL_NCHD uint2 cell_count() const { return region_size.ceil_div(cell_size); }
        };

        // A generic grid is defined w.r.t. a larger outer cell grid. covered_region.offset is the offset of the
        // considered region relative to the left-top-pixel position of the first outer cell. A cell's outer index is
        // its index within the outer cell grid. There will be a left-top-most cell that overlaps with covered_region.
        // covered_region's left-top-pixel does not necessarily coincide with this cell's left-top-pixel. A cell's
        // inner index is its outer index minus the outer index of this first overlapping cell.
        struct generic_cell_grid {
            region covered_region;
            uint2  cell_size;

            NPPDX_DECL_NCHD OuterCellIdx first_outer_cell_idx() const {
                return OuterCellIdx(covered_region.outer_left_top_pos()->dir_quot<+1>(int2(cell_size)));
            }

            NPPDX_DECL_NCHD OuterPos outer_cell_pos(InnerCellIdx cell_idx) const {
                return OuterPos((first_outer_cell_idx() + cell_idx) * int2(cell_size));
            }

            NPPDX_DECL_NCHD InnerPos inner_cell_pos(InnerCellIdx cell_idx) const {
                return InnerPos(outer_cell_pos(cell_idx) - covered_region.offset);
            }

            NPPDX_DECL_NCHD uint2 cell_count() const {
                int2 start_offset = covered_region.outer_left_top_pos() - outer_cell_pos(InnerCellIdx(int2::zero()));
                return uint2(start_offset + int2(covered_region.size)).ceil_div(cell_size);
            }
        };


        template<typename GridT>
        NPPDX_DECL_NCFHD OuterPos get_smem_pos(GridT grid, InnerCellIdx cell_idx, int2 padding_offset) {
            return OuterPos(grid.inner_cell_pos(cell_idx) + padding_offset);
        }

        template<typename GridT>
        NPPDX_DECL_NCFHD int2 get_global_pos(GridT grid, InnerCellIdx cell_idx, int2 image_offset) {
            return grid.outer_cell_pos(cell_idx) - image_offset;
        }

        template<typename GridSizeT>
        struct cell_executor {
            static_assert(IsUInt2D<GridSizeT>);
            NPPDX_DECL_SC uint2 grid_size = GridSizeT::value;

            //==========================================================================
            // Lowest-level
            //==========================================================================

        private:
            template<bool MakePosIdx>
            NPPDX_DECL_SCFHD auto make_idx(uint lin_idx) {
                if constexpr (MakePosIdx) {
                    return InnerCellIdx(int2(uint2 {lin_idx % grid_size.x, lin_idx / grid_size.x}));
                } else {
                    return lin_idx;
                }
            }

            template<bool PassPosIdx, bool Strided, typename Fn>
            NPPDX_DECL_SFD void for_each_execute(Fn&& fn) {
                constexpr uint total_cells = grid_size.x * grid_size.y;
                const uint     tid         = thread_utils::get_local_id();
                if constexpr (Strided) {
                    const uint stride = thread_utils::get_thread_count();
                    for (uint idx = tid; idx < total_cells; idx += stride) {
                        NPPDX_STD::forward<Fn>(fn)(make_idx<PassPosIdx>(idx));
                    }
                } else {
                    if (tid < total_cells) {
                        NPPDX_STD::forward<Fn>(fn)(make_idx<PassPosIdx>(tid));
                    }
                }
            }

        public:
            template<bool Strided, typename Fn>
            NPPDX_DECL_SFD void for_each_linear_idx(Fn&& fn) {
                for_each_execute</*PassPosIdx=*/false, Strided>(NPPDX_STD::forward<Fn>(fn));
            }

            //==========================================================================
            // Cell iteration (no allocation)
            //==========================================================================

            template<bool Strided, typename Fn>
            NPPDX_DECL_SFD void for_each_cell_idx(Fn fn) {
                for_each_execute</*PassPosIdx=*/true, Strided>(NPPDX_STD::forward<Fn>(fn));
            }
        };


        // An executor that assumes that the first cell is left-top aligned with the extended tile.
        template<typename GridSizeT, typename PaddingOffsetT, typename CellCfgT>
        struct aligned_cell_executor: public cell_executor<GridSizeT> {
            static_assert(IsCellConfig<CellCfgT>);
            static_assert(IsInt2D<PaddingOffsetT>);

            using base_type                          = cell_executor<GridSizeT>;
            NPPDX_DECL_SC cell_config cell_cfg       = CellCfgT::value;
            NPPDX_DECL_SC int2        padding_offset = PaddingOffsetT::value;

            NPPDX_DECL_SC auto grid = aligned_cell_grid {base_type::grid_size, cell_cfg.size};

            //==========================================================================
            // High-level convenience: tile operations
            //==========================================================================

            template<bool Strided, typename TileType, typename ProcessFn>
            NPPDX_DECL_SFD void execute_inplace(TileType& tile, uint2 clip_size, ProcessFn&& process) {
                using T = typename TileType::ChannelSliceType::value_type;
                T                        regs[cell_cfg.elements_per_cell()];
                NPPCellData<CellCfgT, T> cell(regs);
                base_type::template for_each_cell_idx<Strided>([&](InnerCellIdx cell_idx) {
                    const OuterPos smem_pos = get_smem_pos(copy(grid), cell_idx, padding_offset);
                    load_cell_from_tile_clipped(tile, cell, smem_pos, clip_size);
                    process(cell);
                    store_cell_to_tile_clipped(cell, tile, smem_pos, clip_size);
                });
            }
        };

    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_DATABASE_CELL_EXECUTOR_HPP
