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

#ifndef NPPDX_DETAIL_DATABASE_CELL_CONFIG_HPP
#define NPPDX_DETAIL_DATABASE_CELL_CONFIG_HPP

#include "nppdx/detail/config.hpp"
#include "nppdx/detail/nppdx_checks.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/operators/formats.hpp"
#include "nppdx/types.hpp"

#include NPPDX_STD_INCLUDE_CSTDINT

namespace nppdx::detail {

    // File proper

    struct cell_config {
        uint2        size;
        unsigned int channel_count;

        NPPDX_DECL_NCHD unsigned int cell_pixels() const { return size.x * size.y; }
        NPPDX_DECL_NCHD unsigned int elements_per_cell() const { return cell_pixels() * channel_count; }
        NPPDX_DECL_NCHD dim3         cell_dim() const { return dim3 {size.x, size.y, 1}; }
    };

    template<typename CellSizeT, uint ChannelsPerPixelV>
    struct CellConfig {
        static_assert(IsUInt2D<CellSizeT>);

        NPPDX_DECL_SC cell_config value {CellSizeT::value, ChannelsPerPixelV};

        using CellSize                      = CellSizeT;
        NPPDX_DECL_SC uint ChannelsPerPixel = ChannelsPerPixelV;
    };

    template<typename T>
    NPPDX_DECL_CI bool IsCellConfig = false;
    template<typename CellSizeT, uint ChannelsPerPixelV>
    NPPDX_DECL_CI bool IsCellConfig<CellConfig<CellSizeT, ChannelsPerPixelV>> = true;

#define NPPDX_MAKE_CELL_CONFIG(cell_cfg) \
    ::nppdx::detail::CellConfig<NPPDX_MAKE_UINT2D(cell_cfg.size), cell_cfg.channel_count>

    NPPDX_DECL_NCHD cell_config get_cell_config(int /*sm*/, packing_format /*format*/) {
        // For the time being, all configurations use the same {{4, 2}, 3} cell config, but this may evolve.
        // Experiment: changed from {4,2} to {2,2} to double block size (128→256) and improve occupancy on SM 12.0.
        return {{4, 2}, 3};
    }


    NPPDX_DECL_NCHD bool is_valid_tile_size(int sm, uint2 tile_size, cell_config cell_cfg, int size_of_pixel,
                                            uint shared_memory_limit) {
        const bool is_sm_valid = is_valid_sm(sm);
        const bool is_tile_size_valid =
            tile_size.x * tile_size.y * cell_cfg.channel_count * size_of_pixel <= shared_memory_limit &&
            tile_size.x > 0 && tile_size.y > 0;
        const bool is_tile_dimensions_valid = tile_size.x % cell_cfg.size.x == 0 && tile_size.y % cell_cfg.size.y == 0;
        return is_sm_valid && is_tile_size_valid && is_tile_dimensions_valid;
    }

    NPPDX_DECL_NCHD bool is_valid_tile_v(uint2 tile_size, int SM, packing_format PackingFormat) {
        return is_valid_tile_size(SM, tile_size, get_cell_config(SM, PackingFormat),
                                  sizeof(packing_format_traits<packing_format::internal>::pixel_type),
                                  sm_shared_memory_size(SM));
    }


    NPPDX_DECL_NCHD unsigned int get_number_of_cells(uint2 tile_size, uint2 cell_size) {
        if (cell_size.x == 0 || cell_size.y == 0) {
            return 0;
        }

        const uint2 cell_count = tile_size.ceil_div(cell_size);
        return cell_count.x * cell_count.y;
    }

} // namespace nppdx::detail

#endif // NPPDX_DETAIL_DATABASE_CELL_CONFIG_HPP
