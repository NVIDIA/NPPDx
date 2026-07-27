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

#ifndef NPPDX_DETAIL_DATABASE_CELL_DATA_HPP
#define NPPDX_DETAIL_DATABASE_CELL_DATA_HPP

#include "nppdx/detail/config.hpp"
#include "nppdx/detail/database/region.hpp"
#include "nppdx/shared_memory.hpp"
#include "nppdx/detail/database/cell_config.hpp"
#include "nppdx/detail/decl.hpp"

#include NPPDX_STD_INCLUDE_CSTDINT

namespace nppdx {
    namespace detail {

        // MEMORY LAYOUT:
        // The 'data' pointer points to memory organized as follows:
        // - First pixels_per_channel elements of type T contain Channel 0 data
        // - Next pixels_per_channel elements of type T contain Channel 1 data
        // - Next pixels_per_channel elements of type T contain Channel 2 data
        // - And so on for all NumChannels
        //
        // Example for a 2x2 cell with 3 channels:
        // data[0..3]   = Channel 0 pixels (e.g., Red)
        // data[4..7]   = Channel 1 pixels (e.g., Green)
        // data[8..11]  = Channel 2 pixels (e.g., Blue)
        template<typename CellCfg, typename T>
        struct NPPCellData {
            T* data;

            using value_type  = T;
            using config_type = CellCfg;

            NPPDX_DECL_SC cell_config cell_cfg = CellCfg::value;

            static constexpr unsigned int cell_width  = CellCfg::value.size.x;
            static constexpr unsigned int cell_height = CellCfg::value.size.y;

            NPPDX_DECL_SCFHD uint channel_offset(int channel) { return channel * cell_cfg.cell_pixels(); }

            NPPDX_DECL_FHD NPPCellData(T* base_ptr): data(base_ptr) {}

            NPPDX_DECL_FHD T& operator()(int channel, int pixel) { return data[channel_offset(channel) + pixel]; }

            NPPDX_DECL_FHD const T& operator()(int channel, int pixel) const {
                return data[channel_offset(channel) + pixel];
            }

            NPPDX_DECL_FHD T& operator()(int2 pos, int channel) {
                return data[channel_offset(channel) + pos.y * cell_cfg.size.x + pos.x];
            }

            NPPDX_DECL_FHD const T& operator()(int2 pos, int channel) const {
                return data[channel_offset(channel) + pos.y * cell_cfg.size.x + pos.x];
            }

            NPPDX_DECL_FHD T* channel_ptr(int channel) { return data + channel_offset(channel); }

            NPPDX_DECL_FHD const T* channel_ptr(int channel) const { return data + channel_offset(channel); }
        };

        template<typename CellCfg, typename Func>
        NPPDX_DECL_FD void apply_to_cell(Func&& func) {
            constexpr cell_config cell_cfg = CellCfg::value;
            for (int ch = 0; ch < int(cell_cfg.channel_count); ++ch) {
                for (int y = 0; y < int(cell_cfg.size.y); ++y) {
#pragma unroll
                    for (int x = 0; x < int(cell_cfg.size.x); ++x) {
                        func(int2 {x, y}, ch);
                    }
                }
            }
        }

        template<typename T, typename CellCfg, typename ChannelSliceType>
        NPPDX_DECL_FD void load_cell_from_tile(
            const nppdx::shared_memory::TileStorage<ChannelSliceType, CellCfg::value.channel_count>& tile,
            NPPCellData<CellCfg, T>& registers, int2 cell_idx, int2 offset = int2 {0, 0}) {
            constexpr cell_config cell_cfg  = CellCfg::value;
            const int2            start_pos = cell_idx * int2(cell_cfg.size) + offset;

            apply_to_cell<CellCfg>(
                [&](int2 inner_pos, int ch) { registers(inner_pos, ch) = tile(start_pos + inner_pos, ch); });
        }

        template<typename T, typename CellCfg, typename ChannelSliceType>
        NPPDX_DECL_FD void store_cell_to_tile(
            const NPPCellData<CellCfg, T>&                                                     registers,
            nppdx::shared_memory::TileStorage<ChannelSliceType, CellCfg::value.channel_count>& tile, int2 cell_idx,
            int2 offset = int2 {0, 0}) {
            constexpr auto cell_cfg  = CellCfg::value;
            const int2     start_pos = cell_idx * int2(cell_cfg.size) + offset;

            apply_to_cell<CellCfg>(
                [&](int2 inner_pos, int ch) { tile(start_pos + inner_pos, ch) = registers(inner_pos, ch); });
        }

        // Store cell to tile - with boundary clipping (for halo regions, pixel-based)
        template<typename T, typename CellCfg, typename ChannelSliceType>
        NPPDX_DECL_FD void store_cell_to_tile_clipped(
            const NPPCellData<CellCfg, T>&                                                     registers,
            nppdx::shared_memory::TileStorage<ChannelSliceType, CellCfg::value.channel_count>& tile, int2 start_pos,
            uint2 clip_size) {
            region clip_region {int2::zero(), clip_size};

            apply_to_cell<CellCfg>([&](int2 inner_pos, int ch) {
                const int2 outer_pos = start_pos + inner_pos;
                if (clip_region.contains(OuterPos(outer_pos))) {
                    tile(outer_pos, ch) = registers(inner_pos, ch);
                }
            });
        }

        // Load cell from tile - with boundary clipping (for halo regions, pixel-based)
        template<typename T, typename CellCfg, typename ChannelSliceType>
        NPPDX_DECL_FD void load_cell_from_tile_clipped(
            const nppdx::shared_memory::TileStorage<ChannelSliceType, CellCfg::value.channel_count>& tile,
            NPPCellData<CellCfg, T>& registers, int2 start_pos, uint2 clip_size, T default_value = T(0)) {
            region clip_region {int2::zero(), clip_size};

            apply_to_cell<CellCfg>([&](int2 inner_pos, int ch) {
                const int2 outer_pos = start_pos + inner_pos;
                registers(inner_pos, ch) =
                    clip_region.contains(OuterPos(outer_pos)) ? tile(outer_pos, ch) : default_value;
            });
        }


    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_DATABASE_CELL_DATA_HPP
