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

#ifndef NPPDX_OPERATORS_TILE_LAYOUT_HPP
#define NPPDX_OPERATORS_TILE_LAYOUT_HPP

#include "nppdx/operators/halo.hpp"
#include "nppdx/operators/tile_size.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/types.hpp"

namespace nppdx {

    struct layout_props {
        uint2 nominal_size;
        Halo4 memory_halo;
        Halo4 cumulative_halo;

        NPPDX_DECL_NCHD uint2 memory_size() const { return nominal_size + uint2(memory_halo.total()); }

        NPPDX_DECL_NCHD uint2 extended_size() const { return uint2(int2(nominal_size) + cumulative_halo.total()); }

        NPPDX_DECL_NCHD Halo4 padding_halo() const { return memory_halo - cumulative_halo; }
    };

    template<typename NominalTileSizeT, typename MemoryHaloT = EmptyMemoryHalo,
             typename CumulativeHaloT = EmptyCumulativeHalo>
    struct TileLayout {
        static_assert(IsUInt2D<NominalTileSizeT>);
        static_assert(IsMemoryHalo<MemoryHaloT>);
        static_assert(IsCumulativeHalo<CumulativeHaloT>);

        NPPDX_DECL_SC layout_props value = {NominalTileSizeT::value, MemoryHaloT::value, CumulativeHaloT::value};

        using NominalTileSize = NominalTileSizeT;
        using MemoryHalo      = MemoryHaloT;
        using CumulativeHalo  = CumulativeHaloT;
    };

    template<typename T>
    NPPDX_DECL_CI bool IsTileLayout = false;
    template<typename NominalTileSizeT, typename MemoryHaloT, typename CumulativeHaloT>
    NPPDX_DECL_CI bool IsTileLayout<TileLayout<NominalTileSizeT, MemoryHaloT, CumulativeHaloT>> = true;

#define NPPDX_MAKE_LAYOUT(LayoutProps)                                          \
    ::nppdx::TileLayout<NPPDX_MAKE_UINT2D((LayoutProps).nominal_size),          \
                        NPPDX_MAKE_HALO(MemoryHalo, (LayoutProps).memory_halo), \
                        NPPDX_MAKE_HALO(CumulativeHalo, (LayoutProps).cumulative_halo)>

} // namespace nppdx

#endif // NPPDX_OPERATORS_TILE_LAYOUT_HPP
