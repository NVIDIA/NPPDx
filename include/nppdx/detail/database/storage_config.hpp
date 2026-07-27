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

#ifndef NPPDX_DETAIL_DATABASE_STORAGE_CONFIG_HPP
#define NPPDX_DETAIL_DATABASE_STORAGE_CONFIG_HPP

#include "nppdx/detail/config.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/operators/formats.hpp"
#include "nppdx/operators/tile_layout.hpp"
#include "nppdx/tile_storage.hpp"
#include "nppdx/types.hpp"

#include NPPDX_STD_INCLUDE_TUPLE

namespace nppdx::detail {

    // Single source of truth for the channel count of any shared-memory tile.
    // All nppdx shared-memory tiles store the internal processing representation,
    // independent of the operator's external packing format or cell layout.
    NPPDX_DECL_CI unsigned int storage_num_channels =
        unsigned(get_packing_format_props(packing_format::internal).channels);

    // Description of a single shared-memory slot a stage needs.
    // size == {0, 0} or num_channels == 0 means "no slot".
    struct tile_shape {
        uint2        size;
        unsigned int num_channels;

        NPPDX_DECL_NCHD bool empty() const { return size.x == 0 || size.y == 0 || num_channels == 0; }
    };

    NPPDX_DECL_NCHD tile_shape make_empty_tile_shape() {
        return {{0, 0}, 0};
    }

    // Aggregate description of an operator's three storage slots.
    struct storage_config {
        tile_shape input;
        tile_shape output;
        tile_shape temp;
    };

    // Helper: most operators only need a memory-tile-sized in/out pair and no temp.
    NPPDX_DECL_NCHD storage_config make_unary_storage_config(const layout_props& layout) {
        const tile_shape mem {layout.memory_size(), storage_num_channels};
        return {mem, mem, make_empty_tile_shape()};
    }

    // Helper: in-place operators consume and produce through the same memory tile.
    NPPDX_DECL_NCHD storage_config make_inplace_storage_config(const layout_props& layout) {
        const tile_shape mem {layout.memory_size(), storage_num_channels};
        return {mem, make_empty_tile_shape(), make_empty_tile_shape()};
    }

    // Lift a value-side tile_shape back into the TileStorage<ChannelSlice<...>, N>
    // tuple type expected by the rest of the runtime. Empty slots map to tuple<>.
    // Co-locating this with tile_shape is safe because tile_storage.hpp deliberately
    // does not pull in nppdx/traits.hpp, so the chain
    //   storage_config.hpp -> tile_storage.hpp
    // never closes back through the trait/operator headers that include
    // storage_config.hpp themselves.
    template<typename ProcessingType, bool Empty, unsigned int X, unsigned int Y, unsigned int N>
    struct slot_storage_tuple {
        using type = NPPDX_STD::tuple<>;
    };

    template<typename ProcessingType, unsigned int X, unsigned int Y, unsigned int N>
    struct slot_storage_tuple<ProcessingType, false, X, Y, N> {
        using type = NPPDX_STD::tuple<shared_memory::TileStorage<shared_memory::ChannelSlice<ProcessingType, X, Y>, N>>;
    };

#define NPPDX_MAKE_SLOT_STORAGE(ProcessingType, TileShape)                                                \
    typename ::nppdx::detail::slot_storage_tuple<ProcessingType, (TileShape).empty(), (TileShape).size.x, \
                                                 (TileShape).size.y, (TileShape).num_channels>::type

} // namespace nppdx::detail

#endif // NPPDX_DETAIL_DATABASE_STORAGE_CONFIG_HPP
