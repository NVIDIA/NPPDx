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

#ifndef NPPDX_OPERATORS_AFFINE_CHANNEL_MAP_HPP
#define NPPDX_OPERATORS_AFFINE_CHANNEL_MAP_HPP

#include "nppdx/operators/function.hpp"

namespace nppdx {

    namespace affine_channel_mask {
        static constexpr uint32_t channel_0 = 0x01u;
        static constexpr uint32_t channel_1 = 0x02u;
        static constexpr uint32_t channel_2 = 0x04u;

        static constexpr uint32_t channel_red   = channel_0;
        static constexpr uint32_t channel_green = channel_1;
        static constexpr uint32_t channel_blue  = channel_2;
        static constexpr uint32_t channels_rgb  = channel_red | channel_green | channel_blue;

        static constexpr uint32_t channel_y       = channel_0;
        static constexpr uint32_t channel_luma    = channel_y;
        static constexpr uint32_t channel_u       = channel_1;
        static constexpr uint32_t channel_v       = channel_2;
        static constexpr uint32_t channels_uv     = channel_u | channel_v;
        static constexpr uint32_t channels_chroma = channels_uv;

        static constexpr uint32_t channels_all = 0xFFFFFFFFu;
    } // namespace affine_channel_mask

    struct range_values_t {
        int bits;

        NPPDX_DECL_CHD int legal_shift() const { return 1 << (bits - 8); }
        NPPDX_DECL_CHD int full_max() const { return (1 << bits) - 1; }
        NPPDX_DECL_CHD int full_mid() const { return 1 << (bits - 1); }
        NPPDX_DECL_CHD int y_min() const { return 16 * legal_shift(); }
        NPPDX_DECL_CHD int y_max() const { return 235 * legal_shift(); }
        NPPDX_DECL_CHD int y_scale() const { return y_max() - y_min(); }
        NPPDX_DECL_CHD int chroma_min() const { return 16 * legal_shift(); }
        NPPDX_DECL_CHD int chroma_max() const { return 240 * legal_shift(); }
        NPPDX_DECL_CHD int chroma_mid() const { return 128 * legal_shift(); }
        NPPDX_DECL_CHD int chroma_scale() const { return chroma_max() - chroma_min(); }
    };

    template<bit_depth Depth>
    NPPDX_DECL_CI range_values_t range_values = range_values_t {static_cast<int>(Depth)};

    // RGB limited range uses the luma scaling on R, G, and B; pass channels_rgb as the mask.
    template<bit_depth Depth, uint32_t ChannelMask = affine_channel_mask::channel_luma>
    using ToLimitedLuma =
        AffineChannelMap<0, range_values<Depth>.y_scale(), range_values<Depth>.full_max(), range_values<Depth>.y_min(),
                         range_values<Depth>.y_min(), range_values<Depth>.y_max(), ChannelMask>;

    template<bit_depth Depth, uint32_t ChannelMask = affine_channel_mask::channel_luma>
    using FromLimitedLuma =
        AffineChannelMap<-range_values<Depth>.y_min(), range_values<Depth>.full_max(), range_values<Depth>.y_scale(), 0,
                         NPPDX_AFFINE_NO_CLIP, NPPDX_AFFINE_NO_CLIP, ChannelMask>;

    template<bit_depth Depth, uint32_t ChannelMask = affine_channel_mask::channels_chroma>
    using ToLimitedChroma =
        AffineChannelMap<-range_values<Depth>.full_mid(), range_values<Depth>.chroma_scale(),
                         range_values<Depth>.full_max(), range_values<Depth>.chroma_mid(),
                         range_values<Depth>.chroma_min(), range_values<Depth>.chroma_max(), ChannelMask>;

    template<bit_depth Depth, uint32_t ChannelMask = affine_channel_mask::channels_chroma>
    using FromLimitedChroma = AffineChannelMap<-range_values<Depth>.chroma_mid(), range_values<Depth>.full_max(),
                                               range_values<Depth>.chroma_scale(), range_values<Depth>.full_mid(),
                                               NPPDX_AFFINE_NO_CLIP, NPPDX_AFFINE_NO_CLIP, ChannelMask>;

} // namespace nppdx

#endif // NPPDX_OPERATORS_AFFINE_CHANNEL_MAP_HPP
