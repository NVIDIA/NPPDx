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

#ifndef NPPDX_USE_COMPAT_FORMATS
#    error "This header should not be included directly. Include nppdx/operators/formats.hpp instead."
#endif

#ifndef NPPDX_FORMATS_COMPAT_HPP
#    define NPPDX_FORMATS_COMPAT_HPP

#    include "nppdx/detail/config.hpp"
#    include NPPDX_STD_INCLUDE_TYPE_TRAITS
#    include NPPDX_STD_INCLUDE_CSTDINT

// Should already be included, including for the sake of autocompletion tools.
#    include "nppdx/operators/formats.hpp"

namespace nppdx {
    // This is true for regular formats. For exceptions, packing_format_traits
    // are specialized.
    template<bool is_floating_point>
    using pixel_type = NPPDX_STD::conditional_t<is_floating_point, float, uint8_t>;

    template<packing_format Format>
    struct packing_format_traits {
        static constexpr packing_format_props props = get_packing_format_props(Format);
        using pixel_type                            = pixel_type<props.is_floating_point>;
        static constexpr int  channels              = props.channels;
        static constexpr int  bits_per_channel      = props.bits_per_channel;
        static constexpr bool is_planar             = props.is_planar();
        static constexpr bool is_subsampled         = props.is_subsampled();
        static constexpr int  subsample_x           = props.subsampling.x;
        static constexpr int  subsample_y           = props.subsampling.y;
        static constexpr int  planes                = props.planes;
        static constexpr int  bytes_per_pixel       = props.bytes_per_pixel;
    };

    // Note: by the looks of it, the deviations from the general template are
    // may actually be bugs. Should be resolved soon.

    // RGB10 specialization
    template<>
    struct packing_format_traits<packing_format::rgb10> {
        static constexpr packing_format_props props = get_packing_format_props(packing_format::rgb10);
        using pixel_type                            = NPPDX_STD::uint32_t;
        static constexpr int  channels              = 3;
        static constexpr int  bits_per_channel      = 10;
        static constexpr bool is_planar             = false;
        static constexpr bool is_subsampled         = false;
        static constexpr int  subsample_x           = 1;
        static constexpr int  subsample_y           = 1;
        static constexpr int  planes                = 1;
        static constexpr int  bytes_per_pixel       = 4; // 1 uint32_t per pixel (packed RGB10)
    };

    // Y210 specialization
    template<>
    struct packing_format_traits<packing_format::y210> {
        static constexpr packing_format_props props = get_packing_format_props(packing_format::y210);
        using pixel_type                            = NPPDX_STD::uint16_t;
        static constexpr int  channels              = 3;
        static constexpr int  bits_per_channel      = 10;
        static constexpr bool is_planar             = false;
        static constexpr bool is_subsampled         = true;
        static constexpr int  subsample_x           = 2;
        static constexpr int  subsample_y           = 1;
        static constexpr int  planes                = 1;
        static constexpr int  bytes_per_pixel       = 4; // 4 uint16_t per 2 pixels = 8 bytes/2px = 4 bytes/pixel
    };

    // UYVP specialization (10-bit YUV 4:2:2 packed into 5 bytes per 2 pixels)
    template<>
    struct packing_format_traits<packing_format::uyvp> {
        static constexpr packing_format_props props = get_packing_format_props(packing_format::uyvp);
        using pixel_type                            = uint16_t; // 10-bit in upper bits, processed as float
        static constexpr int  channels              = 3;
        static constexpr int  bits_per_channel      = 10;
        static constexpr bool is_planar             = false;
        static constexpr bool is_subsampled         = true;
        static constexpr int  subsample_x           = 2;
        static constexpr int  subsample_y           = 1;
        static constexpr int  planes                = 1;
        // 5 bytes per 2 pixels (2.5 bpp). Stored as 2 (integer truncation).
        // Do not use to compute stride; see packing_format_props::minimum_row_byte_stride.
        static constexpr int bytes_per_pixel = 2;
    };

    // v210 specialization (10-bit YUV 4:2:2 broadcast V210: 16 bytes per 6 pixels)
    template<>
    struct packing_format_traits<packing_format::v210> {
        static constexpr packing_format_props props = get_packing_format_props(packing_format::v210);
        using pixel_type                            = uint16_t;
        static constexpr int  channels              = 3;
        static constexpr int  bits_per_channel      = 10;
        static constexpr bool is_planar             = false;
        static constexpr bool is_subsampled         = true;
        static constexpr int  subsample_x           = 2;
        static constexpr int  subsample_y           = 1;
        static constexpr int  planes                = 1;
        // 6 bytes per 2 pixels (3 bpp, exact only when width is even, which V210 requires).
        // Do not use to compute stride; see packing_format_props::minimum_row_byte_stride.
        static constexpr int bytes_per_pixel = 3;
    };

    // Internal processing format - always float32 RGBA
    using processing_format = packing_format_traits<packing_format::rgb24>; // Placeholder, will be specialized

} // namespace nppdx

#endif // NPPDX_FORMATS_COMPAT_HPP
