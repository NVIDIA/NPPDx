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

#ifndef NPPDX_FORMATS_HPP
#define NPPDX_FORMATS_HPP

#include "nppdx/operators/color_space.hpp"
#include "nppdx/utils.hpp"
#include "nppdx/detail/config.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/types.hpp"

#include NPPDX_STD_INCLUDE_CASSERT
#include NPPDX_STD_INCLUDE_CSTDINT

namespace nppdx {
    // Packing format enums
    enum class packing_format
    {
        none,
        internal,

        //----------------------------------------------------------------------------
        // PACKED RGB FORMATS
        //----------------------------------------------------------------------------
        rgb24,  // 8-bit packed rgb format
        rgb10,  // 10-bit packed format. A 10-bit value is stored in the most 
                // significant bits of uint16_t type. Little endian.
        rgb16,  // 16-bit packed format. Little endian.

        //----------------------------------------------------------------------------
        // BGR variants
        //----------------------------------------------------------------------------

        //----------------------------------------------------------------------------
        // PACKED YUV FORMATS
        // YUV 4:2:2
        //----------------------------------------------------------------------------
        y210, // 10-bit. The channels' order is Y0 U0 Y1 V0. A 10-bit value is stored 
              // in the most significant bits of uint16_t type. 
        uyvp, // 10-bit packed into 5 bytes per 2 pixels
        v210, // 10-bit broadcast V210: 16 bytes per 6 pixels (4 LE uint32_t words)
        yuv2, // 8-bit packed into 4 bytes per 2 pixels. The channels' order is 
              // Y0 U0 Y1 V0.

        //----------------------------------------------------------------------------
        // SEMI-PLANAR FORMATS
        // YUV 4:2:0
        //----------------------------------------------------------------------------
        nv12, // 8-bit format
        p010, // 10-bit format. A 10-bit value is stored in the most significant bits 
              // of uint16_t type. Little endian. 

        //----------------------------------------------------------------------------
        // SEMI-PLANAR FORMATS
        // YUV 4:2:2
        //----------------------------------------------------------------------------
        nv16, // 8-bit format
        p216, // 16-bit format

        //----------------------------------------------------------------------------
        // FULLY PLANAR FORMATS
        //----------------------------------------------------------------------------
        rgbp,       // 8-bit format
        bgrp,       // 8-bit format
        yuv420p,    // 8-bit format with 4:2:0 subsampling
        yuv420p10,  // 10-bit format with 4:2:0 subsampling. A 10-bit value is stored 
                    // in the least significant bits of uint16_t type. Little endian.
        yuv422p,    // 8-bit format with 4:2:2 subsampling
        yuv422p10,  // 10-bit format with 4:2:2 subsampling. A 10-bit value is stored 
                    // in the least significant bits of uint16_t type. Little endian.
        yuv444p,    // 8-bit format without subsampling
        yuv444p10   // 10-bit format without subsampling. A 10-bit value is stored 
                    // in the least significant bits of uint16_t type. Little endian.
    };

    // Note for developers: Whenever adding or removing packing_format enumerators,
    // update NPPDX_FOR_PACKING_FORMATS at the bottom of this header accordingly.

    struct packing_format_props {
        packing_format format;
        int            channels;
        int            bits_per_channel;
        int2           subsampling;
        int            planes;
        int            bytes_per_pixel; // Integer x-dimension byte stride per pixel within a
                                        // non-subsampled plane. For packed subsampled formats
                                        // (uyvp, v210, y210) this is an integer approximation of a
                                        // non-integral bytes-per-pixel ratio and MUST NOT be used to
                                        // compute strides — use minimum_row_byte_stride(), which
                                        // special-cases them.
        color_space default_color_space;
        bool        is_floating_point = false;

        NPPDX_DECL_NCHD bool is_planar() const { return channels == planes; }

        NPPDX_DECL_NCHD bool is_subsampled() const { return subsampling.x > 1 || subsampling.y > 1; }

        NPPDX_DECL_NCHD size_t minimum_row_byte_stride(int plane_idx, size_t width) const {
            // Packed 4:2:2 (horizontal subsampling): row stride is per macroblock pair,
            // not width*bytes_per_pixel. These formats are single-plane, so the stride
            // is independent of plane_idx (only plane_idx == 0 is ever observed in practice).
            if (format == nppdx::packing_format::y210) {
                return ceil_div(width, size_t(subsampling.x)) * 4 * sizeof(NPPDX_STD::uint16_t);
            }
            if (format == nppdx::packing_format::uyvp) {
                return ceil_div(width, size_t(subsampling.x)) * 5;
            }
            if (format == nppdx::packing_format::v210) {
                // FFmpeg / Apple: row stride padded to 48-pixel (128-byte) boundaries.
                // Matches image-test reference packing (formats.py _pack_v210).
                return ceil_div(width, size_t(48)) * 128;
            }
            if (plane_idx == 0) {
                return width * bytes_per_pixel;
            }
            
            // Semi-planar NV-family interleaves U and V in the chroma plane (2 samples per chroma
            // column), so the row holds 2 * (width / subsampling.x) samples. The 4:2:0 (nv12/p010)
            // and 4:2:2 (nv16/p216) variants share this packing; they differ only in vertical
            // subsampling, which affects the chroma plane height, not its row stride.
            const size_t sub_packing_factor =
                (format == nppdx::packing_format::nv12 || format == nppdx::packing_format::p010 ||
                 format == nppdx::packing_format::nv16 || format == nppdx::packing_format::p216)
                    ? 2
                    : 1;
            return sub_packing_factor * (width / size_t(subsampling.x)) * bytes_per_pixel;
        }

        // The number of bytes in a buffer with minimal padding representing an
        // image of size width x height.
        NPPDX_DECL_NCHD size_t minimum_buffer_byte_size(size_t width, size_t height) const {
            return height * minimum_row_byte_stride(0, width) +
                   (planes - 1) * (height / size_t(subsampling.y)) * minimum_row_byte_stride(1, width);
        }
    };

    NPPDX_DECL_NCFHD packing_format_props get_packing_format_props(packing_format format) {
        switch (format) {
            case packing_format::none:
                return packing_format_props {/*format=*/packing_format::none,
                                             /*channels=*/0,
                                             /*bits_per_channel=*/0,
                                             /*subsampling=*/int2 {1, 1},
                                             /*planes=*/0,
                                             /*bytes_per_pixel=*/0,
                                             /*default_color_space=*/color_space::rgb};
            case packing_format::internal:
                return packing_format_props {/*format=*/packing_format::internal,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/32,
                                             /*subsampling=*/int2 {1, 1},
                                             /*planes=*/3,
                                             /*bytes_per_pixel=*/4,
                                             /*default_color_space=*/color_space::rgb,
                                             /*is_floating_point=*/true};
            case packing_format::rgb24:
                return packing_format_props {/*format=*/packing_format::rgb24,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/8,
                                             /*subsampling=*/int2 {1, 1},
                                             /*planes=*/1,
                                             /*bytes_per_pixel=*/3,
                                             /*default_color_space=*/color_space::rgb};
            case packing_format::rgb10:
                return packing_format_props {/*format=*/packing_format::rgb10,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/10,
                                             /*subsampling=*/int2 {1, 1},
                                             /*planes=*/1,
                                             /*bytes_per_pixel=*/4,
                                             /*default_color_space=*/color_space::rgb};
            case packing_format::rgb16:
                return packing_format_props {/*format=*/packing_format::rgb16,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/16,
                                             /*subsampling=*/int2 {1, 1},
                                             /*planes=*/1,
                                             /*bytes_per_pixel=*/6,
                                             /*default_color_space=*/color_space::rgb};
            case packing_format::y210:
                return packing_format_props {/*format=*/packing_format::y210,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/10,
                                             /*subsampling=*/int2 {2, 1},
                                             /*planes=*/1,
                                             /*bytes_per_pixel=*/4,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::uyvp:
                return packing_format_props {/*format=*/packing_format::uyvp,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/10,
                                             /*subsampling=*/int2 {2, 1},
                                             /*planes=*/1,
                                             /*bytes_per_pixel=*/2,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::v210:
                return packing_format_props {/*format=*/packing_format::v210,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/10,
                                             /*subsampling=*/int2 {2, 1},
                                             /*planes=*/1,
                                             /*bytes_per_pixel=*/3,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::yuv2:
                return packing_format_props {/*format=*/packing_format::yuv2,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/8,
                                             /*subsampling=*/int2 {2, 1},
                                             /*planes=*/1,
                                             /*bytes_per_pixel=*/2, // YUY2: 4 bytes per 2-pixel pair = 2 bytes/pixel
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::nv12:
                return packing_format_props {/*format=*/packing_format::nv12,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/8,
                                             /*subsampling=*/int2 {2, 2},
                                             /*planes=*/2,
                                             /*bytes_per_pixel=*/1,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::p010: 
                return packing_format_props {/*format=*/packing_format::p010,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/10,
                                             /*subsampling=*/int2 {2, 2},
                                             /*planes=*/2,
                                             /*bytes_per_pixel=*/2,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::nv16: 
                return packing_format_props {/*format=*/packing_format::nv16,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/8,
                                             /*subsampling=*/int2 {2, 1},
                                             /*planes=*/2,
                                             /*bytes_per_pixel=*/1,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::p216: 
                return packing_format_props {/*format=*/packing_format::p216,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/16,
                                             /*subsampling=*/int2 {2, 1},
                                             /*planes=*/2,
                                             /*bytes_per_pixel=*/2,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::bgrp:
                return packing_format_props {/*format=*/packing_format::bgrp,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/8,
                                             /*subsampling=*/int2 {1, 1},
                                             /*planes=*/3,
                                             /*bytes_per_pixel=*/1,
                                             /*default_color_space=*/color_space::rgb};
            case packing_format::rgbp: 
                return packing_format_props {/*format=*/packing_format::rgbp,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/8,
                                             /*subsampling=*/int2 {1, 1},
                                             /*planes=*/3,
                                             /*bytes_per_pixel=*/1,
                                             /*default_color_space=*/color_space::rgb};
            case packing_format::yuv420p:
                return packing_format_props {/*format=*/packing_format::yuv420p,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/8,
                                             /*subsampling=*/int2 {2, 2},
                                             /*planes=*/3,
                                             /*bytes_per_pixel=*/1,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::yuv420p10:
                return packing_format_props {/*format=*/packing_format::yuv420p10,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/10,
                                             /*subsampling=*/int2 {2, 2},
                                             /*planes=*/3,
                                             /*bytes_per_pixel=*/2,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::yuv422p:
                return packing_format_props {/*format=*/packing_format::yuv422p,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/8,
                                             /*subsampling=*/int2 {2, 1},
                                             /*planes=*/3,
                                             /*bytes_per_pixel=*/1,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::yuv422p10:
                return packing_format_props {/*format=*/packing_format::yuv422p10,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/10,
                                             /*subsampling=*/int2 {2, 1},
                                             /*planes=*/3,
                                             /*bytes_per_pixel=*/2,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::yuv444p:
                return packing_format_props {/*format=*/packing_format::yuv444p,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/8,
                                             /*subsampling=*/int2 {1, 1},
                                             /*planes=*/3,
                                             /*bytes_per_pixel=*/1,
                                             /*default_color_space=*/color_space::yuv_bt601};
            case packing_format::yuv444p10:
                return packing_format_props {/*format=*/packing_format::yuv444p10,
                                             /*channels=*/3,
                                             /*bits_per_channel=*/10,
                                             /*subsampling=*/int2 {1, 1},
                                             /*planes=*/3,
                                             /*bytes_per_pixel=*/2,
                                             /*default_color_space=*/color_space::yuv_bt601};
            default: fatal_error();
        }
    }

    template<packing_format Format>
    struct packing_format_helper {
        NPPDX_DECL_SC packing_format_props props = get_packing_format_props(Format);

        using element_type = NPPDX_STD::conditional_t<props.is_floating_point, float, NPPDX_STD::uint8_t>;

        template<typename T>
        NPPDX_DECL_NSCHD auto minimum_image_layout(T* ptr, size2 size) {
            static_assert(NPPDX_STD::is_same_v<NPPDX_STD::remove_const_t<T>, element_type>,
                          "Pointer type must match element type");
            ImageLayout<T, props.planes> ret_val {};

            packing_format_props local_props = props;

            ret_val.ptrs.value[0]         = ptr;
            ret_val.byte_strides.value[0] = static_cast<int>(local_props.minimum_row_byte_stride(0, size.x));

            if constexpr (props.planes > 1) {
                ret_val.ptrs.value[1] = ret_val.ptrs.value[0] + size.y * (ret_val.byte_strides.value[0] / sizeof(T));
                ret_val.byte_strides.value[1] = static_cast<int>(local_props.minimum_row_byte_stride(1, size.x));

                for (int i = 2; i < props.planes; i++) {
                    ret_val.ptrs.value[i] =
                        ret_val.ptrs.value[i - 1] +
                        (size.y / size_t(props.subsampling.y)) * (ret_val.byte_strides.value[i - 1] / sizeof(T));
                    ret_val.byte_strides.value[i] = static_cast<int>(local_props.minimum_row_byte_stride(i, size.x));
                }
            }

            return ret_val;
        }
    };

    // Apply a custom macro to all _external_ packing formats (i.e., not none or
    // internal).
    // This is not type safe and must be used with care but has the advantage
    // that it can be used for switch statements, stringification, etc.
    // In particular, note that the first argument to the macro is the unquoted
    // enumerator name which does not yet name a value. A value can be named as
    // packing_format::arg1 (or ::nppdx::packing_format::arg1 for use in
    // arbitrary namespaces).
    // The macro used must take more than one parameter, but can ignore any of
    // the parameters.
#define NPPDX_FOR_PACKING_FORMATS(MACRO, ...) \
    MACRO(rgb24, __VA_ARGS__)                 \
    MACRO(rgb10, __VA_ARGS__)                 \
    MACRO(rgb16, __VA_ARGS__)                 \
    MACRO(y210, __VA_ARGS__)                  \
    MACRO(uyvp, __VA_ARGS__)                  \
    MACRO(v210, __VA_ARGS__)                  \
    MACRO(nv12, __VA_ARGS__)                  \
    MACRO(p010, __VA_ARGS__)                  \
    MACRO(nv16, __VA_ARGS__)                  \
    MACRO(p216, __VA_ARGS__)                  \
    MACRO(rgbp, __VA_ARGS__)                  \
    MACRO(bgrp, __VA_ARGS__)                  \
    MACRO(yuv420p, __VA_ARGS__)               \
    MACRO(yuv420p10, __VA_ARGS__)             \
    MACRO(yuv422p, __VA_ARGS__)               \
    MACRO(yuv422p10, __VA_ARGS__)             \
    MACRO(yuv444p, __VA_ARGS__)               \
    MACRO(yuv444p10, __VA_ARGS__)             \
    MACRO(yuv2, __VA_ARGS__)

    template<bool IncludeInternal = false, bool IncludeNone = false, typename FormatCarrierT, typename FunctorT,
             typename... ArgsT>
    constexpr auto dispatch_format(const FormatCarrierT& formatCarrier, FunctorT&& functor, ArgsT&&... args) {
        static_assert(has_carried_type<packing_format, FormatCarrierT>);

#define NPPDX_PRIVATE_CASE_BODY_PF(format) \
    NPPDX_STD::forward<FunctorT>(functor)(format, NPPDX_STD::forward<ArgsT>(args)...)

        if constexpr (is_cval<FormatCarrierT>) {
            return NPPDX_PRIVATE_CASE_BODY_PF(formatCarrier);
        } else {
#define NPPDX_PRIVATE_CASE_PF(format, ...) \
    case packing_format::format: return NPPDX_PRIVATE_CASE_BODY_PF(C<packing_format::format>);
            switch (formatCarrier) {
                NPPDX_FOR_PACKING_FORMATS(NPPDX_PRIVATE_CASE_PF)
                default: break;
            }

            if constexpr (IncludeInternal) {
                if (formatCarrier == packing_format::internal) {
                    return NPPDX_PRIVATE_CASE_BODY_PF(C<packing_format::internal>);
                }
            }

            if constexpr (IncludeNone) {
                if (formatCarrier == packing_format::none) {
                    return NPPDX_PRIVATE_CASE_BODY_PF(C<packing_format::none>);
                }
            }

            fatal_error();

#undef NPPDX_PRIVATE_CASE_PF
        }

        // Even though the code below is unreachable, it appears to be a
        // necessary workaround for GCC 7 to correctly infer the return type.
        return NPPDX_PRIVATE_CASE_BODY_PF(C<packing_format::rgb24>);

#undef NPPDX_PRIVATE_CASE_BODY_PF
    }

} // namespace nppdx

#ifndef NPPDX_FORMATS_COMPAT_HPP
#    define NPPDX_USE_COMPAT_FORMATS
#    include "formats_compat.hpp"
#    undef NPPDX_USE_COMPAT_FORMATS
#endif // NPPDX_FORMATS_COMPAT_HPP

#endif // NPPDX_FORMATS_HPP
