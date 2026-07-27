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

#ifndef NPPDX_DETAIL_BACKEND_TEXTURE_PACKED_RGB_HPP
#define NPPDX_DETAIL_BACKEND_TEXTURE_PACKED_RGB_HPP

#include "nppdx/detail/backend/texture/common.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {

            // Single-plane packed RGB. rgb24 (one uchar4 per pixel) uses the generic
            // single-plane load/save path; only the element pack/unpack below differs.

            template<>
            struct tex_pixel_traits<packing_format::rgb24> {
                static constexpr bool is_implemented = true;
                using element_type                   = uchar4;

                __forceinline__ __device__ static void unpack(element_type val, float& c0, float& c1, float& c2) {
                    c0 = float(val.x);
                    c1 = float(val.y);
                    c2 = float(val.z);
                }
                __forceinline__ __device__ static element_type pack(float c0, float c1, float c2) {
                    return make_uchar4(static_cast<uint8_t>(fclampf<bit_depth::bpp_8u>(c0)),
                                       static_cast<uint8_t>(fclampf<bit_depth::bpp_8u>(c1)),
                                       static_cast<uint8_t>(fclampf<bit_depth::bpp_8u>(c2)), 255);
                }
            };

            template<>
            struct texture_format_backend<packing_format::rgb24>: packed_texture_backend<packing_format::rgb24> {};

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_TEXTURE_PACKED_RGB_HPP
