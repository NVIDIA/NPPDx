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

#ifndef NPPDX_DETAIL_BACKEND_TEXTURE_COMMON_HPP
#define NPPDX_DETAIL_BACKEND_TEXTURE_COMMON_HPP

#include "nppdx/detail/backend/formats_impl/clamp_utils.hpp"  // bit_depth, fclampf
#include "nppdx/operators/formats.hpp"

#include NPPDX_STD_INCLUDE_CSTDINT

namespace nppdx {
    namespace detail {
        namespace backend {

            // Value-dependent false for static_assert in unreachable `if constexpr` branches.
            template<packing_format>
            inline constexpr bool dependent_false_format = false;

            // Sample pack/unpack shared by the semi-planar backends. 10-bit-in-16 formats (p010)
            // store the value left-shifted into the container's MSBs; undo on read, redo on write.
            template<int Shift, typename Raw>
            __forceinline__ __device__ float sp_unpack(Raw raw) {
                return float(static_cast<uint32_t>(raw) >> Shift);
            }
            template<typename Sample, bit_depth R, int Shift>
            __forceinline__ __device__ Sample sp_pack(float value) {
                return static_cast<Sample>(static_cast<uint32_t>(fclampf<R>(value)) << Shift);
            }

            // =================================================================
            // Texture/surface pixel traits — per-format pack/unpack for CUDA array elements.
            // =================================================================

            template<packing_format Format>
            struct tex_pixel_traits {
                static constexpr bool is_implemented = false;
            };

            // =================================================================
            // Generic tex_load / surf_save for single-plane packed formats.
            // The per-format element type and pack/unpack come from tex_pixel_traits.
            // =================================================================

            template<packing_format Format, typename CellDataType>
            __forceinline__ __device__ void generic_tex_load(
                TexObjT tex, CellDataType& cell,
                int32_t nCellX, int32_t nCellY, uint32_t imageWidth, uint32_t imageHeight)
            {
                using traits = tex_pixel_traits<Format>;
                static_assert(traits::is_implemented, "tex_pixel_traits not implemented for this format");
                using element_type = typename traits::element_type;

                constexpr int cell_width  = int(CellDataType::cell_cfg.size.x);
                constexpr int cell_height = int(CellDataType::cell_cfg.size.y);

                float* ch0 = cell.channel_ptr(0);
                float* ch1 = cell.channel_ptr(1);
                float* ch2 = cell.channel_ptr(2);

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;

#pragma unroll
                for (int row = 0; row < cell_height; ++row) {
#pragma unroll
                    for (int col = 0; col < cell_width; ++col) {
                        const element_type val = tex2D<element_type>(tex, basePixelX + col, basePixelY + row);
                        const int          idx = row * cell_width + col;
                        traits::unpack(val, ch0[idx], ch1[idx], ch2[idx]);
                    }
                }
            }

            // No bounds check: cudaBoundaryModeZero drops out-of-image writes (array == image).
            template<packing_format Format, typename CellDataType>
            __forceinline__ __device__ void generic_surf_save(
                SurfObjT surf, const CellDataType& cell,
                int32_t nCellX, int32_t nCellY, uint32_t imageWidth, uint32_t imageHeight)
            {
                using traits = tex_pixel_traits<Format>;
                static_assert(traits::is_implemented, "tex_pixel_traits not implemented for this format");

                constexpr int cell_width  = int(CellDataType::cell_cfg.size.x);
                constexpr int cell_height = int(CellDataType::cell_cfg.size.y);

                const float* ch0 = cell.channel_ptr(0);
                const float* ch1 = cell.channel_ptr(1);
                const float* ch2 = cell.channel_ptr(2);

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;
                constexpr int elem_bytes = int(sizeof(typename traits::element_type));

#pragma unroll
                for (int row = 0; row < cell_height; ++row) {
#pragma unroll
                    for (int col = 0; col < cell_width; ++col) {
                        const int  idx = row * cell_width + col;
                        const auto val = traits::pack(ch0[idx], ch1[idx], ch2[idx]);
                        surf2Dwrite(val, surf, (basePixelX + col) * elem_bytes, basePixelY + row,
                                    cudaBoundaryModeZero);
                    }
                }
            }

            // =================================================================
            // texture_format_backend — per-format adapter mapping packing_format to
            // texture/surface load/save, mirroring format_backend in
            // ingest_exgest_operations.hpp. tex_load/surf_save take an array of
            // plane handles (length == packing_format_helper<Format>::props.planes),
            // so the execute() overloads dispatch uniformly with no format `if`s.
            // =================================================================

            template<packing_format Format>
            struct texture_format_backend {
                static constexpr bool is_implemented = false;
            };

            // Single-array (stacked) adapter for multi-plane formats: all planes in one cuArray.
            // Only semi-planar formats specialize.
            template<packing_format Format>
            struct texture_stacked_backend {
                static constexpr bool is_implemented = false;
            };

            // Backend for a single cuArray handle: 1-plane formats use the packed backend,
            // multi-plane use the stacked backend.
            template<packing_format Format, bool SinglePlane = (packing_format_helper<Format>::props.planes == 1)>
            struct single_array_backend : texture_format_backend<Format> {};

            template<packing_format Format>
            struct single_array_backend<Format, /*SinglePlane=*/false> : texture_stacked_backend<Format> {};

            // Backend selected by how many cuArray handles address the image, so the texture/surface
            // execute() overloads dispatch uniformly on the handle count:
            //   1 handle  -> single_array_backend (packed plane for single-plane formats,
            //                stacked array for multi-plane formats)
            //   N handles -> texture_format_backend (one handle per plane)
            template<packing_format Format, int NumHandles>
            struct texture_handle_backend : texture_format_backend<Format> {};

            template<packing_format Format>
            struct texture_handle_backend<Format, 1> : single_array_backend<Format> {};

            // Reusable backend for single-plane packed formats: one texel == one pixel,
            // pack/unpack supplied by tex_pixel_traits<Format>. A new packed format only
            // needs a tex_pixel_traits<Format> specialization plus a one-line
            //   template<> struct texture_format_backend<Format> : packed_texture_backend<Format> {};
            template<packing_format Format>
            struct packed_texture_backend {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void tex_load(const TexObjT* texPlanes,
                                                                CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                uint32_t imageWidth, uint32_t imageHeight) {
                    generic_tex_load<Format>(texPlanes[0], cell, nCellX, nCellY, imageWidth, imageHeight);
                }
                template<typename CellDataType>
                __forceinline__ __device__ static void surf_save(const SurfObjT* surfPlanes,
                                                                 const CellDataType& cell, int32_t nCellX,
                                                                 int32_t nCellY, uint32_t imageWidth,
                                                                 uint32_t imageHeight) {
                    generic_surf_save<Format>(surfPlanes[0], cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_TEXTURE_COMMON_HPP
