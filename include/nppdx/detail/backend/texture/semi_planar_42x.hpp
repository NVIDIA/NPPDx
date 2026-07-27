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

#ifndef NPPDX_DETAIL_BACKEND_TEXTURE_SEMI_PLANAR_42X_HPP
#define NPPDX_DETAIL_BACKEND_TEXTURE_SEMI_PLANAR_42X_HPP

#include "nppdx/detail/backend/texture/common.hpp"
#include "nppdx/utils.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {

            // =================================================================
            // Semi-planar 4:2:x (NV12/NV16 families) — two-plane texture ingest and surface exgest.
            // Plane 0: luma Y, full-res (one LumaT per pixel). Plane 1: chroma UV interleaved,
            // half-width (one ChromaT per horizontal luma pair). The chroma vertical resolution
            // depends on the subsampling:
            //  - 4:2:0 (nv12/p010): chroma is also half-height (one UV per 2x2 luma block);
            //  - 4:2:2 (nv16/p216): chroma keeps full height (one UV per 2x1 luma pair).
            // The register cell holds 3ch (Y,U,V) at full luma res. Parameterized by the
            // subsampling, the plane sample types, the bit depth (clamp range) and a container
            // shift so NV12 (8-bit uchar), P010 (10-bit in the MSBs of uint16), NV16 (8-bit uchar)
            // and P216 (full 16-bit) all share one implementation. Mirrors NV12Loader/NV12Saver
            // in formats_impl/nv12_family.hpp (4:2:2 is VDecimate=false).
            // =================================================================

            // Chroma subsampling selector for the semi-planar family.
            enum class semi_planar_chroma
            {
                yuv_420,
                yuv_422
            };

            // Semi-planar 4:2:x texture ingest for both topologies, selected by Stacked:
            //  - Stacked=false (per-plane): luma from texY, interleaved chroma read as a 2-component
            //    ChromaT from a separate texUV plane at (chromaX, chromaY).
            //  - Stacked=true (single array): one array holds luma then chroma below it (rows
            //    [imageHeight, imageHeight + chromaHeight)); chroma is read per-component from that
            //    same single-channel array (cols 2*chromaX and 2*chromaX+1), so texUV == texY.
            template<semi_planar_chroma Chroma, typename LumaT, typename ChromaT, bit_depth R, int Shift,
                     bool Stacked = false, typename CellDataType>
            __forceinline__ __device__ void semi_planar_42x_tex_load(TexObjT texY, TexObjT texUV, CellDataType& cell,
                                                                     int32_t nCellX, int32_t nCellY,
                                                                     uint32_t imageWidth, uint32_t imageHeight) {
                using ChromaComp    = decltype(ChromaT::x);
                constexpr bool VSub = (Chroma == semi_planar_chroma::yuv_420);

                constexpr int cell_width  = int(CellDataType::cell_cfg.size.x);
                constexpr int cell_height = int(CellDataType::cell_cfg.size.y);

                float* ch_Y = cell.channel_ptr(0);
                float* ch_U = cell.channel_ptr(1);
                float* ch_V = cell.channel_ptr(2);

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;

                const int32_t maxX = int32_t(imageWidth) - 1;
                const int32_t maxY = int32_t(imageHeight) - 1;
                // 4:2:0 chroma is half-res in both axes (needs a full 2x2 block); 4:2:2 chroma is
                // half-width, full-height (needs a 2x1 pair). Guard the degenerate small image the
                // way NV12Loader does (fill chroma with zero when no UV sample exists).
                const bool    hasChroma = imageWidth >= 2 && (!VSub || imageHeight >= 2);
                const int32_t uvMaxX    = int32_t(imageWidth / 2) - 1;
                const int32_t uvMaxY    = VSub ? int32_t(imageHeight / 2) - 1 : maxY;
                // Stacked layout only: chroma rows begin directly below the luma plane.
                [[maybe_unused]] const int32_t chromaRow0 = int32_t(imageHeight);

#pragma unroll
                for (int row = 0; row < cell_height; ++row) {
                    const int32_t lumaY = nppdx::clamp_value(basePixelY + row, 0, maxY);
                    // 4:2:0 samples one chroma row per two luma rows; 4:2:2 keeps a chroma row per luma row.
                    const int32_t chromaY = VSub ? nppdx::clamp_value(lumaY / 2, 0, uvMaxY) : lumaY;
#pragma unroll
                    for (int col = 0; col < cell_width; ++col) {
                        const int32_t lumaX = nppdx::clamp_value(basePixelX + col, 0, maxX);
                        const int     idx   = row * cell_width + col;

                        ch_Y[idx] = sp_unpack<Shift>(tex2D<LumaT>(texY, lumaX, lumaY));
                        if (hasChroma) {
                            const int32_t chromaX = nppdx::clamp_value(lumaX / 2, 0, uvMaxX);
                            if constexpr (Stacked) {
                                ch_U[idx] =
                                    sp_unpack<Shift>(tex2D<ChromaComp>(texUV, 2 * chromaX, chromaRow0 + chromaY));
                                ch_V[idx] =
                                    sp_unpack<Shift>(tex2D<ChromaComp>(texUV, 2 * chromaX + 1, chromaRow0 + chromaY));
                            } else {
                                const ChromaT uv = tex2D<ChromaT>(texUV, chromaX, chromaY);
                                ch_U[idx]        = sp_unpack<Shift>(uv.x);
                                ch_V[idx]        = sp_unpack<Shift>(uv.y);
                            }
                        } else {
                            ch_U[idx] = 0.0f;
                            ch_V[idx] = 0.0f;
                        }
                    }
                }
            }

            // Semi-planar 4:2:x surface exgest for both topologies, selected by Stacked:
            //  - Stacked=false (per-plane): luma to surfY; chroma written as a 2-component ChromaT to a
            //    separate surfUV plane at (chromaX, chromaY). cudaBoundaryModeZero drops out-of-image
            //    writes (arrays == plane sizes), so no explicit bounds check is needed.
            //  - Stacked=true (single array): one array holds luma then chroma below it (rows
            //    [imageHeight, imageHeight + chromaHeight)); chroma is written per-component to that
            //    same single-channel array (cols 2*chromaX and 2*chromaX+1), so surfUV == surfY. The
            //    luma row guard stays: the array is taller than the luma plane (chroma stacked below),
            //    so a stray luma row would otherwise land in the chroma region.
            template<semi_planar_chroma Chroma, typename LumaT, typename ChromaT, bit_depth R, int Shift,
                     bool Stacked = false, typename CellDataType>
            __forceinline__ __device__ void semi_planar_42x_surf_save(SurfObjT surfY, SurfObjT surfUV,
                                                                      const CellDataType& cell, int32_t nCellX,
                                                                      int32_t nCellY, uint32_t imageWidth,
                                                                      uint32_t imageHeight) {
                using ChromaComp    = decltype(ChromaT::x);
                constexpr bool VSub = (Chroma == semi_planar_chroma::yuv_420);

                constexpr int cell_width  = int(CellDataType::cell_cfg.size.x);
                constexpr int cell_height = int(CellDataType::cell_cfg.size.y);
                static_assert((cell_width % 2) == 0, "Semi-planar 4:2:x requires an even cell width");
                static_assert(!VSub || (cell_height % 2) == 0, "Semi-planar 4:2:0 requires an even cell height");

                const float* ch_Y = cell.channel_ptr(0);
                const float* ch_U = cell.channel_ptr(1);
                const float* ch_V = cell.channel_ptr(2);

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;
                constexpr int luma_bytes = int(sizeof(LumaT));
                // Stacked layout only: chroma rows begin directly below the luma plane.
                [[maybe_unused]] const int32_t chromaRow0 = int32_t(imageHeight);

#pragma unroll
                for (int row = 0; row < cell_height; ++row) {
                    const int32_t lumaY = basePixelY + row;
                    if constexpr (Stacked) {
                        if (lumaY < 0 || lumaY >= int32_t(imageHeight)) {
                            continue;
                        }
                    }
                    const float* row_Y = &ch_Y[row * cell_width];
#pragma unroll
                    for (int col = 0; col < cell_width; ++col) {
                        const LumaT y = sp_pack<LumaT, R, Shift>(row_Y[col]);
                        surf2Dwrite(y, surfY, (basePixelX + col) * luma_bytes, lumaY, cudaBoundaryModeZero);
                    }

                    // 4:2:0 writes chroma once per 2x2 block (on the even row of the block) as the
                    // average of the block's four U/V samples; 4:2:2 writes chroma every row as the
                    // average of each 2x1 horizontal pair. Both match NV12Saver.
                    if (!VSub || (row & 1) == 0) {
                        const int32_t chromaY = VSub ? lumaY / 2 : lumaY;
                        const float*  row_U   = &ch_U[row * cell_width];
                        const float*  row_V   = &ch_V[row * cell_width];
#pragma unroll
                        for (int i = 0; i < cell_width / 2; ++i) {
                            float U, V;
                            if constexpr (VSub) {
                                U = (row_U[2 * i] + row_U[2 * i + 1] + row_U[cell_width + 2 * i] +
                                     row_U[cell_width + 2 * i + 1]) /
                                    4.0f;
                                V = (row_V[2 * i] + row_V[2 * i + 1] + row_V[cell_width + 2 * i] +
                                     row_V[cell_width + 2 * i + 1]) /
                                    4.0f;
                            } else {
                                U = (row_U[2 * i] + row_U[2 * i + 1]) / 2.0f;
                                V = (row_V[2 * i] + row_V[2 * i + 1]) / 2.0f;
                            }
                            if constexpr (Stacked) {
                                // Per-component write: U at col 2*chromaX, V at 2*chromaX+1.
                                const int32_t    chromaX    = basePixelX / 2 + i;
                                constexpr int    comp_bytes = int(sizeof(ChromaComp));
                                const ChromaComp u          = sp_pack<ChromaComp, R, Shift>(U);
                                const ChromaComp v          = sp_pack<ChromaComp, R, Shift>(V);
                                surf2Dwrite(u, surfUV, (2 * chromaX) * comp_bytes, chromaRow0 + chromaY,
                                            cudaBoundaryModeZero);
                                surf2Dwrite(v, surfUV, (2 * chromaX + 1) * comp_bytes, chromaRow0 + chromaY,
                                            cudaBoundaryModeZero);
                            } else {
                                ChromaT uv;
                                uv.x = sp_pack<ChromaComp, R, Shift>(U);
                                uv.y = sp_pack<ChromaComp, R, Shift>(V);
                                surf2Dwrite(uv, surfUV, (basePixelX / 2 + i) * static_cast<int>(sizeof(ChromaT)),
                                            chromaY, cudaBoundaryModeZero);
                            }
                        }
                    }
                }
            }

            // Reusable backend for the semi-planar 4:2:x family: two planes (luma + interleaved
            // chroma), parameterized by the chroma subsampling, plane sample types, bit depth and
            // container shift. A new format only needs a one-line
            //   template<> struct texture_format_backend<Format>
            //       : semi_planar_42x_texture_backend<Chroma, LumaT, ChromaT, R, Shift> {};
            template<semi_planar_chroma Chroma, typename LumaT, typename ChromaT, bit_depth R, int Shift>
            struct semi_planar_42x_texture_backend {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void tex_load(const TexObjT* texPlanes, CellDataType& cell,
                                                                int32_t nCellX, int32_t nCellY, uint32_t imageWidth,
                                                                uint32_t imageHeight) {
                    semi_planar_42x_tex_load<Chroma, LumaT, ChromaT, R, Shift>(texPlanes[0], texPlanes[1], cell, nCellX,
                                                                               nCellY, imageWidth, imageHeight);
                }
                template<typename CellDataType>
                __forceinline__ __device__ static void surf_save(const SurfObjT* surfPlanes, const CellDataType& cell,
                                                                 int32_t nCellX, int32_t nCellY, uint32_t imageWidth,
                                                                 uint32_t imageHeight) {
                    semi_planar_42x_surf_save<Chroma, LumaT, ChromaT, R, Shift>(
                        surfPlanes[0], surfPlanes[1], cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

            // NV12: 4:2:0, 8-bit, luma=unsigned char, chroma=uchar2, no container shift.
            template<>
            struct texture_format_backend<packing_format::nv12>:
                semi_planar_42x_texture_backend<semi_planar_chroma::yuv_420, unsigned char, uchar2, bit_depth::bpp_8u,
                                              /*Shift=*/0> {
            };

            // P010: 4:2:0, 10-bit, luma=uint16, chroma=ushort2, value in the upper 10 bits (shift 6).
            template<>
            struct texture_format_backend<packing_format::p010>:
                semi_planar_42x_texture_backend<semi_planar_chroma::yuv_420, unsigned short, ushort2, bit_depth::bpp_10u,
                                              /*Shift=*/6> {
            };

            // NV16: 4:2:2, 8-bit, luma=unsigned char, chroma=uchar2, no container shift.
            template<>
            struct texture_format_backend<packing_format::nv16>:
                semi_planar_42x_texture_backend<semi_planar_chroma::yuv_422, unsigned char, uchar2, bit_depth::bpp_8u,
                                              /*Shift=*/0> {
            };

            // P216: 4:2:2, full 16-bit, luma=uint16, chroma=ushort2, no container shift.
            template<>
            struct texture_format_backend<packing_format::p216>:
                semi_planar_42x_texture_backend<semi_planar_chroma::yuv_422, unsigned short, ushort2, bit_depth::bpp_16u,
                                              /*Shift=*/0> {
            };

            // Stacked single-array backend: one handle, planes stacked in a single cuArray. Used by the
            // 4:2:0 formats (nv12/p010), the layout the video APIs (NVENC CUDAARRAY input) use.
            template<semi_planar_chroma Chroma, typename LumaT, typename ChromaT, bit_depth R, int Shift>
            struct semi_planar_42x_stacked_texture_backend {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void tex_load(const TexObjT* texPlanes, CellDataType& cell,
                                                                int32_t nCellX, int32_t nCellY, uint32_t imageWidth,
                                                                uint32_t imageHeight) {
                    // Here we pass texPlanes[0] twice because we are dealing with stacked format, which
                    // means that Y and UV planes are stored in a single texture
                    semi_planar_42x_tex_load<Chroma, LumaT, ChromaT, R, Shift, /*Stacked=*/true>(
                        texPlanes[0], texPlanes[0], cell, nCellX, nCellY, imageWidth, imageHeight);
                }
                template<typename CellDataType>
                __forceinline__ __device__ static void surf_save(const SurfObjT* surfPlanes, const CellDataType& cell,
                                                                 int32_t nCellX, int32_t nCellY, uint32_t imageWidth,
                                                                 uint32_t imageHeight) {
                    // Pass surfPlanes[0] twice: stacked format keeps Y and UV in a single surface.
                    semi_planar_42x_surf_save<Chroma, LumaT, ChromaT, R, Shift, /*Stacked=*/true>(
                        surfPlanes[0], surfPlanes[0], cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

            template<>
            struct texture_stacked_backend<packing_format::nv12>:
                semi_planar_42x_stacked_texture_backend<semi_planar_chroma::yuv_420, unsigned char, uchar2,
                                                      bit_depth::bpp_8u, /*Shift=*/0> {
            };

            template<>
            struct texture_stacked_backend<packing_format::p010>:
                semi_planar_42x_stacked_texture_backend<semi_planar_chroma::yuv_420, unsigned short, ushort2,
                                                      bit_depth::bpp_10u, /*Shift=*/6> {
            };

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_TEXTURE_SEMI_PLANAR_42X_HPP
