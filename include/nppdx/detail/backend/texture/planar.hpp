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

#ifndef NPPDX_DETAIL_BACKEND_TEXTURE_PLANAR_HPP
#define NPPDX_DETAIL_BACKEND_TEXTURE_PLANAR_HPP

#include "nppdx/detail/backend/texture/common.hpp"
#include "nppdx/utils.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {

            // =================================================================
            // Fully planar 4:4:4 — three full-resolution single-channel planes.
            // Pch0/Pch1/Pch2 give the physical plane index feeding cell channels
            // 0/1/2, so the channel order matches the rest of the pipeline (RGB:
            // channel 0=R,1=G,2=B; YUV: 0=Y,1=U,2=V). rgbp/yuv444p are identity
            // {0,1,2}; bgrp stores B,G,R so it maps {2,1,0}. Mask keeps the value
            // in its valid bits (8-bit: 0xFF; yuv444p10: 10-bit in the LSBs, 0x3FF).
            // Mirrors PlanarLoader/PlanarSaver in formats_impl/planar_family.hpp.
            // =================================================================

            template<typename SampleT, uint32_t Mask, int Pch0, int Pch1, int Pch2, typename CellDataType>
            __forceinline__ __device__ void planar_tex_load(
                const TexObjT* tex, CellDataType& cell,
                int32_t nCellX, int32_t nCellY, uint32_t imageWidth, uint32_t imageHeight)
            {
                constexpr int cell_width  = int(CellDataType::cell_cfg.size.x);
                constexpr int cell_height = int(CellDataType::cell_cfg.size.y);

                float* ch0 = cell.channel_ptr(0);
                float* ch1 = cell.channel_ptr(1);
                float* ch2 = cell.channel_ptr(2);

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;
                const int32_t maxX       = int32_t(imageWidth) - 1;
                const int32_t maxY       = int32_t(imageHeight) - 1;

#pragma unroll
                for (int row = 0; row < cell_height; ++row) {
                    const int32_t y = nppdx::clamp_value(basePixelY + row, 0, maxY);
#pragma unroll
                    for (int col = 0; col < cell_width; ++col) {
                        const int32_t x   = nppdx::clamp_value(basePixelX + col, 0, maxX);
                        const int     idx = row * cell_width + col;
                        ch0[idx] = float(static_cast<uint32_t>(tex2D<SampleT>(tex[Pch0], x, y)) & Mask);
                        ch1[idx] = float(static_cast<uint32_t>(tex2D<SampleT>(tex[Pch1], x, y)) & Mask);
                        ch2[idx] = float(static_cast<uint32_t>(tex2D<SampleT>(tex[Pch2], x, y)) & Mask);
                    }
                }
            }

            // No bounds check: cudaBoundaryModeZero drops out-of-image writes (arrays == plane sizes).
            // No mask on store: fclampf<R> already bounds the value to [0, (1<<bits)-1] and each channel
            // is written to its own single-channel plane, so there are no stray high bits to strip.
            template<typename SampleT, bit_depth R, int Pch0, int Pch1, int Pch2, typename CellDataType>
            __forceinline__ __device__ void planar_surf_save(
                const SurfObjT* surf, const CellDataType& cell,
                int32_t nCellX, int32_t nCellY, uint32_t imageWidth, uint32_t imageHeight)
            {
                constexpr int cell_width  = int(CellDataType::cell_cfg.size.x);
                constexpr int cell_height = int(CellDataType::cell_cfg.size.y);

                const float* ch0 = cell.channel_ptr(0);
                const float* ch1 = cell.channel_ptr(1);
                const float* ch2 = cell.channel_ptr(2);

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;
                constexpr int elem_bytes = int(sizeof(SampleT));

#pragma unroll
                for (int row = 0; row < cell_height; ++row) {
                    const int32_t py = basePixelY + row;
#pragma unroll
                    for (int col = 0; col < cell_width; ++col) {
                        const int     idx = row * cell_width + col;
                        const int32_t px  = (basePixelX + col) * elem_bytes;
                        const SampleT v0  = static_cast<SampleT>(fclampf<R>(ch0[idx]));
                        const SampleT v1  = static_cast<SampleT>(fclampf<R>(ch1[idx]));
                        const SampleT v2  = static_cast<SampleT>(fclampf<R>(ch2[idx]));
                        surf2Dwrite(v0, surf[Pch0], px, py, cudaBoundaryModeZero);
                        surf2Dwrite(v1, surf[Pch1], px, py, cudaBoundaryModeZero);
                        surf2Dwrite(v2, surf[Pch2], px, py, cudaBoundaryModeZero);
                    }
                }
            }

            // Reusable backend for the fully-planar family. A new planar format only needs a one-line
            //   template<> struct texture_format_backend<Format>
            //       : planar_texture_backend<SampleT, R, Mask, Pch0, Pch1, Pch2> {};
            template<typename SampleT, bit_depth R, uint32_t Mask, int Pch0, int Pch1, int Pch2>
            struct planar_texture_backend {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void tex_load(const TexObjT* texPlanes,
                                                                CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                uint32_t imageWidth, uint32_t imageHeight) {
                    planar_tex_load<SampleT, Mask, Pch0, Pch1, Pch2>(texPlanes, cell, nCellX, nCellY, imageWidth,
                                                                     imageHeight);
                }
                template<typename CellDataType>
                __forceinline__ __device__ static void surf_save(const SurfObjT* surfPlanes,
                                                                 const CellDataType& cell, int32_t nCellX,
                                                                 int32_t nCellY, uint32_t imageWidth,
                                                                 uint32_t imageHeight) {
                    planar_surf_save<SampleT, R, Pch0, Pch1, Pch2>(surfPlanes, cell, nCellX, nCellY, imageWidth,
                                                                   imageHeight);
                }
            };

            // YUV 4:4:4 8-bit: planes Y,U,V -> channels 0,1,2.
            template<>
            struct texture_format_backend<packing_format::yuv444p>
                : planar_texture_backend<unsigned char, bit_depth::bpp_8u, 0xFFu, 0, 1, 2> {};

            // YUV 4:4:4 10-bit in the LSBs of uint16 (yuv444p10le): mask 0x3FF, planes Y,U,V -> 0,1,2.
            template<>
            struct texture_format_backend<packing_format::yuv444p10>
                : planar_texture_backend<unsigned short, bit_depth::bpp_10u, 0x3FFu, 0, 1, 2> {};

            // Planar RGB 8-bit: planes R,G,B -> channels 0,1,2 (identity).
            template<>
            struct texture_format_backend<packing_format::rgbp>
                : planar_texture_backend<unsigned char, bit_depth::bpp_8u, 0xFFu, 0, 1, 2> {};

            // Planar BGR 8-bit: planes B,G,R -> channels R,G,B = 0,1,2, so map {2,1,0}.
            template<>
            struct texture_format_backend<packing_format::bgrp>
                : planar_texture_backend<unsigned char, bit_depth::bpp_8u, 0xFFu, 2, 1, 0> {};

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_TEXTURE_PLANAR_HPP
