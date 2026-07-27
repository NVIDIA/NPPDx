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

#ifndef NPPDX_DETAIL_BACKEND_INGEST_EXGEST_YUV_10U_PACKED_FAMILY_HPP
#define NPPDX_DETAIL_BACKEND_INGEST_EXGEST_YUV_10U_PACKED_FAMILY_HPP

#include <cstdint>
#include "nppdx/detail/backend/constants.hpp"
#include "nppdx/detail/backend/formats_impl/format_backend.hpp"
#include "nppdx/detail/backend/formats_impl/clamp_utils.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {

            // YUV 10U Packed backend

// Fast path control (from NPP)
#ifndef USE_FAST_PATH
#    define USE_FAST_PATH 1
#endif

            // -------------------------------------------------------------------------
            // YUV 4:2:2 10-bit packed: UYVP (5 bytes per 2 pixels) and v210 (6 bytes per 2 pixels)
            // -------------------------------------------------------------------------
            #pragma pack(push, 1)
            template<bool isUYVP>
            struct YUV10UPair;
            template<>
            struct YUV10UPair<true> {
                uint32_t w0;
                uint8_t  b1;
            };
            template<>
            struct YUV10UPair<false> {
                uint32_t w0;
                uint16_t s1;
            };
#pragma pack(pop)

            template<bool isUYVP, typename CellDataType>
            __forceinline__ __device__ void load_yuv_10u_packed(const uint8_t** src, const size_t* stride,
                                                                CellDataType& cell, uint32_t nCellX, uint32_t nCellY) {
                float*         ch0        = cell.channel_ptr(0);
                float*         ch1        = cell.channel_ptr(1);
                float*         ch2        = cell.channel_ptr(2);
                constexpr int  cw         = CellDataType::cell_width;
                constexpr int  ch         = CellDataType::cell_height;
                constexpr int  pair_bytes = isUYVP ? 5 : 6;
                const uint8_t* plane      = src[0];
                const size_t   str        = stride[0];

#pragma unroll
                for (int row = 0; row < ch; ++row) {
                    const int      dataOff = row * cw;
                    const uint8_t* rowBase = plane + (nCellY * ch + row) * str + nCellX * (cw / 2) * pair_bytes;
#pragma unroll
                    for (int p = 0; p < cw / 2; ++p) {
                        const YUV10UPair<isUYVP>* pair = (const YUV10UPair<isUYVP>*)(rowBase + p * pair_bytes);
                        uint32_t                  w0   = pair->w0;
                        float                     u    = float(w0 & 0x3FF);
                        float                     y0   = float((w0 >> 10) & 0x3FF);
                        float                     v    = float((w0 >> 20) & 0x3FF);
                        float                     y1;
                        if constexpr (isUYVP)
                            y1 = float(((w0 >> 30) | (uint32_t(pair->b1) << 2)) & 0x3FF);
                        else
                            y1 = float(pair->s1 & 0x3FF);
                        ch0[dataOff + 2 * p]     = y0;
                        ch0[dataOff + 2 * p + 1] = y1;
                        ch1[dataOff + 2 * p] = ch1[dataOff + 2 * p + 1] = u;
                        ch2[dataOff + 2 * p] = ch2[dataOff + 2 * p + 1] = v;
                    }
                }
            }

            // -------------------------------------------------------------------------
            // YUV 4:2:2 10-bit packed quad: UYVP (10 bytes per 4 pixels) and v210 (12 bytes per 4 pixels)
            // Same bit layout as YUV10UPair: w0 = U[0:10] | Y0[10:20] | V[20:30] | Y1_lo[30:32]; then Y1_hi in b1/s1.
            // -------------------------------------------------------------------------
#pragma pack(push, 1)
            template<bool isUYVP>
            struct YUV10UQuad;
            template<>
            struct YUV10UQuad<true> {
                uint64_t d0; // bytes 0-7: pair0 (w0 0-3, b1 4) + pair1 w0 low 3 bytes (5-7)
                uint16_t s1; // bytes 8-9: pair1 w0 byte 8, pair1 b1 byte 9
            };
            template<>
            struct YUV10UQuad<false> {
                uint64_t d0; // bytes 0-7: pair0 full (w0 0-3, s1 4-5)
                uint32_t w1; // bytes 8-11: pair1 full (w0 8-9, s1 10-11 as 16-bit after w0)
            };
#pragma pack(pop)

            // Unaligned load/store for 10u packed quad path: rowBase can be misaligned (e.g. nCellX*10 for UYVP).
            __forceinline__ __device__ uint64_t load_uint64_unaligned(const uint8_t* p) {
                return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
                       ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) | ((uint64_t)p[6] << 48) |
                       ((uint64_t)p[7] << 56);
            }
            __forceinline__ __device__ uint32_t load_uint32_unaligned(const uint8_t* p) {
                return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
            }
            __forceinline__ __device__ uint16_t load_uint16_unaligned(const uint8_t* p) {
                return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
            }
            __forceinline__ __device__ void store_uint64_unaligned(uint8_t* p, uint64_t v) {
                p[0] = (uint8_t)(v);
                p[1] = (uint8_t)(v >> 8);
                p[2] = (uint8_t)(v >> 16);
                p[3] = (uint8_t)(v >> 24);
                p[4] = (uint8_t)(v >> 32);
                p[5] = (uint8_t)(v >> 40);
                p[6] = (uint8_t)(v >> 48);
                p[7] = (uint8_t)(v >> 56);
            }
            __forceinline__ __device__ void store_uint32_unaligned(uint8_t* p, uint32_t v) {
                p[0] = (uint8_t)(v);
                p[1] = (uint8_t)(v >> 8);
                p[2] = (uint8_t)(v >> 16);
                p[3] = (uint8_t)(v >> 24);
            }
            __forceinline__ __device__ void store_uint16_unaligned(uint8_t* p, uint16_t v) {
                p[0] = (uint8_t)(v);
                p[1] = (uint8_t)(v >> 8);
            }

            // Explicit specializations: isUYVP true and false, CellDataType = NPPCellData<4,2,3,float>.
            template<>
            __forceinline__ __device__ void load_yuv_10u_packed<true, cell_4x2_3ch_float>(
                const uint8_t** src, const size_t* stride, cell_4x2_3ch_float& cell, uint32_t nCellX, uint32_t nCellY) {
                float*         ch0   = cell.channel_ptr(0);
                float*         ch1   = cell.channel_ptr(1);
                float*         ch2   = cell.channel_ptr(2);
                const uint8_t* plane = src[0];
                const size_t   str   = stride[0];

#pragma unroll
                for (int row = 0; row < 2; ++row) {
                    const int      dataOff = row * 4;
                    const uint8_t* rowBase = plane + (nCellY * 2 + row) * str + nCellX * 2 * 5;

                    const uint64_t d0 = load_uint64_unaligned(rowBase);
                    const uint16_t s1 = load_uint16_unaligned(rowBase + 8);

                    const uint32_t w0_0 = (uint32_t)(d0);
                    const uint8_t  b1_0 = (uint8_t)(d0 >> 32);
                    const uint32_t w0_1 = (uint32_t)((d0 >> 40) | ((uint32_t)(s1 & 0xFFu) << 24));
                    const uint8_t  b1_1 = (uint8_t)(s1 >> 8);

                    ch0[dataOff + 0] = float((w0_0 >> 10) & 0x3FF);
                    ch0[dataOff + 1] = float(((w0_0 >> 30) | (uint32_t(b1_0) << 2)) & 0x3FF);
                    ch0[dataOff + 2] = float((w0_1 >> 10) & 0x3FF);
                    ch0[dataOff + 3] = float(((w0_1 >> 30) | (uint32_t(b1_1) << 2)) & 0x3FF);
                    ch1[dataOff + 0] = ch1[dataOff + 1] = float(w0_0 & 0x3FF);
                    ch1[dataOff + 2] = ch1[dataOff + 3] = float(w0_1 & 0x3FF);
                    ch2[dataOff + 0] = ch2[dataOff + 1] = float((w0_0 >> 20) & 0x3FF);
                    ch2[dataOff + 2] = ch2[dataOff + 3] = float((w0_1 >> 20) & 0x3FF);
                }
            }

            template<>
            __forceinline__ __device__ void load_yuv_10u_packed<false, cell_4x2_3ch_float>(
                const uint8_t** src, const size_t* stride, cell_4x2_3ch_float& cell, uint32_t nCellX, uint32_t nCellY) {
                float*         ch0   = cell.channel_ptr(0);
                float*         ch1   = cell.channel_ptr(1);
                float*         ch2   = cell.channel_ptr(2);
                const uint8_t* plane = src[0];
                const size_t   str   = stride[0];

#pragma unroll
                for (int row = 0; row < 2; ++row) {
                    const int      dataOff = row * 4;
                    const uint8_t* rowBase = plane + (nCellY * 2 + row) * str + nCellX * 2 * 6;

                    const uint64_t d0 = load_uint64_unaligned(rowBase);
                    const uint32_t w1 = load_uint32_unaligned(rowBase + 8);

                    const uint32_t w0_0 = (uint32_t)(d0);
                    const uint16_t s1_0 = (uint16_t)(d0 >> 32);
                    const uint32_t w0_1 = (uint32_t)((d0 >> 48) | ((uint64_t)(w1 & 0xFFFFu) << 16));
                    const uint16_t s1_1 = (uint16_t)(w1 >> 16);

                    ch0[dataOff + 0] = float((w0_0 >> 10) & 0x3FF);
                    ch0[dataOff + 1] = float(s1_0 & 0x3FF);
                    ch0[dataOff + 2] = float((w0_1 >> 10) & 0x3FF);
                    ch0[dataOff + 3] = float(s1_1 & 0x3FF);
                    ch1[dataOff + 0] = ch1[dataOff + 1] = float(w0_0 & 0x3FF);
                    ch1[dataOff + 2] = ch1[dataOff + 3] = float(w0_1 & 0x3FF);
                    ch2[dataOff + 0] = ch2[dataOff + 1] = float((w0_0 >> 20) & 0x3FF);
                    ch2[dataOff + 2] = ch2[dataOff + 3] = float((w0_1 >> 20) & 0x3FF);
                }
            }


            template<bool isUYVP, typename CellDataType>
            __forceinline__ __device__ void save_yuv_10u_packed(uint8_t** dst, const size_t* stride, bool clip,
                                                                float alpha, const CellDataType& cell, uint32_t nCellX,
                                                                uint32_t nCellY) {
                (void)alpha;
                const float*  ch0        = cell.channel_ptr(0);
                const float*  ch1        = cell.channel_ptr(1);
                const float*  ch2        = cell.channel_ptr(2);
                constexpr int cw         = CellDataType::cell_width;
                constexpr int ch         = CellDataType::cell_height;
                constexpr int pair_bytes = isUYVP ? 5 : 6;
                uint8_t*      plane      = dst[0];
                const size_t  str        = stride[0];

#pragma unroll
                for (int row = 0; row < ch; ++row) {
                    const int dataOff = row * cw;
                    uint8_t*  rowBase = plane + (nCellY * ch + row) * str + nCellX * (cw / 2) * pair_bytes;
#pragma unroll
                    for (int p = 0; p < cw / 2; ++p) {
                        float               y0   = ch0[dataOff + 2 * p];
                        float               y1   = ch0[dataOff + 2 * p + 1];
                        float               u    = (ch1[dataOff + 2 * p] + ch1[dataOff + 2 * p + 1]) * 0.5f;
                        float               v    = (ch2[dataOff + 2 * p] + ch2[dataOff + 2 * p + 1]) * 0.5f;
                        uint32_t            u32  = clip ? (uint32_t)fclampf<bit_depth::bpp_10u>(u) : (uint32_t)u;
                        uint32_t            y0_  = clip ? (uint32_t)fclampf<bit_depth::bpp_10u>(y0) : (uint32_t)y0;
                        uint32_t            v32  = clip ? (uint32_t)fclampf<bit_depth::bpp_10u>(v) : (uint32_t)v;
                        uint32_t            y1_  = clip ? (uint32_t)fclampf<bit_depth::bpp_10u>(y1) : (uint32_t)y1;
                        YUV10UPair<isUYVP>* pair = (YUV10UPair<isUYVP>*)(rowBase + p * pair_bytes);
                        pair->w0                 = (u32 & 0x3FF) | ((y0_ & 0x3FF) << 10) | ((v32 & 0x3FF) << 20);
                        if constexpr (isUYVP)
                            pair->w0 |= ((y1_ & 0x3) << 30), pair->b1 = (y1_ >> 2) & 0xFF;
                        else
                            pair->s1 = (uint16_t)(y1_ & 0x3FF);
                    }
                }
            }

            // Explicit specializations: save_yuv_10u_packed for isUYVP true/false, NPPCellData<4,2,3,float> (one quad per row).
            // clip is tested once so the whole SM takes one branch.
            template<>
            __forceinline__ __device__ void save_yuv_10u_packed<true, cell_4x2_3ch_float>(
                uint8_t** dst, const size_t* stride, bool clip, float alpha, const cell_4x2_3ch_float& cell,
                uint32_t nCellX, uint32_t nCellY) {
                (void)alpha;
                const float* ch0   = cell.channel_ptr(0);
                const float* ch1   = cell.channel_ptr(1);
                const float* ch2   = cell.channel_ptr(2);
                uint8_t*     plane = dst[0];
                const size_t str   = stride[0];

                if (clip) {
#pragma unroll
                    for (int row = 0; row < 2; ++row) {
                        const int dataOff = row * 4;
                        uint8_t*  rowBase = plane + (nCellY * 2 + row) * str + nCellX * 2 * 5;

                        const float    u0    = (ch1[dataOff + 0] + ch1[dataOff + 1]) * 0.5f;
                        const float    u1    = (ch1[dataOff + 2] + ch1[dataOff + 3]) * 0.5f;
                        const float    v0    = (ch2[dataOff + 0] + ch2[dataOff + 1]) * 0.5f;
                        const float    v1    = (ch2[dataOff + 2] + ch2[dataOff + 3]) * 0.5f;
                        const uint32_t u32_0 = (uint32_t)fclampf<bit_depth::bpp_10u>(u0);
                        const uint32_t u32_1 = (uint32_t)fclampf<bit_depth::bpp_10u>(u1);
                        const uint32_t v32_0 = (uint32_t)fclampf<bit_depth::bpp_10u>(v0);
                        const uint32_t v32_1 = (uint32_t)fclampf<bit_depth::bpp_10u>(v1);
                        const uint32_t y0_0  = (uint32_t)fclampf<bit_depth::bpp_10u>(ch0[dataOff + 0]);
                        const uint32_t y1_0  = (uint32_t)fclampf<bit_depth::bpp_10u>(ch0[dataOff + 1]);
                        const uint32_t y0_1  = (uint32_t)fclampf<bit_depth::bpp_10u>(ch0[dataOff + 2]);
                        const uint32_t y1_1  = (uint32_t)fclampf<bit_depth::bpp_10u>(ch0[dataOff + 3]);

                        const uint32_t w0_0 =
                            (u32_0 & 0x3FF) | ((y0_0 & 0x3FF) << 10) | ((v32_0 & 0x3FF) << 20) | ((y1_0 & 0x3) << 30);
                        const uint8_t  b1_0 = (y1_0 >> 2) & 0xFF;
                        const uint32_t w0_1 =
                            (u32_1 & 0x3FF) | ((y0_1 & 0x3FF) << 10) | ((v32_1 & 0x3FF) << 20) | ((y1_1 & 0x3) << 30);
                        const uint8_t b1_1 = (y1_1 >> 2) & 0xFF;

                        const uint64_t d0 =
                            (uint64_t)w0_0 | ((uint64_t)b1_0 << 32) | ((uint64_t)(w0_1 & 0xFFFFFFu) << 40);
                        const uint16_t s1 = (uint16_t)((w0_1 >> 24) | (b1_1 << 8));
                        store_uint64_unaligned(rowBase, d0);
                        store_uint16_unaligned(rowBase + 8, s1);
                    }
                } else {
#pragma unroll
                    for (int row = 0; row < 2; ++row) {
                        const int dataOff = row * 4;
                        uint8_t*  rowBase = plane + (nCellY * 2 + row) * str + nCellX * 2 * 5;

                        const float    u0    = (ch1[dataOff + 0] + ch1[dataOff + 1]) * 0.5f;
                        const float    u1    = (ch1[dataOff + 2] + ch1[dataOff + 3]) * 0.5f;
                        const float    v0    = (ch2[dataOff + 0] + ch2[dataOff + 1]) * 0.5f;
                        const float    v1    = (ch2[dataOff + 2] + ch2[dataOff + 3]) * 0.5f;
                        const uint32_t u32_0 = (uint32_t)u0;
                        const uint32_t u32_1 = (uint32_t)u1;
                        const uint32_t v32_0 = (uint32_t)v0;
                        const uint32_t v32_1 = (uint32_t)v1;
                        const uint32_t y0_0  = (uint32_t)ch0[dataOff + 0];
                        const uint32_t y1_0  = (uint32_t)ch0[dataOff + 1];
                        const uint32_t y0_1  = (uint32_t)ch0[dataOff + 2];
                        const uint32_t y1_1  = (uint32_t)ch0[dataOff + 3];

                        const uint32_t w0_0 =
                            (u32_0 & 0x3FF) | ((y0_0 & 0x3FF) << 10) | ((v32_0 & 0x3FF) << 20) | ((y1_0 & 0x3) << 30);
                        const uint8_t  b1_0 = (y1_0 >> 2) & 0xFF;
                        const uint32_t w0_1 =
                            (u32_1 & 0x3FF) | ((y0_1 & 0x3FF) << 10) | ((v32_1 & 0x3FF) << 20) | ((y1_1 & 0x3) << 30);
                        const uint8_t b1_1 = (y1_1 >> 2) & 0xFF;

                        const uint64_t d0 =
                            (uint64_t)w0_0 | ((uint64_t)b1_0 << 32) | ((uint64_t)(w0_1 & 0xFFFFFFu) << 40);
                        const uint16_t s1 = (uint16_t)((w0_1 >> 24) | (b1_1 << 8));
                        store_uint64_unaligned(rowBase, d0);
                        store_uint16_unaligned(rowBase + 8, s1);
                    }
                }
            }

            template<>
            __forceinline__ __device__ void save_yuv_10u_packed<false, cell_4x2_3ch_float>(
                uint8_t** dst, const size_t* stride, bool clip, float alpha, const cell_4x2_3ch_float& cell,
                uint32_t nCellX, uint32_t nCellY) {
                (void)alpha;
                const float* ch0   = cell.channel_ptr(0);
                const float* ch1   = cell.channel_ptr(1);
                const float* ch2   = cell.channel_ptr(2);
                uint8_t*     plane = dst[0];
                const size_t str   = stride[0];

                if (clip) {
#pragma unroll
                    for (int row = 0; row < 2; ++row) {
                        const int dataOff = row * 4;
                        uint8_t*  rowBase = plane + (nCellY * 2 + row) * str + nCellX * 2 * 6;

                        const float    u0    = (ch1[dataOff + 0] + ch1[dataOff + 1]) * 0.5f;
                        const float    u1    = (ch1[dataOff + 2] + ch1[dataOff + 3]) * 0.5f;
                        const float    v0    = (ch2[dataOff + 0] + ch2[dataOff + 1]) * 0.5f;
                        const float    v1    = (ch2[dataOff + 2] + ch2[dataOff + 3]) * 0.5f;
                        const uint32_t u32_0 = (uint32_t)fclampf<bit_depth::bpp_10u>(u0);
                        const uint32_t u32_1 = (uint32_t)fclampf<bit_depth::bpp_10u>(u1);
                        const uint32_t v32_0 = (uint32_t)fclampf<bit_depth::bpp_10u>(v0);
                        const uint32_t v32_1 = (uint32_t)fclampf<bit_depth::bpp_10u>(v1);
                        const uint32_t y0_0  = (uint32_t)fclampf<bit_depth::bpp_10u>(ch0[dataOff + 0]);
                        const uint32_t y1_0  = (uint32_t)fclampf<bit_depth::bpp_10u>(ch0[dataOff + 1]);
                        const uint32_t y0_1  = (uint32_t)fclampf<bit_depth::bpp_10u>(ch0[dataOff + 2]);
                        const uint32_t y1_1  = (uint32_t)fclampf<bit_depth::bpp_10u>(ch0[dataOff + 3]);

                        const uint32_t w0_0 = (u32_0 & 0x3FF) | ((y0_0 & 0x3FF) << 10) | ((v32_0 & 0x3FF) << 20);
                        const uint16_t s1_0 = (uint16_t)(y1_0 & 0x3FF);
                        const uint32_t w0_1 = (u32_1 & 0x3FF) | ((y0_1 & 0x3FF) << 10) | ((v32_1 & 0x3FF) << 20);
                        const uint16_t s1_1 = (uint16_t)(y1_1 & 0x3FF);

                        const uint64_t d0 =
                            (uint64_t)w0_0 | ((uint64_t)s1_0 << 32) | ((uint64_t)(w0_1 & 0xFFFFu) << 48);
                        const uint32_t w1 = (w0_1 >> 16) | ((uint32_t)s1_1 << 16);
                        store_uint64_unaligned(rowBase, d0);
                        store_uint32_unaligned(rowBase + 8, w1);
                    }
                } else {
#pragma unroll
                    for (int row = 0; row < 2; ++row) {
                        const int dataOff = row * 4;
                        uint8_t*  rowBase = plane + (nCellY * 2 + row) * str + nCellX * 2 * 6;

                        const float    u0    = (ch1[dataOff + 0] + ch1[dataOff + 1]) * 0.5f;
                        const float    u1    = (ch1[dataOff + 2] + ch1[dataOff + 3]) * 0.5f;
                        const float    v0    = (ch2[dataOff + 0] + ch2[dataOff + 1]) * 0.5f;
                        const float    v1    = (ch2[dataOff + 2] + ch2[dataOff + 3]) * 0.5f;
                        const uint32_t u32_0 = (uint32_t)u0;
                        const uint32_t u32_1 = (uint32_t)u1;
                        const uint32_t v32_0 = (uint32_t)v0;
                        const uint32_t v32_1 = (uint32_t)v1;
                        const uint32_t y0_0  = (uint32_t)ch0[dataOff + 0];
                        const uint32_t y1_0  = (uint32_t)ch0[dataOff + 1];
                        const uint32_t y0_1  = (uint32_t)ch0[dataOff + 2];
                        const uint32_t y1_1  = (uint32_t)ch0[dataOff + 3];

                        const uint32_t w0_0 = (u32_0 & 0x3FF) | ((y0_0 & 0x3FF) << 10) | ((v32_0 & 0x3FF) << 20);
                        const uint16_t s1_0 = (uint16_t)(y1_0 & 0x3FF);
                        const uint32_t w0_1 = (u32_1 & 0x3FF) | ((y0_1 & 0x3FF) << 10) | ((v32_1 & 0x3FF) << 20);
                        const uint16_t s1_1 = (uint16_t)(y1_1 & 0x3FF);

                        const uint64_t d0 =
                            (uint64_t)w0_0 | ((uint64_t)s1_0 << 32) | ((uint64_t)(w0_1 & 0xFFFFu) << 48);
                        const uint32_t w1 = (w0_1 >> 16) | ((uint32_t)s1_1 << 16);
                        store_uint64_unaligned(rowBase, d0);
                        store_uint32_unaligned(rowBase + 8, w1);
                    }
                }
            }

            template<bool isUYVP, typename CellDataType>
            __forceinline__ __device__ void safe_load_yuv_10u_packed(const uint8_t** src, const size_t* stride,
                                                                     CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                     int32_t imageWidth, int32_t imageHeight) {
                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;
                constexpr int     pair_bytes  = isUYVP ? 5 : 6;
                const int32_t     basePixelX  = nCellX * cell_width;
                const int32_t     basePixelY  = nCellY * cell_height;
#if USE_FAST_PATH
                if (basePixelX >= 0 && basePixelX + cell_width <= imageWidth && basePixelY >= 0 &&
                    basePixelY + cell_height <= imageHeight) {
                    load_yuv_10u_packed<isUYVP, CellDataType>(src, stride, cell, nCellX, nCellY);
                    return;
                }
#endif
                float*         ch0         = cell.channel_ptr(0);
                float*         ch1         = cell.channel_ptr(1);
                float*         ch2         = cell.channel_ptr(2);
                const uint8_t* plane       = src[0];
                const size_t   str         = stride[0];
                const bool     needLeft    = (basePixelX < 0);
                const int32_t  lastPairIdx = (imageWidth / 2) - 1;

#pragma unroll
                for (int32_t row = 0; row < cell_height; ++row) {
                    const int32_t  pixelY  = basePixelY + row;
                    const int32_t  clampY  = nppdx_max(0, nppdx_min(pixelY, imageHeight - 1));
                    const int32_t  dataOff = row * cell_width;
                    const uint8_t* rowPtr  = plane + (size_t)clampY * str;

                    float edgeY0, edgeY1, edgeU, edgeV;
                    if (needLeft) {
                        const YUV10UPair<isUYVP>* p0 = (const YUV10UPair<isUYVP>*)rowPtr;
                        edgeU                        = float(p0->w0 & 0x3FF);
                        edgeY0                       = float((p0->w0 >> 10) & 0x3FF);
                        edgeV                        = float((p0->w0 >> 20) & 0x3FF);
                        if constexpr (isUYVP)
                            edgeY1 = float(((p0->w0 >> 30) | (uint32_t(p0->b1) << 2)) & 0x3FF);
                        else
                            edgeY1 = float(p0->s1 & 0x3FF);
                    } else {
                        const YUV10UPair<isUYVP>* p0 =
                            (const YUV10UPair<isUYVP>*)(rowPtr + (size_t)lastPairIdx * pair_bytes);
                        edgeU  = float(p0->w0 & 0x3FF);
                        edgeY0 = float((p0->w0 >> 10) & 0x3FF);
                        edgeV  = float((p0->w0 >> 20) & 0x3FF);
                        if constexpr (isUYVP)
                            edgeY1 = float(((p0->w0 >> 30) | (uint32_t(p0->b1) << 2)) & 0x3FF);
                        else
                            edgeY1 = float(p0->s1 & 0x3FF);
                    }

#pragma unroll
                    for (int32_t i = 0; i < cell_width; ++i) {
                        const int32_t imgX = basePixelX + i;
                        if (imgX >= 0 && imgX < imageWidth) {
                            const int32_t             pairIdx = imgX / 2;
                            const YUV10UPair<isUYVP>* pair =
                                (const YUV10UPair<isUYVP>*)(rowPtr + (size_t)pairIdx * pair_bytes);
                            uint32_t w0      = pair->w0;
                            ch1[dataOff + i] = float(w0 & 0x3FF);
                            ch0[dataOff + i] = float((w0 >> 10) & 0x3FF);
                            ch2[dataOff + i] = float((w0 >> 20) & 0x3FF);
                            if ((imgX & 1) != 0) {
                                if constexpr (isUYVP)
                                    ch0[dataOff + i] = float(((w0 >> 30) | (uint32_t(pair->b1) << 2)) & 0x3FF);
                                else
                                    ch0[dataOff + i] = float(pair->s1 & 0x3FF);
                            }
                        } else {
                            ch0[dataOff + i] = needLeft ? edgeY0 : edgeY1;
                            ch1[dataOff + i] = edgeU;
                            ch2[dataOff + i] = edgeV;
                        }
                    }
                }
            }

            template<bool isUYVP, typename CellDataType>
            __forceinline__ __device__ void safe_save_yuv_10u_packed(uint8_t** dst, const size_t* stride, bool clip,
                                                                     float alpha, const CellDataType& cell,
                                                                     int32_t nCellX, int32_t nCellY, int32_t imageWidth,
                                                                     int32_t imageHeight) {
                (void)alpha;
                const float*  ch0        = cell.channel_ptr(0);
                const float*  ch1        = cell.channel_ptr(1);
                const float*  ch2        = cell.channel_ptr(2);
                constexpr int cw         = CellDataType::cell_width;
                constexpr int ch         = CellDataType::cell_height;
                constexpr int pair_bytes = isUYVP ? 5 : 6;
                const int32_t basePixelX = nCellX * cw;
                const int32_t basePixelY = nCellY * ch;
#if USE_FAST_PATH
                if (basePixelX >= 0 && basePixelX + cw <= imageWidth && basePixelY >= 0 &&
                    basePixelY + ch <= imageHeight) {
                    // Fast save uses uint32_t cell indices; here nCellX/Y are non-negative on this branch.
                    save_yuv_10u_packed<isUYVP, CellDataType>(
                        dst, stride, clip, alpha, cell, static_cast<uint32_t>(nCellX), static_cast<uint32_t>(nCellY));
                    return;
                }
#endif
                const int32_t srcStart = nppdx_max(0, -basePixelX);
                const int32_t srcEnd   = nppdx_min(cw, imageWidth - basePixelX);
                if (srcStart >= srcEnd)
                    return;
                const int32_t firstImgX = basePixelX + srcStart;
                uint8_t*      plane     = dst[0];
                const size_t  str       = stride[0];

#pragma unroll
                for (int32_t row = 0; row < ch; ++row) {
                    const int32_t pixelY = basePixelY + row;
                    if (pixelY < 0 || pixelY >= imageHeight)
                        continue;
                    const int32_t dataOff = row * cw;
                    uint8_t*      rowDst  = plane + (size_t)pixelY * str;

                    for (int32_t i = srcStart; i < srcEnd; i += 2) {
                        float y0 = ch0[dataOff + i];
                        float y1 = (i + 1 < srcEnd) ? ch0[dataOff + i + 1] : y0;
                        float u =
                            (ch1[dataOff + i] + (i + 1 < srcEnd ? ch1[dataOff + i + 1] : ch1[dataOff + i])) * 0.5f;
                        float v =
                            (ch2[dataOff + i] + (i + 1 < srcEnd ? ch2[dataOff + i + 1] : ch2[dataOff + i])) * 0.5f;
                        uint32_t            u32     = clip ? (uint32_t)fclampf<bit_depth::bpp_10u>(u) : (uint32_t)u;
                        uint32_t            y0_     = clip ? (uint32_t)fclampf<bit_depth::bpp_10u>(y0) : (uint32_t)y0;
                        uint32_t            v32     = clip ? (uint32_t)fclampf<bit_depth::bpp_10u>(v) : (uint32_t)v;
                        uint32_t            y1_     = clip ? (uint32_t)fclampf<bit_depth::bpp_10u>(y1) : (uint32_t)y1;
                        const int32_t       pairIdx = (firstImgX + (i - srcStart)) / 2;
                        YUV10UPair<isUYVP>* pair    = (YUV10UPair<isUYVP>*)(rowDst + (size_t)pairIdx * pair_bytes);
                        pair->w0                    = (u32 & 0x3FF) | ((y0_ & 0x3FF) << 10) | ((v32 & 0x3FF) << 20);
                        if constexpr (isUYVP)
                            pair->w0 |= ((y1_ & 0x3) << 30), pair->b1 = (y1_ >> 2) & 0xFF;
                        else
                            pair->s1 = (uint16_t)(y1_ & 0x3FF);
                    }
                }
            }            

            // YUV 10U Packed formats

            template<>
            struct format_backend<packing_format::uyvp> {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                 int32_t imageWidth, int32_t imageHeight) {
                    safe_load_yuv_10u_packed<true, CellDataType>(src, stride, cell, nCellX, nCellY, imageWidth,
                                                                 imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell, int32_t nCellX,
                                                                 int32_t nCellY, uint32_t imageWidth,
                                                                 uint32_t imageHeight) {
                    safe_save_yuv_10u_packed<true, CellDataType>(dst, stride, clip, alpha, cell, nCellX, nCellY,
                                                                 (int32_t)imageWidth, (int32_t)imageHeight);
                }
            };

        } // namespace backend
    } // namespace detail
} // namespace nppdx
#endif // NPPDX_DETAIL_BACKEND_INGEST_EXGEST_YUV_10U_PACKED_FAMILY_HPP
