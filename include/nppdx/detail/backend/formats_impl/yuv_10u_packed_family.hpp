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
            // UYVP is the SMPTE ST 2110-20:2022 pgroup for 10-bit YCbCr 4:2:2.
            //
            // ST 2110-20 6.2.4 Table 2: sampling YCbCr-4:2:2, depth 10, pgroup size 5
            // octets*, pgroup coverage 2 pixels, sample order C'B, Y0', C'R, Y1' - which
            // this code calls U, Y0, V, Y1. Four 10-bit samples fill 40 bits exactly, so
            // pgroups are contiguous with no padding and only every other one starts on
            // an even address.
            //
            // * `Octets` refers to 8-bit bytes in the specification.
            //
            // ST 2110-20 6.1.1 requires each multi-byte value be transmitted most
            // significant byte first, with its most significant bits occupying the
            // lowest-numbered bit positions. Every sample here spans two bytes, so:
            //
            //   byte 0:  C'B[9:2]
            //   byte 1:  C'B[1:0] Y0'[9:4]
            //   byte 2:  Y0'[3:0] C'R[9:6]
            //   byte 3:  C'R[5:0] Y1'[9:8]
            //   byte 4:  Y1'[7:0]
            //
            //   C'B = (b0 << 2) | (b1 >> 6)
            //   Y0' = ((b1 & 0x3F) << 4) | (b2 >> 4)
            //   C'R = ((b2 & 0x0F) << 6) | (b3 >> 2)
            //   Y1' = ((b3 & 0x03) << 8) | b4
            //
            // and the inverse:
            //
            //   byte 0 = C'B >> 2
            //   byte 1 = ((C'B & 0x03) << 6) | (Y0' >> 4)
            //   byte 2 = ((Y0' & 0x0F) << 4) | (C'R >> 6)
            //   byte 3 = ((C'R & 0x3F) << 2) | (Y1' >> 8)
            //   byte 4 = Y1' & 0xFF
            //
            // Every 10-bit sample straddles a byte boundary, so no color component lines
            // up with a little-endian uint32. Permuting the first four bytes restores a
            // contiguous word, simplifying extraction to a single unaligned 32-bit read
            // plus one byte: {w0, b1}
            //
            //   w0p = __byte_perm(w0, 0, 0x0123)   // reverse byte order of w0
            //   C'B = w0p >> 22                    // bits 31..22, already 10 bits
            //   Y0' = (w0p >> 12) & 0x3FF          // bits 21..12
            //   C'R = (w0p >>  2) & 0x3FF          // bits 11..2
            //   Y1' = ((w0p & 0x03) << 8) | b1     // bits 1..0, then byte 4
            //
            // ODD WIDTH DEVIATION FROM SPECIFICATION: the trailing pixel is neither read
            // nor written, given the mismatch between the SMPTE UYVP definition and how
            // cudaMallocPitch works. We cannot guarantee a full pgroup (2 pixels) exists
            // for the odd pixel column, nor that the last row has one even when the
            // stride is large enough.
            //
            // Caller-provided memory gives us a pointer, a row stride, a width and a
            // height, which cannot tell us how many bytes follow the last pixel of the
            // last row. A caller who sized rows as ceil(width * 2.5) gets 88 bytes for
            // width 35, where 18 whole pgroups need 90. Even a conforming stride, big
            // enough to hold that extra pgroup, says nothing about the final row, which
            // may be truncated to the row's data length rather than the full pitch.
            //
            // Effectively an odd-width image is processed as though it were one pixel
            // narrower. Width 1 is degenerate: there is no complete pair at all, so
            // ingest zeroes the cell rather than decode bytes that may not exist, and
            // exgest writes nothing.
            //
            // Odd widths do not occur in practice for this format - this code path is a
            // safety net.
            // -------------------------------------------------------------------------
#pragma pack(push, 1)
            struct YUV10UPair {
                uint32_t w0; // bytes 0-3, little-endian; reverse byte order before extracting
                uint8_t  b1; // byte 4 = Y1'[7:0]
            };
#pragma pack(pop)

            template<typename CellDataType>
            __forceinline__ __device__ void load_yuv_10u_packed(const uint8_t** src, const size_t* stride,
                                                                CellDataType& cell, uint32_t nCellX, uint32_t nCellY) {
                float*         ch0        = cell.channel_ptr(0);
                float*         ch1        = cell.channel_ptr(1);
                float*         ch2        = cell.channel_ptr(2);
                constexpr int  cw         = CellDataType::cell_width;
                constexpr int  ch         = CellDataType::cell_height;
                constexpr int  pair_bytes = 5;
                const uint8_t* plane      = src[0];
                const size_t   str        = stride[0];

#pragma unroll
                for (int row = 0; row < ch; ++row) {
                    const int      dataOff = row * cw;
                    const uint8_t* rowBase = plane + (nCellY * ch + row) * str + nCellX * (cw / 2) * pair_bytes;
#pragma unroll
                    for (int p = 0; p < cw / 2; ++p) {
                        const YUV10UPair* pair = (const YUV10UPair*)(rowBase + p * pair_bytes);
                        const uint32_t    w0p   = __byte_perm(pair->w0, 0, 0x0123);
                        float             u    = float(w0p >> 22);
                        float             y0   = float((w0p >> 12) & 0x3FF);
                        float             v    = float((w0p >> 2) & 0x3FF);
                        float             y1   = float(((w0p & 0x03) << 8) | pair->b1);
                        ch0[dataOff + 2 * p]     = y0;
                        ch0[dataOff + 2 * p + 1] = y1;
                        ch1[dataOff + 2 * p] = ch1[dataOff + 2 * p + 1] = u;
                        ch2[dataOff + 2 * p] = ch2[dataOff + 2 * p + 1] = v;
                    }
                }
            }

            // Unaligned load/store for the 4-pixel path: a 4-pixel cell spans 10 bytes,
            // so rowBase is only 2-byte aligned at best and a plain uint64_t read faults.
            __forceinline__ __device__ uint64_t load_uint64_unaligned(const uint8_t* p) {
                return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
                       ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) | ((uint64_t)p[6] << 48) |
                       ((uint64_t)p[7] << 56);
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
            __forceinline__ __device__ void store_uint16_unaligned(uint8_t* p, uint16_t v) {
                p[0] = (uint8_t)(v);
                p[1] = (uint8_t)(v >> 8);
            }

            // Writes the 10 bytes of two pgroups. Five 16-bit stores beat ten 8-bit ones
            // ~2.5x, because a partial sector write costs a read-modify-write and stores do
            // not merge across instructions. That needs p to be even; the caller decides,
            // since the test is on the plane and the stride and so is uniform across the
            // block. aligned2 false falls back to the byte stores.
            __forceinline__ __device__ void store_two_pgroups(uint8_t* p, uint32_t w0_0, uint8_t b1_0,
                                                              uint32_t w0_1, uint8_t b1_1, bool aligned2) {
                if (aligned2) {
                    // Bytes 0-9 are w0_0[0..3], b1_0, w0_1[0..3], b1_1, so o[2] and o[4]
                    // each straddle a pgroup boundary.
                    uint16_t* o = (uint16_t*)p;
                    o[0]        = (uint16_t)(w0_0 & 0xFFFFu);
                    o[1]        = (uint16_t)(w0_0 >> 16);
                    o[2]        = (uint16_t)(b1_0 | ((w0_1 & 0xFFu) << 8));
                    o[3]        = (uint16_t)((w0_1 >> 8) & 0xFFFFu);
                    o[4]        = (uint16_t)(((w0_1 >> 24) & 0xFFu) | (b1_1 << 8));
                } else {
                    const uint64_t d0 =
                        (uint64_t)w0_0 | ((uint64_t)b1_0 << 32) | ((uint64_t)(w0_1 & 0xFFFFFFu) << 40);
                    const uint16_t s1 = (uint16_t)((w0_1 >> 24) | (b1_1 << 8));
                    store_uint64_unaligned(p, d0);
                    store_uint16_unaligned(p + 8, s1);
                }
            }

            // Explicit specialization for CellDataType = NPPCellData<4,2,3,float>: one
            // 10-byte quad per row, loaded as 8+2 bytes rather than two 5-byte groups.
            template<>
            __forceinline__ __device__ void load_yuv_10u_packed<cell_4x2_3ch_float>(
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

                    // Two pgroups across 10 bytes: pgroup 0 is bytes 0-4, pgroup 1 is
                    // bytes 5-9, the last two of which arrive in s1.
                    const uint32_t w0_0 = (uint32_t)(d0);
                    const uint8_t  b1_0 = (uint8_t)(d0 >> 32);
                    const uint32_t w0_1 = (uint32_t)((d0 >> 40) | ((uint32_t)(s1 & 0xFFu) << 24));
                    const uint8_t  b1_1 = (uint8_t)(s1 >> 8);

                    const uint32_t w0p_0 = __byte_perm(w0_0, 0, 0x0123);
                    const uint32_t w0p_1 = __byte_perm(w0_1, 0, 0x0123);

                    ch0[dataOff + 0] = float((w0p_0 >> 12) & 0x3FF);
                    ch0[dataOff + 1] = float(((w0p_0 & 0x03) << 8) | b1_0);
                    ch0[dataOff + 2] = float((w0p_1 >> 12) & 0x3FF);
                    ch0[dataOff + 3] = float(((w0p_1 & 0x03) << 8) | b1_1);
                    ch1[dataOff + 0] = ch1[dataOff + 1] = float(w0p_0 >> 22);
                    ch1[dataOff + 2] = ch1[dataOff + 3] = float(w0p_1 >> 22);
                    ch2[dataOff + 0] = ch2[dataOff + 1] = float((w0p_0 >> 2) & 0x3FF);
                    ch2[dataOff + 2] = ch2[dataOff + 3] = float((w0p_1 >> 2) & 0x3FF);
                }
            }

            template<typename CellDataType>
            __forceinline__ __device__ void save_yuv_10u_packed(uint8_t** dst, const size_t* stride, bool clip,
                                                                float alpha, const CellDataType& cell, uint32_t nCellX,
                                                                uint32_t nCellY) {
                (void)alpha;
                const float*  ch0        = cell.channel_ptr(0);
                const float*  ch1        = cell.channel_ptr(1);
                const float*  ch2        = cell.channel_ptr(2);
                constexpr int cw         = CellDataType::cell_width;
                constexpr int ch         = CellDataType::cell_height;
                constexpr int pair_bytes = 5;
                uint8_t*      plane      = dst[0];
                const size_t  str        = stride[0];

                // clip is uniform across the block, so branch on it once here rather
                // than four times per pair inside the unrolled loop. fclampf already
                // bounds its result to [0, 1023], so the clipped side needs no & 0x3FF;
                // the unclipped side still does, since the caller's value is arbitrary.
                if (clip) {
#pragma unroll
                    for (int row = 0; row < ch; ++row) {
                        const int dataOff = row * cw;
                        uint8_t*  rowBase = plane + (nCellY * ch + row) * str + nCellX * (cw / 2) * pair_bytes;
#pragma unroll
                        for (int p = 0; p < cw / 2; ++p) {
                            const float    y0  = ch0[dataOff + 2 * p];
                            const float    y1  = ch0[dataOff + 2 * p + 1];
                            const float    u   = (ch1[dataOff + 2 * p] + ch1[dataOff + 2 * p + 1]) * 0.5f;
                            const float    v   = (ch2[dataOff + 2 * p] + ch2[dataOff + 2 * p + 1]) * 0.5f;
                            const uint32_t u32 = (uint32_t)fclampf<bit_depth::bpp_10u>(u);
                            const uint32_t y0_ = (uint32_t)fclampf<bit_depth::bpp_10u>(y0);
                            const uint32_t v32 = (uint32_t)fclampf<bit_depth::bpp_10u>(v);
                            const uint32_t y1_ = (uint32_t)fclampf<bit_depth::bpp_10u>(y1);
                            const uint32_t w0p  = (u32 << 22) | (y0_ << 12) | (v32 << 2) | (y1_ >> 8);
                            YUV10UPair*    pair = (YUV10UPair*)(rowBase + p * pair_bytes);
                            pair->w0            = __byte_perm(w0p, 0, 0x0123);
                            pair->b1            = (uint8_t)(y1_ & 0xFF);
                        }
                    }
                } else {
#pragma unroll
                    for (int row = 0; row < ch; ++row) {
                        const int dataOff = row * cw;
                        uint8_t*  rowBase = plane + (nCellY * ch + row) * str + nCellX * (cw / 2) * pair_bytes;
#pragma unroll
                        for (int p = 0; p < cw / 2; ++p) {
                            const float    y0  = ch0[dataOff + 2 * p];
                            const float    y1  = ch0[dataOff + 2 * p + 1];
                            const float    u   = (ch1[dataOff + 2 * p] + ch1[dataOff + 2 * p + 1]) * 0.5f;
                            const float    v   = (ch2[dataOff + 2 * p] + ch2[dataOff + 2 * p + 1]) * 0.5f;
                            const uint32_t u32 = (uint32_t)u & 0x3FF;
                            const uint32_t y0_ = (uint32_t)y0 & 0x3FF;
                            const uint32_t v32 = (uint32_t)v & 0x3FF;
                            const uint32_t y1_ = (uint32_t)y1 & 0x3FF;
                            const uint32_t w0p  = (u32 << 22) | (y0_ << 12) | (v32 << 2) | (y1_ >> 8);
                            YUV10UPair*    pair = (YUV10UPair*)(rowBase + p * pair_bytes);
                            pair->w0            = __byte_perm(w0p, 0, 0x0123);
                            pair->b1            = (uint8_t)(y1_ & 0xFF);
                        }
                    }
                }
            }

            // Explicit specialization for NPPCellData<4,2,3,float>: one 10-byte quad per
            // row. clip is tested once so the whole SM takes one branch.
            template<>
            __forceinline__ __device__ void save_yuv_10u_packed<cell_4x2_3ch_float>(
                uint8_t** dst, const size_t* stride, bool clip, float alpha, const cell_4x2_3ch_float& cell,
                uint32_t nCellX, uint32_t nCellY) {
                (void)alpha;
                const float* ch0   = cell.channel_ptr(0);
                const float* ch1   = cell.channel_ptr(1);
                const float* ch2   = cell.channel_ptr(2);
                uint8_t*     plane = dst[0];
                const size_t str   = stride[0];

                // Five 16-bit stores beat ten 8-bit ones ~2.5x: partial sector writes cost a
                // read-modify-write and do not merge across instructions. To guarantee no
                // divergence and best aligned performance, BOTH the plane AND the stride
                // need to be even.
                const bool alignment_2 = (reinterpret_cast<uintptr_t>(plane) % 2u) == 0u && (str % 2u) == 0u;

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
                            __byte_perm((u32_0 << 22) | (y0_0 << 12) | (v32_0 << 2) | (y1_0 >> 8), 0, 0x0123);
                        const uint8_t  b1_0 = (uint8_t)(y1_0 & 0xFF);
                        const uint32_t w0_1 =
                            __byte_perm((u32_1 << 22) | (y0_1 << 12) | (v32_1 << 2) | (y1_1 >> 8), 0, 0x0123);
                        const uint8_t b1_1 = (uint8_t)(y1_1 & 0xFF);

                        store_two_pgroups(rowBase, w0_0, b1_0, w0_1, b1_1, alignment_2);
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

                        const uint32_t w0_0 = __byte_perm(((u32_0 & 0x3FF) << 22) | ((y0_0 & 0x3FF) << 12) |
                                                              ((v32_0 & 0x3FF) << 2) | ((y1_0 & 0x3FF) >> 8),
                                                          0, 0x0123);
                        const uint8_t  b1_0 = (uint8_t)(y1_0 & 0xFF);
                        const uint32_t w0_1 = __byte_perm(((u32_1 & 0x3FF) << 22) | ((y0_1 & 0x3FF) << 12) |
                                                              ((v32_1 & 0x3FF) << 2) | ((y1_1 & 0x3FF) >> 8),
                                                          0, 0x0123);
                        const uint8_t b1_1 = (uint8_t)(y1_1 & 0xFF);

                        store_two_pgroups(rowBase, w0_0, b1_0, w0_1, b1_1, alignment_2);
                    }
                }
            }

            template<typename CellDataType>
            __forceinline__ __device__ void safe_load_yuv_10u_packed(const uint8_t** src, const size_t* stride,
                                                                     CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                     int32_t imageWidth, int32_t imageHeight) {
                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;
                constexpr int     pair_bytes  = 5;
                const int32_t     basePixelX  = nCellX * cell_width;
                const int32_t     basePixelY  = nCellY * cell_height;
#if USE_FAST_PATH
                if (basePixelX >= 0 && basePixelX + cell_width <= imageWidth && basePixelY >= 0 &&
                    basePixelY + cell_height <= imageHeight) {
                    load_yuv_10u_packed<CellDataType>(src, stride, cell, nCellX, nCellY);
                    return;
                }
#endif
                float*         ch0   = cell.channel_ptr(0);
                float*         ch1   = cell.channel_ptr(1);
                float*         ch2   = cell.channel_ptr(2);
                const uint8_t* plane = src[0];
                const size_t   str   = stride[0];

                // Right-edge replication source: the last COMPLETE pgroup, whose Y1 is
                // the rightmost sample this function will read. An odd width's trailing
                // group is never touched - see the odd-width note at the top of this
                // file - so the odd column replicates the Y1 of the pair before it.
                // Width 1 has no complete pair and degenerates to the Y0 of group 0.
                const bool    haveCompletePair = imageWidth >= 2;
                const int32_t lastGroupIdx     = haveCompletePair ? (imageWidth / 2) - 1 : 0;
                // Columns at or beyond this are treated as off the right edge and
                // replicate, so no partial group is ever read.
                const int32_t evenWidth = imageWidth & ~1;

                // Width 0 or 1 has no complete pair, so floor(width / 2) * 5 == 0 bytes
                // are provably present and there is nothing safe to read - not even the
                // first group, which a caller may have sized as ceil(width * 2.5) == 3
                // bytes for width 1. Both edge reads below would take 4 bytes from group 0.
                // Zero the cell instead of decoding bytes that may not be allocated; exgest
                // writes nothing at these widths, so the two directions still agree.
                if (evenWidth == 0) {
#pragma unroll
                    for (int32_t k = 0; k < cell_width * cell_height; ++k) {
                        ch0[k] = 0.0f;
                        ch1[k] = 0.0f;
                        ch2[k] = 0.0f;
                    }
                    return;
                }

#pragma unroll
                for (int32_t row = 0; row < cell_height; ++row) {
                    const int32_t  pixelY  = basePixelY + row;
                    const int32_t  clampY  = nppdx_max(0, nppdx_min(pixelY, imageHeight - 1));
                    const int32_t  dataOff = row * cell_width;
                    const uint8_t* rowPtr  = plane + (size_t)clampY * str;

                    // Both edges are resolved per row rather than per pixel. A cell can
                    // straddle both when the image is narrower than the cell, so picking
                    // one set from basePixelX alone is not enough.
                    const YUV10UPair* pLeft  = (const YUV10UPair*)rowPtr;
                    const uint32_t    w0pLeft = __byte_perm(pLeft->w0, 0, 0x0123);
                    const float       leftY  = float((w0pLeft >> 12) & 0x3FF);
                    const float       leftU  = float(w0pLeft >> 22);
                    const float       leftV  = float((w0pLeft >> 2) & 0x3FF);

                    const YUV10UPair* pRight  = (const YUV10UPair*)(rowPtr + (size_t)lastGroupIdx * pair_bytes);
                    const uint32_t    w0pRight = __byte_perm(pRight->w0, 0, 0x0123);
                    const float       rightU  = float(w0pRight >> 22);
                    const float       rightV  = float((w0pRight >> 2) & 0x3FF);
                    const float       rightY  = haveCompletePair
                                                    ? float(((w0pRight & 0x03) << 8) | pRight->b1)
                                                    : float((w0pRight >> 12) & 0x3FF);

#pragma unroll
                    for (int32_t i = 0; i < cell_width; ++i) {
                        const int32_t imgX = basePixelX + i;
                        if (imgX >= 0 && imgX < evenWidth) {
                            const int32_t     pairIdx = imgX / 2;
                            const YUV10UPair* pair    = (const YUV10UPair*)(rowPtr + (size_t)pairIdx * pair_bytes);
                            const uint32_t    w0p      = __byte_perm(pair->w0, 0, 0x0123);
                            ch1[dataOff + i] = float(w0p >> 22);
                            ch2[dataOff + i] = float((w0p >> 2) & 0x3FF);
                            ch0[dataOff + i] = ((imgX & 1) != 0) ? float(((w0p & 0x03) << 8) | pair->b1)
                                                                 : float((w0p >> 12) & 0x3FF);
                        } else if (imgX < 0) {
                            ch0[dataOff + i] = leftY;
                            ch1[dataOff + i] = leftU;
                            ch2[dataOff + i] = leftV;
                        } else {
                            ch0[dataOff + i] = rightY;
                            ch1[dataOff + i] = rightU;
                            ch2[dataOff + i] = rightV;
                        }
                    }
                }
            }

            template<typename CellDataType>
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
                constexpr int pair_bytes = 5;
                const int32_t basePixelX = nCellX * cw;
                const int32_t basePixelY = nCellY * ch;
#if USE_FAST_PATH
                if (basePixelX >= 0 && basePixelX + cw <= imageWidth && basePixelY >= 0 &&
                    basePixelY + ch <= imageHeight) {
                    // Fast save uses uint32_t cell indices; here nCellX/Y are non-negative on this branch.
                    save_yuv_10u_packed<CellDataType>(
                        dst, stride, clip, alpha, cell, static_cast<uint32_t>(nCellX), static_cast<uint32_t>(nCellY));
                    return;
                }
#endif
                // Whole pgroups only: a partial trailing group may not exist in the
                // caller's allocation. See the odd-width note at the top of this file.
                // Even widths are unaffected, since imageWidth & ~1 == imageWidth.
                const int32_t evenWidth = imageWidth & ~1;
                const int32_t srcStart  = nppdx_max(0, -basePixelX);
                const int32_t srcEnd    = nppdx_min(cw, evenWidth - basePixelX);
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

                    // The cell only lives in registers if every index into it is a
                    // compile-time constant, so this spans the constant range and moves
                    // the bounds into a predicate. Iterating [srcStart, srcEnd) instead
                    // makes ch0[dataOff + i] a runtime index, which forces the whole cell
                    // to local memory for the entire kernel - including the interior cells
                    // that return above, and including any fused processing stage that
                    // shares the cell. safe_load_yuv_10u_packed already does it this way.
#pragma unroll
                    for (int32_t i = 0; i < cw; i += 2) {
                        if (i < srcStart || i >= srcEnd)
                            continue;
                        // i + 1 is always in range here, so both pixels of the pair are
                        // real. basePixelX is a multiple of an even cw and evenWidth is
                        // even, so srcEnd is even; i steps by 2, so i < srcEnd implies
                        // i <= srcEnd - 2. A half-pair cannot reach this loop, because
                        // whole pgroups are the only thing exgest writes.
                        float y0 = ch0[dataOff + i];
                        float y1 = ch0[dataOff + i + 1];
                        float u  = (ch1[dataOff + i] + ch1[dataOff + i + 1]) * 0.5f;
                        float v  = (ch2[dataOff + i] + ch2[dataOff + i + 1]) * 0.5f;
                        // One uniform branch on clip rather than four per-component
                        // selects. fclampf already returns [0, 1023], so only the
                        // unclipped side needs masking to 10 bits.
                        uint32_t u32, y0_, v32, y1_;
                        if (clip) {
                            u32 = (uint32_t)fclampf<bit_depth::bpp_10u>(u);
                            y0_ = (uint32_t)fclampf<bit_depth::bpp_10u>(y0);
                            v32 = (uint32_t)fclampf<bit_depth::bpp_10u>(v);
                            y1_ = (uint32_t)fclampf<bit_depth::bpp_10u>(y1);
                        } else {
                            u32 = (uint32_t)u & 0x3FF;
                            y0_ = (uint32_t)y0 & 0x3FF;
                            v32 = (uint32_t)v & 0x3FF;
                            y1_ = (uint32_t)y1 & 0x3FF;
                        }
                        const int32_t  pairIdx = (firstImgX + (i - srcStart)) / 2;
                        const uint32_t w0p      = (u32 << 22) | (y0_ << 12) | (v32 << 2) | (y1_ >> 8);
                        YUV10UPair*    pair    = (YUV10UPair*)(rowDst + (size_t)pairIdx * pair_bytes);
                        pair->w0               = __byte_perm(w0p, 0, 0x0123);
                        pair->b1               = (uint8_t)(y1_ & 0xFF);
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
                    safe_load_yuv_10u_packed<CellDataType>(src, stride, cell, nCellX, nCellY, imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell, int32_t nCellX,
                                                                 int32_t nCellY, uint32_t imageWidth,
                                                                 uint32_t imageHeight) {
                    safe_save_yuv_10u_packed<CellDataType>(dst, stride, clip, alpha, cell, nCellX, nCellY,
                                                           (int32_t)imageWidth, (int32_t)imageHeight);
                }
            };

        } // namespace backend
    } // namespace detail
} // namespace nppdx
#endif // NPPDX_DETAIL_BACKEND_INGEST_EXGEST_YUV_10U_PACKED_FAMILY_HPP
