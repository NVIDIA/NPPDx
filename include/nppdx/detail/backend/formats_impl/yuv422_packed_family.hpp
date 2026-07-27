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

#ifndef NPPDX_DETAIL_BACKEND_INGEST_EXGEST_YUV422_PACKED_FAMILY_HPP
#define NPPDX_DETAIL_BACKEND_INGEST_EXGEST_YUV422_PACKED_FAMILY_HPP

#include <cstdint>
#include "nppdx/detail/utils/force_align.hpp"
#include "nppdx/detail/backend/constants.hpp"
#include "nppdx/detail/backend/formats_impl/format_backend.hpp"
#include "nppdx/detail/backend/formats_impl/clamp_utils.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {

            // YUV422 Packed backend

// Fast path control (from NPP)
#ifndef USE_FAST_PATH
#    define USE_FAST_PATH 1
#endif

            // YUV 4:2:2 packed format helpers (Y0 U0 Y1 V0 pattern)
            // CHY0: position of first Y (usually 0)
            // CHY1: position of second Y (usually 2)
            // CHU: position of U (usually 1)
            // CHV: position of V (usually 3)
            // Template parameters allow handling different bit depths and channel orders

            //4:2:2 packed template - fast path (no boundary checking)
            //
            // InputShift: right-shifts each integer sample before float conversion, for formats that
            // pack their data in the upper bits of a wider container (Y210: 10-bit data in upper 10
            // of uint16 → InputShift = 6). Default 0 → no shift, byte-identical to plain `float(src)`.
            template<typename T, int CHY0, int CHY1, int CHU, int CHV, int InputShift = 0,
                     typename CellDataType>
            __forceinline__ __device__ void load_yuv_422_packed(const uint8_t** src, const size_t* stride,
                                                                CellDataType& cell, uint32_t nCellX, uint32_t nCellY) {
                float* ch0 = cell.channel_ptr(0); // Y channel
                float* ch1 = cell.channel_ptr(1); // U channel
                float* ch2 = cell.channel_ptr(2); // V channel

                constexpr unsigned int num_pairs      = CellDataType::cell_width / 2; // 4:2:2 has 2 Y per UV pair
                constexpr unsigned int block_elements = num_pairs * 4;                // 4 channels per pair

                constexpr unsigned int cell_height = CellDataType::cell_height;
                constexpr unsigned int cell_width  = CellDataType::cell_width;


#pragma pack(push, 1)
                struct PixelBlock {
                    union {
                        T                                      ch[block_elements];
                        ForceAlign<block_elements * sizeof(T)> _align;
                    };
                };
#pragma pack(pop)

                const uint8_t* srcPlane  = src[0];
                const size_t   srcStride = stride[0];


                // Fast path - no boundary checking, assumes cell fits entirely
                const size_t src0 = (size_t)srcPlane + nCellY * cell_height * srcStride + nCellX * sizeof(PixelBlock);

                // Process each row
#pragma unroll
                for (unsigned int row = 0; row < cell_height; ++row) {
                    const size_t       srcRow  = src0 + row * srcStride;
                    PixelBlock         block   = *(PixelBlock*)srcRow;
                    const unsigned int dataOff = row * cell_width;

                    // Extract Y, U, V from packed format
#pragma unroll
                    for (unsigned int p = 0; p < num_pairs; ++p) {
                        const unsigned int blockOff = p * 4;
                        const unsigned int pixOff   = p * 2;

                        ch0[dataOff + pixOff]     = float(block.ch[CHY0 + blockOff] >> InputShift);
                        ch0[dataOff + pixOff + 1] = float(block.ch[CHY1 + blockOff] >> InputShift);
                        ch1[dataOff + pixOff]     = float(block.ch[CHU + blockOff] >> InputShift);
                        ch1[dataOff + pixOff + 1] = float(block.ch[CHU + blockOff] >> InputShift); // duplicate U
                        ch2[dataOff + pixOff]     = float(block.ch[CHV + blockOff] >> InputShift);
                        ch2[dataOff + pixOff + 1] = float(block.ch[CHV + blockOff] >> InputShift); // duplicate V
                    }
                }

            } //load_yuv_422_packed

            //4:2:2 packed template - fast path save (no boundary checking)
            //
            // OutputShift: left-shifts the rounded sample after the (T) cast, for formats that pack
            // their data in the upper bits of a wider container (Y210: 10-bit data in upper 10 of
            // uint16 → Depth = bpp_10u, OutputShift = 6). Default 0 → no shift, byte-identical to
            // the original cast.
            template<typename T, bit_depth Depth, int CHY0, int CHY1, int CHU, int CHV,
                     int OutputShift = 0, typename CellDataType>
            __forceinline__ __device__ void save_yuv_422_packed(uint8_t** dst, const size_t* stride, bool clip,
                                                                float alpha, const CellDataType& cell, uint32_t nCellX,
                                                                uint32_t nCellY) {
                (void)alpha; // unused for YUV_422 formats

                const float* ch0 = cell.channel_ptr(0); // Y channel
                const float* ch1 = cell.channel_ptr(1); // U channel
                const float* ch2 = cell.channel_ptr(2); // V channel

                constexpr unsigned int cell_height = CellDataType::cell_height;
                constexpr unsigned int cell_width  = CellDataType::cell_width;

                constexpr unsigned int num_pairs      = cell_width / 2; // 4:2:2 has 2 Y per UV pair
                constexpr unsigned int block_elements = num_pairs * 4;  // 4 channels per pair

#pragma pack(push, 1)
                struct PixelBlock {
                    union {
                        T                                      ch[block_elements];
                        ForceAlign<block_elements * sizeof(T)> _align;
                    };
                };
#pragma pack(pop)

                uint8_t*     dstPlane  = dst[0];
                const size_t dstStride = stride[0];
                const size_t dst0 = (size_t)dstPlane + nCellY * cell_height * dstStride + nCellX * sizeof(PixelBlock);


                PixelBlock block;

                // Process each row
#pragma unroll
                for (unsigned int row = 0; row < cell_height; ++row) {
                    const size_t       dstRow  = dst0 + row * dstStride;
                    const unsigned int dataOff = row * cell_width;

                    // Pack Y, U, V into packed format
                    if (clip) {
#pragma unroll
                        for (unsigned int p = 0; p < num_pairs; ++p) {
                            const unsigned int blockOff = p * 4;
                            const unsigned int pixOff   = p * 2;

                            block.ch[CHY0 + blockOff] =
                                (T)((uint16_t)fclampf<Depth>(ch0[dataOff + pixOff]) << OutputShift);
                            block.ch[CHY1 + blockOff] =
                                (T)((uint16_t)fclampf<Depth>(ch0[dataOff + pixOff + 1]) << OutputShift);
                            block.ch[CHU + blockOff]  = (T)((uint16_t)fclampf<Depth>(
                                (ch1[dataOff + pixOff] + ch1[dataOff + pixOff + 1]) * 0.5f) << OutputShift);
                            block.ch[CHV + blockOff]  = (T)((uint16_t)fclampf<Depth>(
                                (ch2[dataOff + pixOff] + ch2[dataOff + pixOff + 1]) * 0.5f) << OutputShift);
                        }
                    } else {
#pragma unroll
                        for (unsigned int p = 0; p < num_pairs; ++p) {
                            const unsigned int blockOff = p * 4;
                            const unsigned int pixOff   = p * 2;

                            block.ch[CHY0 + blockOff] = (T)((uint16_t)(ch0[dataOff + pixOff]) << OutputShift);
                            block.ch[CHY1 + blockOff] = (T)((uint16_t)(ch0[dataOff + pixOff + 1]) << OutputShift);
                            block.ch[CHU + blockOff]  = (T)((uint16_t)(
                                (ch1[dataOff + pixOff] + ch1[dataOff + pixOff + 1]) * 0.5f) << OutputShift);
                            block.ch[CHV + blockOff]  = (T)((uint16_t)(
                                (ch2[dataOff + pixOff] + ch2[dataOff + pixOff + 1]) * 0.5f) << OutputShift);
                        }
                    }

                    *(PixelBlock*)dstRow = block;
                }

            } //save_yuv_422_packed

            // <4,2,3> uint16_t Y210 specialization: 2x ushort4 per row (CHY0=0, CHY1=2, CHU=1, CHV=3), register-friendly.
            // Note: OutputShift=0 here is a placeholder for the legacy 16-bit-aligned call path. The new y210 backend
            // uses <uint16_t, bpp_10u, 0, 2, 1, 3, 6, cell_4x2_3ch_float> and falls through to the generic template.
            template<>
            __forceinline__ __device__ void save_yuv_422_packed<uint16_t, bit_depth::bpp_16u, 0, 2, 1, 3, 0,
                                                                cell_4x2_3ch_float>(uint8_t** dst, const size_t* stride,
                                                                                    bool clip, float alpha,
                                                                                    const cell_4x2_3ch_float& cell,
                                                                                    uint32_t nCellX, uint32_t nCellY) {
                (void)alpha;
                const float*           ch0         = cell.channel_ptr(0);
                const float*           ch1         = cell.channel_ptr(1);
                const float*           ch2         = cell.channel_ptr(2);
                constexpr unsigned int cell_width  = 4;
                constexpr unsigned int cell_height = 2;
                constexpr size_t       rowBytes    = 2 * 4 * sizeof(uint16_t); // 2 pairs * 4 elements
                uint8_t*               dstPlane    = dst[0];
                const size_t           dstStride   = stride[0];
                const size_t           dst0 = (size_t)dstPlane + nCellY * cell_height * dstStride + nCellX * rowBytes;

                if (clip) {
#pragma unroll
                    for (unsigned int row = 0; row < cell_height; ++row) {
                        uint8_t*     rowBase     = (uint8_t*)(dst0 + row * dstStride);
                        const size_t dataOff     = row * cell_width;
                        const bool   rowAligned8 = (reinterpret_cast<uintptr_t>(rowBase) & 7u) == 0u;
                        if (!rowAligned8) {
#pragma unroll
                            for (unsigned int p = 0; p < 2; ++p) {
                                const unsigned int pixOff = p * 2;
                                uint16_t*          base   = (uint16_t*)(rowBase + p * 8);
                                base[0] = (uint16_t)fclampf<bit_depth::bpp_16u>(ch0[dataOff + pixOff]);
                                base[1] = (uint16_t)fclampf<bit_depth::bpp_16u>(
                                    (ch1[dataOff + pixOff] + ch1[dataOff + pixOff + 1]) * 0.5f);
                                base[2] = (uint16_t)fclampf<bit_depth::bpp_16u>(ch0[dataOff + pixOff + 1]);
                                base[3] = (uint16_t)fclampf<bit_depth::bpp_16u>(
                                    (ch2[dataOff + pixOff] + ch2[dataOff + pixOff + 1]) * 0.5f);
                            }
                        } else {
                            ushort4 a, b;
                            a.x = (uint16_t)fclampf<bit_depth::bpp_16u>(ch0[dataOff + 0]);
                            a.y = (uint16_t)fclampf<bit_depth::bpp_16u>((ch1[dataOff + 0] + ch1[dataOff + 1]) * 0.5f);
                            a.z = (uint16_t)fclampf<bit_depth::bpp_16u>(ch0[dataOff + 1]);
                            a.w = (uint16_t)fclampf<bit_depth::bpp_16u>((ch2[dataOff + 0] + ch2[dataOff + 1]) * 0.5f);
                            b.x = (uint16_t)fclampf<bit_depth::bpp_16u>(ch0[dataOff + 2]);
                            b.y = (uint16_t)fclampf<bit_depth::bpp_16u>((ch1[dataOff + 2] + ch1[dataOff + 3]) * 0.5f);
                            b.z = (uint16_t)fclampf<bit_depth::bpp_16u>(ch0[dataOff + 3]);
                            b.w = (uint16_t)fclampf<bit_depth::bpp_16u>((ch2[dataOff + 2] + ch2[dataOff + 3]) * 0.5f);
                            *(ushort4*)(rowBase)     = a;
                            *(ushort4*)(rowBase + 8) = b;
                        }
                    }
                } else {
#pragma unroll
                    for (unsigned int row = 0; row < cell_height; ++row) {
                        uint8_t*     rowBase     = (uint8_t*)(dst0 + row * dstStride);
                        const size_t dataOff     = row * cell_width;
                        const bool   rowAligned8 = (reinterpret_cast<uintptr_t>(rowBase) & 7u) == 0u;
                        if (!rowAligned8) {
#pragma unroll
                            for (unsigned int p = 0; p < 2; ++p) {
                                const unsigned int pixOff = p * 2;
                                uint16_t*          base   = (uint16_t*)(rowBase + p * 8);
                                base[0]                   = (uint16_t)ch0[dataOff + pixOff];
                                base[1] = (uint16_t)((ch1[dataOff + pixOff] + ch1[dataOff + pixOff + 1]) * 0.5f);
                                base[2] = (uint16_t)ch0[dataOff + pixOff + 1];
                                base[3] = (uint16_t)((ch2[dataOff + pixOff] + ch2[dataOff + pixOff + 1]) * 0.5f);
                            }
                        } else {
                            ushort4 a, b;
                            a.x                      = (uint16_t)ch0[dataOff + 0];
                            a.y                      = (uint16_t)((ch1[dataOff + 0] + ch1[dataOff + 1]) * 0.5f);
                            a.z                      = (uint16_t)ch0[dataOff + 1];
                            a.w                      = (uint16_t)((ch2[dataOff + 0] + ch2[dataOff + 1]) * 0.5f);
                            b.x                      = (uint16_t)ch0[dataOff + 2];
                            b.y                      = (uint16_t)((ch1[dataOff + 2] + ch1[dataOff + 3]) * 0.5f);
                            b.z                      = (uint16_t)ch0[dataOff + 3];
                            b.w                      = (uint16_t)((ch2[dataOff + 2] + ch2[dataOff + 3]) * 0.5f);
                            *(ushort4*)(rowBase)     = a;
                            *(ushort4*)(rowBase + 8) = b;
                        }
                    }
                }
            }

            // Helper: Load one row of YUV 4:2:2 packed data with edge clamping
            // Handles left edge, right edge, and valid interior pixels
            // Outputs cell_width pixels to ch0/ch1/ch2 starting at dataOffset
            // InputShift: right-shifts each integer sample before float conversion (see load_yuv_422_packed).
            template<typename T, int CHY0, int CHY1, int CHU, int CHV, int cell_width, int InputShift = 0>
            __forceinline__ __device__ void load_yuv422_row(const uint8_t* rowData, float* ch0, float* ch1, float* ch2,
                                                            int dataOffset, int32_t basePixelX, int32_t imageWidth) {


                constexpr int32_t number_of_elements_per_pair = 4;
#pragma pack(push, 1)
                struct PixelPair {
                    union {
                        T                                                   ch[number_of_elements_per_pair];
                        ForceAlign<number_of_elements_per_pair * sizeof(T)> _align;
                    };
                };
#pragma pack(pop)

                const PixelPair* rowPtr = (const PixelPair*)rowData;

                // For odd widths, the last pair is incomplete (only Y+U stored, V is in next row's memory)
                // lastCompletePairIdx = last pair with valid V data
                const bool    isOddWidth          = (imageWidth & 1);
                const int32_t lastCompletePairIdx = (imageWidth / 2) - 1; // Last pair with valid U and V
                const int32_t lastPairIdx         = (imageWidth - 1) / 2; // May be incomplete for odd widths

                // Get edge values for clamping - use last COMPLETE pair for V
                const PixelPair leftEdge  = rowPtr[0];
                const PixelPair rightEdge = rowPtr[lastCompletePairIdx];

                // For odd widths, read Y and U of the last pixel from the incomplete pair
                // V comes from the last complete pair (pair before the half-pair)
                const T* lastPixelPtr = reinterpret_cast<const T*>(rowData) + lastPairIdx * number_of_elements_per_pair;
                const float rightEdgeY = isOddWidth ? float(lastPixelPtr[CHY0] >> InputShift)
                                                    : float(rightEdge.ch[CHY1] >> InputShift);
                const float rightEdgeU = isOddWidth ? float(lastPixelPtr[CHU] >> InputShift)
                                                    : float(rightEdge.ch[CHU] >> InputShift);
                const float rightEdgeV = isOddWidth ? float(rightEdge.ch[CHV] >> InputShift)
                                                    : float(rightEdge.ch[CHV] >> InputShift);
                // Pre-fill all pixels with appropriate edge values (enables unrolling)
#pragma unroll
                for (int i = 0; i < cell_width; ++i) {
                    const int32_t imgX = basePixelX + i;
                    // Use left edge for pixels off left, right edge for pixels off right
                    // For odd widths: Y and U from incomplete pair, V from last complete pair
                    const float edgeY   = (imgX < 0) ? float(leftEdge.ch[CHY0] >> InputShift) : rightEdgeY;
                    const float edgeU   = (imgX < 0) ? float(leftEdge.ch[CHU] >> InputShift) : rightEdgeU;
                    const float edgeV   = (imgX < 0) ? float(leftEdge.ch[CHV] >> InputShift) : rightEdgeV;
                    ch0[dataOffset + i] = edgeY;
                    ch1[dataOffset + i] = edgeU;
                    ch2[dataOffset + i] = edgeV;
                }

                // Overwrite valid pixels from source
                const int32_t validStart = nppdx_max(0, -basePixelX);
                const int32_t validEnd   = nppdx_min(cell_width, imageWidth - basePixelX);

                if (validStart < validEnd) {
                    // Process valid pixel range
#pragma unroll
                    for (int i = 0; i < cell_width; ++i) {
                        if (i >= validStart && i < validEnd) {
                            const int32_t imgX           = basePixelX + i;
                            const int32_t pairIdx        = imgX / 2;
                            const bool    isSecondInPair = (imgX & 1);

                            if (pairIdx <= lastCompletePairIdx) {
                                // Complete pair: read Y, U, V directly
                                const PixelPair pair = rowPtr[pairIdx];
                                ch0[dataOffset + i]  = isSecondInPair ? float(pair.ch[CHY1] >> InputShift)
                                                                      : float(pair.ch[CHY0] >> InputShift);
                                ch1[dataOffset + i]  = float(pair.ch[CHU] >> InputShift);
                                ch2[dataOffset + i]  = float(pair.ch[CHV] >> InputShift);
                            } else {
                                // Incomplete pair (odd width's last pixel): Y and U from half-pair, V from previous pair
                                const T* halfPair =
                                    reinterpret_cast<const T*>(rowData) + pairIdx * number_of_elements_per_pair;
                                ch0[dataOffset + i] = float(halfPair[CHY0] >> InputShift);    // Y from incomplete pair
                                ch1[dataOffset + i] = float(halfPair[CHU] >> InputShift);     // U from incomplete pair
                                ch2[dataOffset + i] = float(rightEdge.ch[CHV] >> InputShift); // V from last complete pair
                            }
                        }
                    }
                }
            }

            // Safe path: same pattern as safe_load_Nchoose3 (needLeft, one edge, single loop, no prevClampY).
            // Packed layout: 4 elements per pair (Y0,U,Y1,V); cell_width/2 pairs per row.
            template<typename T, int CHY0, int CHY1, int CHU, int CHV, int InputShift = 0,
                     typename CellDataType>
            __forceinline__ __device__ void safe_load_yuv_422_packed(const uint8_t** src, const size_t* stride,
                                                                     CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                     int32_t imageWidth, int32_t imageHeight) {
                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;

#if USE_FAST_PATH
                constexpr size_t packed_align = (cell_width / 2) * 4 * sizeof(T);
                if (basePixelX >= 0 && basePixelX + cell_width - 1 < imageWidth && basePixelY >= 0 &&
                    basePixelY + cell_height - 1 < imageHeight && (stride[0] % packed_align) == 0 &&
                    (reinterpret_cast<uintptr_t>(src[0]) % packed_align) == 0) {
                    load_yuv_422_packed<T, CHY0, CHY1, CHU, CHV, InputShift>(src, stride, cell, nCellX, nCellY);
                    return;
                }
#endif

                constexpr int32_t elements_per_pair = 4;
                float*            ch0               = cell.channel_ptr(0);
                float*            ch1               = cell.channel_ptr(1);
                float*            ch2               = cell.channel_ptr(2);
                const uint8_t*    srcPlane          = src[0];
                const size_t      srcStride         = stride[0];
                const bool        needLeft          = (basePixelX < 0);

                // Last complete pair (4:2:2 is even-width; odd-width last pixel gets replicate, not partial read)
                const int32_t lastCompletePairIdx = (imageWidth / 2) - 1;
                const bool    hasCompletePair     = (lastCompletePairIdx >= 0);

#pragma unroll
                for (int32_t row = 0; row < cell_height; ++row) {
                    const int32_t pixelY     = basePixelY + row;
                    const int32_t clampY     = nppdx_max(0, nppdx_min(pixelY, imageHeight - 1));
                    const int32_t dataOffset = row * cell_width;

                    const T* rowPtr = (const T*)(srcPlane + clampY * srcStride);

                    // One edge (left or right). Left replicate = first in pair; right replicate = second in pair. Right edge: from last complete pair (odd-width: ignore partial pair).
                    float edgeY0, edgeY1, edgeU, edgeV;
                    if (needLeft) {
                        edgeY0 = float(rowPtr[0 * elements_per_pair + CHY0] >> InputShift);
                        edgeY1 = edgeY0; // left edge: replicate first in pair only
                        edgeU  = float(rowPtr[0 * elements_per_pair + CHU] >> InputShift);
                        edgeV  = float(rowPtr[0 * elements_per_pair + CHV] >> InputShift);
                    } else {
                        if (hasCompletePair) {
                            edgeY0 = float(rowPtr[lastCompletePairIdx * elements_per_pair + CHY1] >> InputShift);
                            edgeY1 = edgeY0; // right edge: replicate second in pair only
                            edgeU  = float(rowPtr[lastCompletePairIdx * elements_per_pair + CHU] >> InputShift);
                            edgeV  = float(rowPtr[lastCompletePairIdx * elements_per_pair + CHV] >> InputShift);
                        } else {
                            // imageWidth 0 or 1: no complete pair; use pair 0 for Y,U and replicate U for V
                            edgeY0 = float(rowPtr[0 * elements_per_pair + CHY0] >> InputShift);
                            edgeY1 = edgeY0;
                            edgeU  = float(rowPtr[0 * elements_per_pair + CHU] >> InputShift);
                            edgeV  = edgeU;
                        }
                    }

                    // Single loop (full cell_width, unrollable): in bounds -> fetch from pair, else -> replicate edge
#pragma unroll
                    for (int32_t i = 0; i < cell_width; ++i) {
                        const int32_t imgX = basePixelX + i;
                        if (imgX >= 0 && imgX < imageWidth) {
                            const int32_t pairIdx        = imgX / 2;
                            const bool    isSecondInPair = (imgX & 1) != 0;
                            if (hasCompletePair && pairIdx <= lastCompletePairIdx) {
                                const int32_t off   = pairIdx * elements_per_pair;
                                ch0[dataOffset + i] = float(rowPtr[off + (isSecondInPair ? CHY1 : CHY0)] >> InputShift);
                                ch1[dataOffset + i] = float(rowPtr[off + CHU] >> InputShift);
                                ch2[dataOffset + i] = float(rowPtr[off + CHV] >> InputShift);
                            } else {
                                // Odd width last pixel: default = clone from second pixel of last complete pair.
                                // 4 overrides when incomplete pair (slots 0,1) layout is known: new Y + clone chroma.
                                if (hasCompletePair) {
                                    const int32_t off_prev = lastCompletePairIdx * elements_per_pair;
                                    const int32_t off_inc  = (lastCompletePairIdx + 1) * elements_per_pair;
                                    // Override 1: Y0 @ 0, U @ 1 -> new Y from 0, U from 1, V cloned
                                    if constexpr ((CHY0 == 0 || CHY1 == 0) && CHU == 1) {
                                        ch0[dataOffset + i] = float(rowPtr[off_inc + 0] >> InputShift);
                                        ch1[dataOffset + i] = float(rowPtr[off_inc + 1] >> InputShift);
                                        ch2[dataOffset + i] = float(rowPtr[off_prev + CHV] >> InputShift);
                                    }
                                    // Override 2: Y0 @ 0, V @ 1 -> new Y from 0, V from 1, U cloned
                                    else if constexpr ((CHY0 == 0 || CHY1 == 0) && CHV == 1) {
                                        ch0[dataOffset + i] = float(rowPtr[off_inc + 0] >> InputShift);
                                        ch1[dataOffset + i] = float(rowPtr[off_prev + CHU] >> InputShift);
                                        ch2[dataOffset + i] = float(rowPtr[off_inc + 1] >> InputShift);
                                    }
                                    // Override 3: Y0 @ 0, Y1 @ 1 -> new Y from 0, clone U and V
                                    else if constexpr ((CHY0 == 0 && CHY1 == 1) || (CHY1 == 0 && CHY0 == 1)) {
                                        ch0[dataOffset + i] = float(rowPtr[off_inc + 0] >> InputShift);
                                        ch1[dataOffset + i] = float(rowPtr[off_prev + CHU] >> InputShift);
                                        ch2[dataOffset + i] = float(rowPtr[off_prev + CHV] >> InputShift);
                                    }
                                    // Override 4: Y0 @ 1 -> new Y from 1; slot 0 is U, V, or Y2
                                    else if constexpr (CHY0 == 1 || CHY1 == 1) {
                                        ch0[dataOffset + i] = float(rowPtr[off_inc + 1] >> InputShift);
                                        if constexpr (CHU == 0) {
                                            ch1[dataOffset + i] = float(rowPtr[off_inc + 0] >> InputShift);
                                            ch2[dataOffset + i] = float(rowPtr[off_prev + CHV] >> InputShift);
                                        } else if constexpr (CHV == 0) {
                                            ch1[dataOffset + i] = float(rowPtr[off_prev + CHU] >> InputShift);
                                            ch2[dataOffset + i] = float(rowPtr[off_inc + 0] >> InputShift);
                                        } else {
                                            ch1[dataOffset + i] = float(rowPtr[off_prev + CHU] >> InputShift);
                                            ch2[dataOffset + i] = float(rowPtr[off_prev + CHV] >> InputShift);
                                        }
                                    }
                                    // Default: clone second pixel of last complete pair
                                    else {
                                        ch0[dataOffset + i] = float(rowPtr[off_prev + CHY1] >> InputShift);
                                        ch1[dataOffset + i] = float(rowPtr[off_prev + CHU] >> InputShift);
                                        ch2[dataOffset + i] = float(rowPtr[off_prev + CHV] >> InputShift);
                                    }
                                } else {
                                    ch0[dataOffset + i] = float(rowPtr[0 * elements_per_pair + CHY0] >> InputShift);
                                    ch1[dataOffset + i] = float(rowPtr[0 * elements_per_pair + CHU] >> InputShift);
                                    ch2[dataOffset + i] = float(rowPtr[0 * elements_per_pair + CHV] >> InputShift);
                                }
                            }
                        } else {
                            ch0[dataOffset + i] = (i & 1) ? edgeY1 : edgeY0;
                            ch1[dataOffset + i] = edgeU;
                            ch2[dataOffset + i] = edgeV;
                        }
                    }
                }
            }

            // Helper: Save YUV 4:2:2 pixel pairs with boundary checking
            // Processes pixels from srcStart to srcEnd in the cell data, writing to consecutive pairs
            // OutputShift: left-shifts the rounded sample after the (T) cast (see save_yuv_422_packed).
            template<typename T, bit_depth Depth, int CHY0, int CHY1, int CHU, int CHV, int cell_width,
                     int OutputShift = 0>
            __forceinline__ __device__ void save_yuv422_row(uint8_t* rowData, const float* ch0, const float* ch1,
                                                            const float* ch2, int dataOffset, int32_t srcStart,
                                                            int32_t srcEnd, int32_t firstImgX, int32_t imageWidth,
                                                            bool clip) {

                constexpr int32_t number_of_elements_per_pair = 4;
#pragma pack(push, 1)
                struct PixelPair {
                    union {
                        T                         ch[4];
                        ForceAlign<4 * sizeof(T)> _align;
                    };
                };
#pragma pack(pop)

                PixelPair* rowPtr = (PixelPair*)rowData;

                // Process pixel pairs. For 4:2:2, we write complete pairs.
                // If firstImgX is odd, we need special handling for the first pixel.
                const bool startsOdd = (firstImgX & 1);
                int        srcIdx    = srcStart;
                int32_t    imgX      = firstImgX;

                // Handle first pixel if it starts on odd boundary (second pixel in a pair)
                if (startsOdd && srcIdx < srcEnd) {
                    // Read-modify-write: load existing pair, update Y1
                    const int32_t pairIdx = imgX / 2;
                    PixelPair     pair    = rowPtr[pairIdx];
                    const float   y1      = ch0[dataOffset + srcIdx];
                    const float   u       = ch1[dataOffset + srcIdx];
                    const float   v       = ch2[dataOffset + srcIdx];
                    // Average U/V with existing values in pair (existing values on disk are
                    // packed with OutputShift; bring them into cell space before averaging).
                    if (clip) {
                        pair.ch[CHY1] = (T)((uint16_t)fclampf<Depth>(y1) << OutputShift);
                        pair.ch[CHU]  = (T)((uint16_t)fclampf<Depth>(
                            (float(pair.ch[CHU] >> OutputShift) + u) * 0.5f) << OutputShift);
                        pair.ch[CHV]  = (T)((uint16_t)fclampf<Depth>(
                            (float(pair.ch[CHV] >> OutputShift) + v) * 0.5f) << OutputShift);
                    } else {
                        pair.ch[CHY1] = (T)((uint16_t)y1 << OutputShift);
                        pair.ch[CHU]  = (T)((uint16_t)(
                            (float(pair.ch[CHU] >> OutputShift) + u) * 0.5f) << OutputShift);
                        pair.ch[CHV]  = (T)((uint16_t)(
                            (float(pair.ch[CHV] >> OutputShift) + v) * 0.5f) << OutputShift);
                    }
                    rowPtr[pairIdx] = pair;
                    ++srcIdx;
                    ++imgX;
                }

                // Process full pairs (2 pixels at a time)
#pragma unroll
                for (int p = 0; p < cell_width / 2; ++p) {
                    if (srcIdx + 1 < srcEnd) {
                        const int32_t pairIdx = imgX / 2;
                        const float   y0      = ch0[dataOffset + srcIdx];
                        const float   y1      = ch0[dataOffset + srcIdx + 1];
                        const float   u_avg   = (ch1[dataOffset + srcIdx] + ch1[dataOffset + srcIdx + 1]) * 0.5f;
                        const float   v_avg   = (ch2[dataOffset + srcIdx] + ch2[dataOffset + srcIdx + 1]) * 0.5f;

                        PixelPair pixel;
                        if (clip) {
                            pixel.ch[CHY0] = (T)((uint16_t)fclampf<Depth>(y0) << OutputShift);
                            pixel.ch[CHY1] = (T)((uint16_t)fclampf<Depth>(y1) << OutputShift);
                            pixel.ch[CHU]  = (T)((uint16_t)fclampf<Depth>(u_avg) << OutputShift);
                            pixel.ch[CHV]  = (T)((uint16_t)fclampf<Depth>(v_avg) << OutputShift);
                        } else {
                            pixel.ch[CHY0] = (T)((uint16_t)y0 << OutputShift);
                            pixel.ch[CHY1] = (T)((uint16_t)y1 << OutputShift);
                            pixel.ch[CHU]  = (T)((uint16_t)u_avg << OutputShift);
                            pixel.ch[CHV]  = (T)((uint16_t)v_avg << OutputShift);
                        }
                        rowPtr[pairIdx] = pixel;
                        srcIdx += 2;
                        imgX += 2;
                    }
                }

                // Handle trailing odd pixel. Incomplete pair has room for only 2 elements (slots 0,1); do not write past row or off buffer.
                if (srcIdx < srcEnd) {
                    const int32_t pairIdx  = imgX / 2;
                    const float   y0       = ch0[dataOffset + srcIdx];
                    const float   u        = ch1[dataOffset + srcIdx];
                    const float   v        = ch2[dataOffset + srcIdx];
                    const bool    lastCol  = (imgX == imageWidth - 1);
                    const bool    oddWidth = (imageWidth & 1) != 0;
                    const bool    onlyTwo  = (oddWidth && lastCol); // incomplete pair: write only CHY0 and CHU

                    T* lastPair = reinterpret_cast<T*>(rowData) + (pairIdx * number_of_elements_per_pair);
                    if (clip) {
                        lastPair[CHY0] = (T)((uint16_t)fclampf<Depth>(y0) << OutputShift);
                        lastPair[CHU]  = (T)((uint16_t)fclampf<Depth>(u) << OutputShift);
                        if (!onlyTwo) {
                            lastPair[CHY1] = (T)((uint16_t)fclampf<Depth>(y0) << OutputShift);
                            lastPair[CHV]  = (T)((uint16_t)fclampf<Depth>(v) << OutputShift);
                        }
                    } else {
                        lastPair[CHY0] = (T)((uint16_t)y0 << OutputShift);
                        lastPair[CHU]  = (T)((uint16_t)u << OutputShift);
                        if (!onlyTwo) {
                            lastPair[CHY1] = (T)((uint16_t)y0 << OutputShift);
                            lastPair[CHV]  = (T)((uint16_t)v << OutputShift);
                        }
                    }
                }
            }

            //4:2:2 packed template - safe path save with boundary checking
            template<typename T, bit_depth Depth, int CHY0, int CHY1, int CHU, int CHV,
                     int OutputShift = 0, typename CellDataType>
            __forceinline__ __device__ void safe_save_yuv_422_packed(uint8_t** dst, const size_t* stride, bool clip,
                                                                     float alpha, const CellDataType& cell,
                                                                     uint32_t nCellX, uint32_t nCellY,
                                                                     int32_t imageWidth, int32_t imageHeight) {
                (void)alpha; // unused for YUV formats

                const float* ch0 = cell.channel_ptr(0); // Y channel
                const float* ch1 = cell.channel_ptr(1); // U channel
                const float* ch2 = cell.channel_ptr(2); // V channel

                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;

#if USE_FAST_PATH
                // Fast path: entire cell within bounds, stride and base pointer aligned for PixelBlock writes
                constexpr size_t packed_align = (cell_width / 2) * 4 * sizeof(T);
                if (basePixelX >= 0 && basePixelX + cell_width - 1 < imageWidth && basePixelY >= 0 &&
                    basePixelY + cell_height - 1 < imageHeight && (stride[0] % packed_align) == 0 &&
                    (reinterpret_cast<uintptr_t>(dst[0]) % packed_align) == 0) {
                    save_yuv_422_packed<T, Depth, CHY0, CHY1, CHU, CHV, OutputShift>(dst, stride, clip, alpha, cell, nCellX, nCellY);
                    return;
                }
#endif

                // Calculate valid pixel range within the cell
                // srcStart: first cell index that maps to a valid image column
                // srcEnd: one past last valid cell index
                const int32_t srcStart = nppdx_max(0, -basePixelX);
                const int32_t srcEnd   = nppdx_min(cell_width, imageWidth - basePixelX);

                if (srcStart >= srcEnd)
                    return; // No valid pixels to write

                // First valid image X coordinate
                const int32_t firstImgX = basePixelX + srcStart;

                uint8_t*     dstPlane  = dst[0];
                const size_t dstStride = stride[0];

                // Process each row
#pragma unroll
                for (int32_t row = 0; row < cell_height; ++row) {
                    const int32_t pixelY = basePixelY + row;
                    if (pixelY >= 0 && pixelY < imageHeight) {
                        uint8_t*      rowDst     = dstPlane + pixelY * dstStride;
                        const int32_t dataOffset = row * cell_width;
                        save_yuv422_row<T, Depth, CHY0, CHY1, CHU, CHV, cell_width, OutputShift>(
                            rowDst, ch0, ch1, ch2, dataOffset, srcStart, srcEnd, firstImgX, imageWidth, clip);
                    }
                }
            } //safe_save_yuv_422_packed

            // YUV422 Packed formats

            template<>
            struct format_backend<packing_format::y210> {
                static constexpr bool is_implemented = true;

                // Y210: YUV 4:2:2 packed, 16-bit words (canonical 10-bit in [15:6]; ingest may use full word).
                // Pattern: Y0 U0 Y1 V0 (channel order: 0, 2, 1, 3)

                // 10-bit data lives in bits [15:6] of each uint16_t. InputShift / OutputShift = 6
                // pushes the int<->float conversion down into the generic helpers, so the cell
                // always holds canonical 10-bit-aligned values [0, 1023] — no wrapper pass, no
                // temp_cell scratch.
                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, const int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_load_yuv_422_packed<uint16_t, 0, 2, 1, 3, /*InputShift=*/6>(
                        src, stride, cell, nCellX, nCellY, imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_save_yuv_422_packed<uint16_t, bit_depth::bpp_10u, 0, 2, 1, 3, /*OutputShift=*/6>(
                        dst, stride, clip, alpha, cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::yuv2> {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, const int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_load_yuv_422_packed<uint8_t, 0, 2, 1, 3>(src, stride, cell, nCellX, nCellY, imageWidth,
                                                                   imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_save_yuv_422_packed<uint8_t, bit_depth::bpp_8u, 0, 2, 1, 3>(
                        dst, stride, clip, alpha, cell, nCellX, nCellY, imageWidth, imageHeight);
                }                
            };

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_INGEST_EXGEST_YUV422_PACKED_FAMILY_HPP
