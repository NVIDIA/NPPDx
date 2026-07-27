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

#ifndef NPPDX_DETAIL_BACKEND_INGEST_EXGEST_RGB10_FORMAT_BACKEND_HPP
#define NPPDX_DETAIL_BACKEND_INGEST_EXGEST_RGB10_FORMAT_BACKEND_HPP

#include <cstdint>
#include "nppdx/detail/backend/constants.hpp"
#include "nppdx/detail/backend/formats_impl/format_backend.hpp"
#include "nppdx/detail/backend/formats_impl/clamp_utils.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {

// Fast path control (from NPP)
#ifndef USE_FAST_PATH
#    define USE_FAST_PATH 1
#endif

            template<>
            struct format_backend<packing_format::rgb10> {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void fast_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY) {
                    float*         ch0       = cell.channel_ptr(0);
                    float*         ch1       = cell.channel_ptr(1);
                    float*         ch2       = cell.channel_ptr(2);
                    const uint8_t* srcPlane  = src[0];
                    const size_t   srcStride = stride[0];

                    constexpr unsigned int cell_width  = CellDataType::cell_cfg.size.x;
                    constexpr unsigned int cell_height = CellDataType::cell_cfg.size.y;

                    // Fast path - no boundary checking, assumes cell fits entirely
                    const size_t src0 =
                        (size_t)srcPlane + nCellY * cell_height * srcStride + nCellX * cell_width * sizeof(uint32_t);

                    // Load pixels row by row (each pixel is 32-bit R210)
#pragma unroll
                    for (unsigned int row = 0; row < cell_height; ++row) {
                        const size_t       srcRow  = src0 + row * srcStride;
                        const uint32_t*    pixels  = (const uint32_t*)srcRow;
                        const unsigned int dataOff = row * cell_width;

#pragma unroll
                        for (unsigned int col = 0; col < cell_width; ++col) {
                            ch0[dataOff + col] = float((pixels[col] >> 20) & 0x3FF); // R
                            ch1[dataOff + col] = float((pixels[col] >> 10) & 0x3FF); // G
                            ch2[dataOff + col] = float(pixels[col] & 0x3FF);         // B
                        }
                    }
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void fast_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY) {
                    (void)alpha;
                    const float* ch0 = cell.channel_ptr(0);
                    const float* ch1 = cell.channel_ptr(1);
                    const float* ch2 = cell.channel_ptr(2);

                    constexpr unsigned int cell_width  = CellDataType::cell_cfg.size.x;
                    constexpr unsigned int cell_height = CellDataType::cell_cfg.size.y;

                    uint8_t*     dstPlane  = dst[0];
                    const size_t dstStride = stride[0];
                    const size_t dst0 =
                        (size_t)dstPlane + nCellY * cell_height * dstStride + nCellX * cell_width * sizeof(uint32_t);

                    // Save pixels row by row
#pragma unroll
                    for (unsigned int row = 0; row < cell_height; ++row) {
                        const size_t       dstRow  = dst0 + row * dstStride;
                        uint32_t*          pixels  = (uint32_t*)dstRow;
                        const unsigned int dataOff = row * cell_width;

                        if (clip) {
#pragma unroll
                            for (unsigned int col = 0; col < cell_width; ++col) {
                                pixels[col] = (uint32_t(fclampf<bit_depth::bpp_10u>(ch0[dataOff + col])) << 20) |
                                              (uint32_t(fclampf<bit_depth::bpp_10u>(ch1[dataOff + col])) << 10) |
                                              uint32_t(fclampf<bit_depth::bpp_10u>(ch2[dataOff + col]));
                            }
                        } else {
#pragma unroll
                            for (unsigned int col = 0; col < cell_width; ++col) {
                                pixels[col] = (uint32_t(ch0[dataOff + col]) << 20) |
                                              (uint32_t(ch1[dataOff + col]) << 10) | uint32_t(ch2[dataOff + col]);
                            }
                        }
                    }
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, const int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    float* ch0 = cell.channel_ptr(0);
                    float* ch1 = cell.channel_ptr(1);
                    float* ch2 = cell.channel_ptr(2);

                    const int32_t basePixelX = nCellX * CellDataType::cell_cfg.size.x;
                    const int32_t basePixelY = nCellY * CellDataType::cell_cfg.size.y;

#if USE_FAST_PATH
                    if (basePixelX >= 0 && (basePixelX + CellDataType::cell_cfg.size.x - 1) < imageWidth &&
                        basePixelY >= 0 && (basePixelY + CellDataType::cell_cfg.size.y - 1) < imageHeight) {
                        fast_load(src, stride, cell, (uint32_t)nCellX, (uint32_t)nCellY);
                        return;
                    }
#endif

                    const uint8_t*    srcPlane    = src[0];
                    const size_t      srcStride   = stride[0];
                    constexpr int32_t cell_width  = CellDataType::cell_cfg.size.x;
                    constexpr int32_t cell_height = CellDataType::cell_cfg.size.y;

                    // Process each row
#pragma unroll
                    for (int32_t row = 0; row < cell_height; ++row) {
                        const int32_t   pixelY  = basePixelY + row;
                        const int32_t   clampY  = nppdx_max(0, nppdx_min(pixelY, (int32_t)imageHeight - 1));
                        const int32_t   dataOff = row * cell_width;
                        const uint32_t* rowPtr  = (const uint32_t*)(srcPlane + clampY * srcStride);

                        // Get edge pixels for clamping
                        const uint32_t leftEdgePixel  = rowPtr[0];
                        const uint32_t rightEdgePixel = rowPtr[imageWidth - 1];
                        const float    leftR          = float((leftEdgePixel >> 20) & 0x3FF);
                        const float    leftG          = float((leftEdgePixel >> 10) & 0x3FF);
                        const float    leftB          = float(leftEdgePixel & 0x3FF);
                        const float    rightR         = float((rightEdgePixel >> 20) & 0x3FF);
                        const float    rightG         = float((rightEdgePixel >> 10) & 0x3FF);
                        const float    rightB         = float(rightEdgePixel & 0x3FF);

                        // Step 1: Pre-fill with edge values (fully unrollable)
#pragma unroll
                        for (int i = 0; i < cell_width; ++i) {
                            const int32_t imgX = basePixelX + i;
                            // Use left edge for pixels off left, right edge for pixels off right
                            ch0[dataOff + i] = (imgX < 0) ? leftR : rightR;
                            ch1[dataOff + i] = (imgX < 0) ? leftG : rightG;
                            ch2[dataOff + i] = (imgX < 0) ? leftB : rightB;
                        }

                        // Step 2: Overwrite valid pixels (fully unrollable with conditional)
#pragma unroll
                        for (int i = 0; i < cell_width; ++i) {
                            const int32_t imgX = basePixelX + i;
                            if (imgX >= 0 && imgX < (int32_t)imageWidth) {
                                const uint32_t pixel = rowPtr[imgX];
                                ch0[dataOff + i]     = float((pixel >> 20) & 0x3FF);
                                ch1[dataOff + i]     = float((pixel >> 10) & 0x3FF);
                                ch2[dataOff + i]     = float(pixel & 0x3FF);
                            }
                        }
                    }
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    (void)alpha;
                    const float* ch0 = cell.channel_ptr(0);
                    const float* ch1 = cell.channel_ptr(1);
                    const float* ch2 = cell.channel_ptr(2);

                    const int32_t basePixelX = nCellX * CellDataType::cell_cfg.size.x;
                    const int32_t basePixelY = nCellY * CellDataType::cell_cfg.size.y;

#if USE_FAST_PATH
                    if (basePixelX >= 0 && (basePixelX + CellDataType::cell_cfg.size.x - 1) < imageWidth &&
                        basePixelY >= 0 && (basePixelY + CellDataType::cell_cfg.size.y - 1) < imageHeight) {
                        fast_save(dst, stride, clip, alpha, cell, (uint32_t)nCellX, (uint32_t)nCellY);
                        return;
                    }
#endif

                    uint8_t*          dstPlane    = dst[0];
                    const size_t      dstStride   = stride[0];
                    constexpr int32_t cell_width  = CellDataType::cell_cfg.size.x;
                    constexpr int32_t cell_height = CellDataType::cell_cfg.size.y;
                    const size_t      dst0 =
                        (size_t)dstPlane + nCellY * cell_height * dstStride + nCellX * cell_width * sizeof(uint32_t);

                    const int32_t validPixels = nppdx_min(cell_width, (int32_t)imageWidth - basePixelX);
                    if (validPixels <= 0)
                        return;

                    // Process each row
#pragma unroll
                    for (int32_t row = 0; row < cell_height; ++row) {
                        const int32_t pixelY = basePixelY + row;
                        if ((pixelY >= 0) && (pixelY < (int32_t)imageHeight)) {
                            const size_t  dstRow   = dst0 + row * dstStride;
                            const int32_t dataOff  = row * cell_width;
                            uint32_t*     pixelPtr = (uint32_t*)dstRow;

#pragma unroll
                            for (int i = 0; i < validPixels; ++i) {
                                if (clip) {
                                    pixelPtr[i] = (uint32_t(fclampf<bit_depth::bpp_10u>(ch0[dataOff + i])) << 20) |
                                                  (uint32_t(fclampf<bit_depth::bpp_10u>(ch1[dataOff + i])) << 10) |
                                                  uint32_t(fclampf<bit_depth::bpp_10u>(ch2[dataOff + i]));
                                } else {
                                    pixelPtr[i] = (uint32_t(ch0[dataOff + i]) << 20) |
                                                  (uint32_t(ch1[dataOff + i]) << 10) | uint32_t(ch2[dataOff + i]);
                                }
                            }
                        }
                    }
                }
            };

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_INGEST_EXGEST_RGB10_FORMAT_BACKEND_HPP