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

#ifndef NPPDX_DETAIL_BACKEND_INGEST_EXGEST_I42X_PLANAR_FAMILY_HPP
#define NPPDX_DETAIL_BACKEND_INGEST_EXGEST_I42X_PLANAR_FAMILY_HPP

#include <cstdint>
#include "nppdx/detail/utils/force_align.hpp"
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

            // I42x backend

            struct yuv_32f {
                float y, u, v;
            };

            template<uint32_t InputMask, typename T>
            __forceinline__ __device__ float i42x_load_sample(T value) {
                return static_cast<float>(value & InputMask);
            }

            template<typename T, uint32_t OutputMask>
            __forceinline__ __device__ T i42x_store_sample(float value) {
                return static_cast<T>(static_cast<T>(value) & OutputMask);
            }

            template<typename T, bool VDecimate, uint32_t InputMask, typename CellDataType>
            __forceinline__ __device__ static void load_I42x_planar(const uint8_t** src, const size_t* stride,
                                                                    CellDataType& cell, uint32_t nCellX,
                                                                    uint32_t nCellY) {
                float* ch0 = cell.channel_ptr(0);
                float* ch1 = cell.channel_ptr(1);
                float* ch2 = cell.channel_ptr(2);

                constexpr unsigned int cell_width   = CellDataType::cell_width;
                constexpr unsigned int cell_height  = CellDataType::cell_height;
                constexpr unsigned int chroma_width = cell_width / 2;

                struct LumaBlock {
                    union {
                        T                                  ch[cell_width];
                        ForceAlign<cell_width * sizeof(T)> _align;
                    };
                }; // Y - full precision
                struct ChromaBlock {
                    union {
                        T                                    ch[chroma_width];
                        ForceAlign<chroma_width * sizeof(T)> _align;
                    };
                }; // Chroma - half precision

                // Fast path - no boundary checking, assumes cell fits entirely
                const int    chvscale = 1 + uint32_t(!VDecimate); // YUV420P: 1, YUV422P: 2
                const size_t ch0ptr0 =
                    size_t(src[0]) + (size_t)nCellY * cell_height * stride[0] + nCellX * sizeof(LumaBlock);
                const size_t ch1ptr0 =
                    size_t(src[1]) + (size_t)nCellY * chvscale * stride[1] + nCellX * sizeof(ChromaBlock);
                const size_t ch2ptr0 =
                    size_t(src[2]) + (size_t)nCellY * chvscale * stride[2] + nCellX * sizeof(ChromaBlock);

                // Process each row
#pragma unroll
                for (unsigned int row = 0; row < cell_height; ++row) {
                    const size_t       ch0ptrR = ch0ptr0 + row * stride[0];
                    const unsigned int dataOff = row * cell_width;

                    // Load luma
                    LumaBlock block0 = *(LumaBlock*)ch0ptrR;
#pragma unroll
                    for (unsigned int i = 0; i < cell_width; ++i) {
                        ch0[dataOff + i] = i42x_load_sample<InputMask>(block0.ch[i]);
                    }

                    // Load chroma
                    if constexpr (!VDecimate) {
                        // 4:2:2 - each row has its own UV
                        const size_t ch1ptrR = ch1ptr0 + row * stride[1];
                        const size_t ch2ptrR = ch2ptr0 + row * stride[2];
                        ChromaBlock  block1  = *(ChromaBlock*)ch1ptrR;
                        ChromaBlock  block2  = *(ChromaBlock*)ch2ptrR;
#pragma unroll
                        for (unsigned int i = 0; i < chroma_width; ++i) {
                            ch1[dataOff + i * 2]     = i42x_load_sample<InputMask>(block1.ch[i]);
                            ch1[dataOff + i * 2 + 1] = i42x_load_sample<InputMask>(block1.ch[i]);
                            ch2[dataOff + i * 2]     = i42x_load_sample<InputMask>(block2.ch[i]);
                            ch2[dataOff + i * 2 + 1] = i42x_load_sample<InputMask>(block2.ch[i]);
                        }
                    } else {
                        // 4:2:0 - UV shared between rows
                        if (row == 0) {
                            ChromaBlock block1 = *(ChromaBlock*)ch1ptr0;
                            ChromaBlock block2 = *(ChromaBlock*)ch2ptr0;
#pragma unroll
                            for (unsigned int i = 0; i < chroma_width; ++i) {
                                ch1[i * 2]     = i42x_load_sample<InputMask>(block1.ch[i]);
                                ch1[i * 2 + 1] = i42x_load_sample<InputMask>(block1.ch[i]);
                                ch2[i * 2]     = i42x_load_sample<InputMask>(block2.ch[i]);
                                ch2[i * 2 + 1] = i42x_load_sample<InputMask>(block2.ch[i]);
                            }
                        } else {
                            // Copy UV from row 0
#pragma unroll
                            for (unsigned int i = 0; i < cell_width; ++i) {
                                ch1[dataOff + i] = ch1[i];
                                ch2[dataOff + i] = ch2[i];
                            }
                        }
                    }
                }
            } // load_I42x_planar


            template<typename T, bit_depth R, bool VDecimate, uint32_t OutputMask, typename CellDataType>
            __forceinline__ __device__ static void save_I42x_planar(uint8_t** dst, const size_t* stride, bool clip,
                                                                    float alpha, const CellDataType& cell,
                                                                    uint32_t nCellX, uint32_t nCellY) {
                (void)alpha;
                const float* ch0 = cell.channel_ptr(0);
                const float* ch1 = cell.channel_ptr(1);
                const float* ch2 = cell.channel_ptr(2);

                constexpr unsigned int cell_width   = CellDataType::cell_width;
                constexpr unsigned int cell_height  = CellDataType::cell_height;
                constexpr unsigned int chroma_width = cell_width / 2;

                struct LumaBlock {
                    union {
                        T                                  ch[cell_width];
                        ForceAlign<cell_width * sizeof(T)> _align;
                    };
                }; // Y - full precision
                struct ChromaBlock {
                    union {
                        T                                    ch[chroma_width];
                        ForceAlign<chroma_width * sizeof(T)> _align;
                    };
                }; // Chroma - half precision

                constexpr int chvscale = 1 + uint32_t(!VDecimate); // YUV420P: 1, YUV422P: 2
                const size_t  ch0ptr0  = (size_t)dst[0] + nCellY * cell_height * stride[0] + nCellX * sizeof(LumaBlock);
                const size_t  ch1ptr0  = (size_t)dst[1] + nCellY * chvscale * stride[1] + nCellX * sizeof(ChromaBlock);
                const size_t  ch2ptr0  = (size_t)dst[2] + nCellY * chvscale * stride[2] + nCellX * sizeof(ChromaBlock);

                LumaBlock   block0;
                ChromaBlock block1, block2;

                // Process each row
#pragma unroll
                for (unsigned int row = 0; row < cell_height; ++row) {
                    const size_t       ch0ptrR = ch0ptr0 + row * stride[0];
                    const unsigned int dataOff = row * cell_width;

                    // Write luma
                    if (clip) {
#pragma unroll
                        for (unsigned int i = 0; i < cell_width; ++i) {
                            block0.ch[i] = i42x_store_sample<T, OutputMask>(fclampf<R>(ch0[dataOff + i]));
                        }
                    } else {
#pragma unroll
                        for (unsigned int i = 0; i < cell_width; ++i) {
                            block0.ch[i] = i42x_store_sample<T, OutputMask>(ch0[dataOff + i]);
                        }
                    }
                    *(LumaBlock*)ch0ptrR = block0;

                    // Write chroma
                    if constexpr (!VDecimate) {
                        // 4:2:2 - write chroma for every row
                        const size_t ch1ptrR = ch1ptr0 + row * stride[1];
                        const size_t ch2ptrR = ch2ptr0 + row * stride[2];
                        if (clip) {
#pragma unroll
                            for (unsigned int i = 0; i < chroma_width; ++i) {
                                block1.ch[i] = i42x_store_sample<T, OutputMask>(
                                    fclampf<R>((ch1[dataOff + i * 2] + ch1[dataOff + i * 2 + 1]) * 0.5f));
                                block2.ch[i] = i42x_store_sample<T, OutputMask>(
                                    fclampf<R>((ch2[dataOff + i * 2] + ch2[dataOff + i * 2 + 1]) * 0.5f));
                            }
                        } else {
#pragma unroll
                            for (unsigned int i = 0; i < chroma_width; ++i) {
                                block1.ch[i] = i42x_store_sample<T, OutputMask>(
                                    (ch1[dataOff + i * 2] + ch1[dataOff + i * 2 + 1]) * 0.5f);
                                block2.ch[i] = i42x_store_sample<T, OutputMask>(
                                    (ch2[dataOff + i * 2] + ch2[dataOff + i * 2 + 1]) * 0.5f);
                            }
                        }
                        *(ChromaBlock*)ch1ptrR = block1;
                        *(ChromaBlock*)ch2ptrR = block2;
                    } else {
                        // 4:2:0 - only write chroma for first row, averaging 2x2 blocks
                        if (row == 0) {
                            if (clip) {
#pragma unroll
                                for (unsigned int i = 0; i < chroma_width; ++i) {
                                    block1.ch[i] = i42x_store_sample<T, OutputMask>(
                                        fclampf<R>((ch1[i * 2] + ch1[i * 2 + 1] + ch1[cell_width + i * 2] +
                                                    ch1[cell_width + i * 2 + 1]) *
                                                   0.25f));
                                    block2.ch[i] = i42x_store_sample<T, OutputMask>(
                                        fclampf<R>((ch2[i * 2] + ch2[i * 2 + 1] + ch2[cell_width + i * 2] +
                                                    ch2[cell_width + i * 2 + 1]) *
                                                   0.25f));
                                }
                            } else {
#pragma unroll
                                for (unsigned int i = 0; i < chroma_width; ++i) {
                                    block1.ch[i] = i42x_store_sample<T, OutputMask>(
                                        (ch1[i * 2] + ch1[i * 2 + 1] + ch1[cell_width + i * 2] +
                                         ch1[cell_width + i * 2 + 1]) *
                                        0.25f);
                                    block2.ch[i] = i42x_store_sample<T, OutputMask>(
                                        (ch2[i * 2] + ch2[i * 2 + 1] + ch2[cell_width + i * 2] +
                                         ch2[cell_width + i * 2 + 1]) *
                                        0.25f);
                                }
                            }
                            *(ChromaBlock*)ch1ptr0 = block1;
                            *(ChromaBlock*)ch2ptr0 = block2;
                        }
                    }
                }
            } // save_I42x_planar

            // ===== GENERIC I42X PLANAR FUNCTIONS (for both YUV420P and YUV422P) =====

            // Safe path: needLeft, one edge per row, single loop per row, conservative replicate policy (odd height/width).
            template<typename T, bool VDecimate, uint32_t InputMask = 0xFFFFFFFFu, typename CellDataType>
            __forceinline__ __device__ void safe_load_I42x_planar(const uint8_t** src, const size_t* stride,
                                                                  CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                  int32_t imageWidth, int32_t imageHeight) {
                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;

#if USE_FAST_PATH
                constexpr size_t luma_align   = cell_width * sizeof(T);
                constexpr size_t chroma_align = (cell_width / 2) * sizeof(T);
                if (basePixelX >= 0 && basePixelX + cell_width - 1 < imageWidth && basePixelY >= 0 &&
                    basePixelY + cell_height - 1 < imageHeight && (stride[0] % luma_align) == 0 &&
                    (stride[1] % chroma_align) == 0 && (stride[2] % chroma_align) == 0 &&
                    (reinterpret_cast<uintptr_t>(src[0]) % luma_align) == 0 &&
                    (reinterpret_cast<uintptr_t>(src[1]) % chroma_align) == 0 &&
                    (reinterpret_cast<uintptr_t>(src[2]) % chroma_align) == 0) {
                    load_I42x_planar<T, VDecimate, InputMask>(src, stride, cell, (uint32_t)nCellX, (uint32_t)nCellY);
                    return;
                }
#endif

                float*        ch0        = cell.channel_ptr(0);
                float*        ch1        = cell.channel_ptr(1);
                float*        ch2        = cell.channel_ptr(2);
                const int     chvshift   = uint32_t(VDecimate);
                const int32_t maxChromaY = VDecimate ? (imageHeight / 2) - 1 : imageHeight - 1;
                const int32_t maxChromaX = (imageWidth / 2) - 1;
                const bool    needLeft   = (basePixelX < 0);

                // Right edge (for halo): odd width = last Y + previous chroma; even = last pixel.
                const int32_t rightLumaIdx = (imageWidth <= 1) ? 0 : imageWidth - 1;
                const int32_t rightChromaX =
                    (imageWidth <= 1) ? 0 : nppdx_max(0, (imageWidth & 1) ? (imageWidth - 2) / 2 : maxChromaX);

#pragma unroll
                for (int32_t row = 0; row < cell_height; ++row) {
                    const int32_t pixelY = basePixelY + row;
                    // Luma: read from current row (including last row when height is odd).
                    // 4:2:0 odd height: chroma has no row for last luma row; clone from previous (maxChromaY).
                    const int32_t lumaRow = nppdx_max(0, nppdx_min(pixelY, (int32_t)imageHeight - 1));
                    const int32_t chromaY = nppdx_min(lumaRow >> chvshift, maxChromaY);
                    const int32_t dataOff = row * cell_width;

                    const T* ch0ptr = (const T*)(src[0] + (size_t)lumaRow * stride[0]);
                    const T* ch1ptr = (const T*)(src[1] + (size_t)chromaY * stride[1]);
                    const T* ch2ptr = (const T*)(src[2] + (size_t)chromaY * stride[2]);

                    float edgeY, edgeU, edgeV;
                    if (needLeft) {
                        edgeY = i42x_load_sample<InputMask>(ch0ptr[0]);
                        edgeU = i42x_load_sample<InputMask>(ch1ptr[0]);
                        edgeV = i42x_load_sample<InputMask>(ch2ptr[0]);
                    } else {
                        edgeY = i42x_load_sample<InputMask>(ch0ptr[rightLumaIdx]);
                        edgeU = i42x_load_sample<InputMask>(ch1ptr[rightChromaX]);
                        edgeV = i42x_load_sample<InputMask>(ch2ptr[rightChromaX]);
                    }

                    // Single loop: in bounds -> fetch. Odd width last pixel: keep new Y, clone chroma from previous (planar).
#pragma unroll
                    for (int32_t i = 0; i < cell_width; ++i) {
                        const int32_t imgX = basePixelX + i;
                        if (imgX >= 0 && imgX < imageWidth) {
                            ch0[dataOff + i]      = i42x_load_sample<InputMask>(ch0ptr[imgX]);
                            const int32_t chromaX = (imgX == imageWidth - 1 && (imageWidth & 1))
                                                        ? (imgX - 1) / 2
                                                        : nppdx_min(imgX / 2, maxChromaX);
                            ch1[dataOff + i]      = i42x_load_sample<InputMask>(ch1ptr[chromaX]);
                            ch2[dataOff + i]      = i42x_load_sample<InputMask>(ch2ptr[chromaX]);
                        } else {
                            ch0[dataOff + i] = edgeY;
                            ch1[dataOff + i] = edgeU;
                            ch2[dataOff + i] = edgeV;
                        }
                    }
                }
            }


            template<typename T, bit_depth R, bool VDecimate, uint32_t OutputMask = 0xFFFFFFFFu, typename CellDataType>
            __forceinline__ __device__ void safe_save_I42x_planar(uint8_t** dst, const size_t* stride, bool clip,
                                                                  float alpha, const CellDataType& cell,
                                                                  const int32_t nCellX, const int32_t nCellY,
                                                                  const uint32_t imageWidth,
                                                                  const uint32_t imageHeight) {
                (void)alpha;
                const float* ch0 = cell.channel_ptr(0);
                const float* ch1 = cell.channel_ptr(1);
                const float* ch2 = cell.channel_ptr(2);

                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;

#if USE_FAST_PATH
                // Fast path requires: bounds check AND stride/pointer alignment for LumaBlock/ChromaBlock writes
                constexpr size_t luma_align   = cell_width * sizeof(T);
                constexpr size_t chroma_align = (cell_width / 2) * sizeof(T);
                if (basePixelX >= 0 && basePixelX + cell_width - 1 < imageWidth && basePixelY >= 0 &&
                    basePixelY + cell_height - 1 < imageHeight && (stride[0] % luma_align) == 0 &&
                    (stride[1] % chroma_align) == 0 && (stride[2] % chroma_align) == 0 &&
                    (reinterpret_cast<uintptr_t>(dst[0]) % luma_align) == 0 &&
                    (reinterpret_cast<uintptr_t>(dst[1]) % chroma_align) == 0 &&
                    (reinterpret_cast<uintptr_t>(dst[2]) % chroma_align) == 0) {
                    save_I42x_planar<T, R, VDecimate, OutputMask>(dst, stride, clip, alpha, cell, (uint32_t)nCellX,
                                                                   (uint32_t)nCellY);
                    return;
                }
#endif

                // Calculate chroma plane dimensions
                const int32_t chromaWidth  = imageWidth / 2;
                const int32_t chromaHeight = VDecimate ? (imageHeight / 2) : imageHeight;
                const int32_t baseChromaX  = nppdx_min((basePixelX / 2), chromaWidth - 1);
                const int32_t baseChromaY  = VDecimate ? nppdx_min((basePixelY / 2), chromaHeight - 1) : basePixelY;
                const int32_t maxChromaX   = chromaWidth - 1;

                const size_t ch0ptr0 =
                    (size_t)dst[0] + nCellY * cell_height * stride[0] + nCellX * sizeof(T) * cell_width;
                const size_t ch1ptr0 = (size_t)dst[1] + baseChromaY * stride[1] + baseChromaX * sizeof(T);
                const size_t ch2ptr0 = (size_t)dst[2] + baseChromaY * stride[2] + baseChromaX * sizeof(T);

                const int validPixels = nppdx_min(cell_width, (int32_t)imageWidth - basePixelX);
                if (validPixels <= 0)
                    return;

                    // Process each row
#pragma unroll
                for (int32_t row = 0; row < cell_height; ++row) {
                    const int32_t pixelY  = basePixelY + row;
                    const int32_t dataOff = row * cell_width;
                    const size_t  ch0ptrR = ch0ptr0 + row * stride[0];

                    // Build pixel array for this row
                    yuv_32f pixels[cell_width];
#pragma unroll
                    for (int i = 0; i < cell_width; ++i) {
                        pixels[i] = {ch0[dataOff + i], ch1[dataOff + i], ch2[dataOff + i]};
                    }

                    // 4:2:0 vertical decimation: average UV with next row (if first row)
                    if constexpr (VDecimate) {
                        if (row == 0 && basePixelY + 1 < (int32_t)imageHeight) {
#pragma unroll
                            for (int i = 0; i < cell_width; ++i) {
                                pixels[i].u = (ch1[i] + ch1[cell_width + i]) * 0.5f;
                                pixels[i].v = (ch2[i] + ch2[cell_width + i]) * 0.5f;
                            }
                        }
                    }

                    if ((pixelY >= 0) && (pixelY < (int32_t)imageHeight)) {
                        if (clip) {
#pragma unroll
                            for (int i = 0; i < cell_width; ++i) {
                                if (i < validPixels) {
                                    pixels[i].y = fclampf<R>(pixels[i].y);
                                    pixels[i].u = fclampf<R>(pixels[i].u);
                                    pixels[i].v = fclampf<R>(pixels[i].v);
                                }
                            }
                        }

                        // Write Luma
#pragma unroll
                        for (int i = 0; i < cell_width; ++i) {
                            if (i < validPixels)
                                ((T*)ch0ptrR)[i] = i42x_store_sample<T, OutputMask>(pixels[i].y);
                        }

                        // Write decimated chroma (only for first row in 4:2:0, all rows in 4:2:2)
                        // Key: only write chroma at image-level pair boundaries (even image X)
                        const int32_t firstEvenOffset = (basePixelX % 2);

                        if constexpr (!VDecimate) {
                            // 4:2:2 - write chroma for every row
                            const size_t ch1ptrR = ch1ptr0 + row * stride[1];
                            const size_t ch2ptrR = ch2ptr0 + row * stride[2];
#pragma unroll
                            for (int i = firstEvenOffset; i < cell_width; i += 2) {
                                if (i < validPixels) {
                                    const int32_t imgX          = basePixelX + i;
                                    const int32_t globalChromaX = imgX / 2;
                                    // Only write if within chroma bounds
                                    if (globalChromaX <= maxChromaX) {
                                        const int32_t localIdx = globalChromaX - baseChromaX;
                                        if (i + 1 < validPixels) {
                                            // Full pair - average both pixels
                                            ((T*)ch1ptrR)[localIdx] =
                                                i42x_store_sample<T, OutputMask>((pixels[i].u + pixels[i + 1].u) * 0.5f);
                                            ((T*)ch2ptrR)[localIdx] =
                                                i42x_store_sample<T, OutputMask>((pixels[i].v + pixels[i + 1].v) * 0.5f);
                                        } else {
                                            // Single pixel at cell edge - write its chroma directly
                                            ((T*)ch1ptrR)[localIdx] = i42x_store_sample<T, OutputMask>(pixels[i].u);
                                            ((T*)ch2ptrR)[localIdx] = i42x_store_sample<T, OutputMask>(pixels[i].v);
                                        }
                                    }
                                }
                            }
                        } else {
                            // 4:2:0 - only write chroma for first row, and only if within vertical chroma bounds
                            const int32_t globalChromaY = basePixelY / 2;
                            if (row == 0 && globalChromaY <= (chromaHeight - 1)) {
#pragma unroll
                                for (int i = firstEvenOffset; i < cell_width; i += 2) {
                                    if (i < validPixels) {
                                        const int32_t imgX          = basePixelX + i;
                                        const int32_t globalChromaX = imgX / 2;
                                        // Only write if within chroma bounds
                                        if (globalChromaX <= maxChromaX) {
                                            const int32_t localIdx = globalChromaX - baseChromaX;
                                            if (i + 1 < validPixels) {
                                                // Full pair - average both pixels
                                                ((T*)ch1ptr0)[localIdx] =
                                                    i42x_store_sample<T, OutputMask>((pixels[i].u + pixels[i + 1].u) * 0.5f);
                                                ((T*)ch2ptr0)[localIdx] =
                                                    i42x_store_sample<T, OutputMask>((pixels[i].v + pixels[i + 1].v) * 0.5f);
                                            } else {
                                                // Single pixel at cell edge - write its chroma directly
                                                ((T*)ch1ptr0)[localIdx] = i42x_store_sample<T, OutputMask>(pixels[i].u);
                                                ((T*)ch2ptr0)[localIdx] = i42x_store_sample<T, OutputMask>(pixels[i].v);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // I42x formats 

            template<>
            struct format_backend<packing_format::yuv420p> {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                 int32_t imageWidth, int32_t imageHeight) {
                    safe_load_I42x_planar<uint8_t, true>(src, stride, cell, nCellX, nCellY, imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_save_I42x_planar<uint8_t, bit_depth::bpp_8u, true>(dst, stride, clip, alpha, cell, nCellX,
                                                                            nCellY, imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::yuv420p10> {
                static constexpr bool is_implemented = true;

                // Fully planar YUV 4:2:0, 10-bit data stored in the LSBs of uint16_t (bits [9:0]),
                // matching FFmpeg's yuv420p10le. InputMask / OutputMask = 0x3FF keep the cell in
                // canonical 10-bit values [0, 1023].
                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                 int32_t imageWidth, int32_t imageHeight) {
                    safe_load_I42x_planar<uint16_t, /*VDecimate=*/true, /*InputMask=*/0x3FF>(
                        src, stride, cell, nCellX, nCellY, imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_save_I42x_planar<uint16_t, bit_depth::bpp_10u, /*VDecimate=*/true, /*OutputMask=*/0x3FF>(
                        dst, stride, clip, alpha, cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::yuv422p> {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                 int32_t imageWidth, int32_t imageHeight) {
                    safe_load_I42x_planar<uint8_t, false>(src, stride, cell, nCellX, nCellY, imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell, int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_save_I42x_planar<uint8_t, bit_depth::bpp_8u, false>(dst, stride, clip, alpha, cell, nCellX,
                                                                             nCellY, imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::yuv422p10> {
                static constexpr bool is_implemented = true;

                // Fully planar YUV 4:2:2, 10-bit data stored in the LSBs of uint16_t (bits [9:0]),
                // matching FFmpeg's yuv422p10le. Chroma keeps full vertical resolution
                // (VDecimate = false); InputMask / OutputMask = 0x3FF.
                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                 int32_t imageWidth, int32_t imageHeight) {
                    safe_load_I42x_planar<uint16_t, /*VDecimate=*/false, /*InputMask=*/0x3FF>(
                        src, stride, cell, nCellX, nCellY, imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_save_I42x_planar<uint16_t, bit_depth::bpp_10u, /*VDecimate=*/false, /*OutputMask=*/0x3FF>(
                        dst, stride, clip, alpha, cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_INGEST_EXGEST_I42X_PLANAR_FAMILY_HPP