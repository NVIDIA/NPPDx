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

#ifndef NPPDX_DETAIL_BACKEND_INGEST_EXGEST_PLANAR_FAMILY_HPP
#define NPPDX_DETAIL_BACKEND_INGEST_EXGEST_PLANAR_FAMILY_HPP

#include <cstdint>
#include "nppdx/detail/utils/force_align.hpp"
#include "nppdx/detail/backend/constants.hpp"
#include "nppdx/detail/backend/formats_impl/format_backend.hpp"
#include "nppdx/detail/backend/formats_impl/clamp_utils.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {
            
            // Planar backend

            template<typename T, typename CellDataType>
            struct PlanarBase {

                static constexpr unsigned int cell_width  = CellDataType::config_type::CellSize::X;
                static constexpr unsigned int cell_height = CellDataType::config_type::CellSize::Y;
                
                struct BaseBlock {
                    union {
                        T                                  value[cell_width];
                        ForceAlign<cell_width * sizeof(T)> _align;
                    };
                };

                __forceinline__ __device__ static bool use_fast_path(
                    const uint8_t* r_plane, const size_t r_stride,
                    const uint8_t* g_plane, const size_t g_stride,
                    const uint8_t* b_plane, const size_t b_stride,
                    const int32_t base_pixel_x, const int32_t base_pixel_y,
                    const int32_t image_width, const int32_t image_height
                ) {

                    // Safe path if the cell isn't entirely inside the image.
                    if (base_pixel_x < 0 || base_pixel_y < 0 ||
                        base_pixel_x + static_cast<int32_t>(cell_width) > image_width ||
                        base_pixel_y + static_cast<int32_t>(cell_height) > image_height) {
                        return false;
                    }

                    // The fast path reads/writes a whole BaseBlock (cell_width samples) at once, so each
                    // plane's base pointer and row stride must be aligned to the cell row size. The cell
                    // column offset (base_pixel_x * sizeof(T) = cell_x * cell_width * sizeof(T)) is always
                    // a multiple of this, so checking base + stride is sufficient.
                    constexpr size_t row_align = cell_width * sizeof(T);
                    if ((r_stride % row_align) != 0 || (g_stride % row_align) != 0 || (b_stride % row_align) != 0) {
                        return false;
                    }
                    if ((reinterpret_cast<uintptr_t>(r_plane) % row_align) != 0 ||
                        (reinterpret_cast<uintptr_t>(g_plane) % row_align) != 0 ||
                        (reinterpret_cast<uintptr_t>(b_plane) % row_align) != 0) {
                        return false;
                    }

                    return true;
                };

            };

            template<typename T, typename CellDataType, uint32_t InputMask = 0xFFFFFFFFu>
            struct PlanarLoader {

                using Base                                = PlanarBase<T, CellDataType>;
                using LoadBlock                           = typename Base::BaseBlock;
                static constexpr unsigned int cell_width  = CellDataType::config_type::CellSize::X;
                static constexpr unsigned int cell_height = CellDataType::config_type::CellSize::Y;

                __forceinline__ __device__ static float to_float(T value) {
                    return static_cast<float>(value & InputMask);
                }

                __forceinline__ __device__ static void load(const uint8_t** src, const size_t* stride,
                                                            CellDataType& cell, const int32_t cell_x,
                                                            const int32_t cell_y, const int32_t image_width,
                                                            const int32_t image_height) {
                    load(src[0], src[1], src[2], stride[0], stride[1], stride[2], cell, cell_x, cell_y, image_width, image_height);
                }

            private:

                __forceinline__ __device__ static void load(const uint8_t* r_plane, const uint8_t* g_plane, const uint8_t* b_plane,
                                                            const size_t r_stride, const size_t g_stride, const size_t b_stride,
                                                            CellDataType& cell, const int32_t cell_x, const int32_t cell_y,
                                                            const int32_t image_width, const int32_t image_height) {
                    int32_t base_pixel_x = cell_x * cell_width;
                    int32_t base_pixel_y = cell_y * cell_height;

                    float* ch_R = cell.channel_ptr(0);
                    float* ch_G = cell.channel_ptr(1);
                    float* ch_B = cell.channel_ptr(2);

                    if (Base::use_fast_path(r_plane, r_stride, g_plane, g_stride, b_plane, b_stride, base_pixel_x, base_pixel_y, image_width, image_height)) {
                        fast_load(r_plane, g_plane, b_plane, r_stride, g_stride, b_stride, base_pixel_x, base_pixel_y,
                                  ch_R, ch_G, ch_B);
                    }
                    else {
                        safe_load(r_plane, g_plane, b_plane, r_stride, g_stride, b_stride, base_pixel_x, base_pixel_y,
                                  image_width, image_height, ch_R, ch_G, ch_B);
                    }
                };

                __forceinline__ __device__ static void fast_load(const uint8_t* r_plane, const uint8_t* g_plane, const uint8_t* b_plane,
                                                                 const size_t r_stride, const size_t g_stride, const size_t b_stride,
                                                                 const int32_t base_pixel_x, const int32_t base_pixel_y,
                                                                 float* ch_R, float* ch_G, float* ch_B) {
                    // Cell fits entirely inside the image: read one aligned BaseBlock (cell_width
                    // samples) per plane per row. use_fast_path has verified base + stride alignment.
                    const size_t r_base = static_cast<size_t>(base_pixel_y) * r_stride + static_cast<size_t>(base_pixel_x) * sizeof(T);
                    const size_t g_base = static_cast<size_t>(base_pixel_y) * g_stride + static_cast<size_t>(base_pixel_x) * sizeof(T);
                    const size_t b_base = static_cast<size_t>(base_pixel_y) * b_stride + static_cast<size_t>(base_pixel_x) * sizeof(T);
#pragma unroll
                    for (int row = 0; row < cell_height; ++row) {
                        const LoadBlock loaded_R = *reinterpret_cast<const LoadBlock*>(r_plane + r_base + static_cast<size_t>(row) * r_stride);
                        const LoadBlock loaded_G = *reinterpret_cast<const LoadBlock*>(g_plane + g_base + static_cast<size_t>(row) * g_stride);
                        const LoadBlock loaded_B = *reinterpret_cast<const LoadBlock*>(b_plane + b_base + static_cast<size_t>(row) * b_stride);
                        const int off = row * cell_width;
#pragma unroll
                        for (int col = 0; col < cell_width; ++col) {
                            ch_R[off + col] = to_float(loaded_R.value[col]);
                            ch_G[off + col] = to_float(loaded_G.value[col]);
                            ch_B[off + col] = to_float(loaded_B.value[col]);
                        }
                    }
                };

                __forceinline__ __device__ static void safe_load(const uint8_t* r_plane, const uint8_t* g_plane, const uint8_t* b_plane,
                                                                 const size_t r_stride, const size_t g_stride, const size_t b_stride,
                                                                 const int32_t base_pixel_x, const int32_t base_pixel_y,
                                                                 const int32_t image_width, const int32_t image_height,
                                                                 float* ch_R, float* ch_G, float* ch_B) {
                    // Degenerate image: leave the cell zeroed.
                    if (image_width <= 0 || image_height <= 0) {
#pragma unroll
                        for (int i = 0; i < cell_width * cell_height; ++i) {
                            ch_R[i] = 0.0f;
                            ch_G[i] = 0.0f;
                            ch_B[i] = 0.0f;
                        }
                        return;
                    }

                    // Clamp out-of-bounds reads to the image edge (replicate halo).
#pragma unroll
                    for (int row = 0; row < cell_height; ++row) {
                        const int32_t clampY = nppdx::clamp_value(base_pixel_y + row, 0, image_height - 1);
                        const T* r_row = reinterpret_cast<const T*>(r_plane + static_cast<size_t>(clampY) * r_stride);
                        const T* g_row = reinterpret_cast<const T*>(g_plane + static_cast<size_t>(clampY) * g_stride);
                        const T* b_row = reinterpret_cast<const T*>(b_plane + static_cast<size_t>(clampY) * b_stride);
                        const int off = row * cell_width;
#pragma unroll
                        for (int col = 0; col < cell_width; ++col) {
                            const int32_t clampX = nppdx::clamp_value(base_pixel_x + col, 0, image_width - 1);
                            ch_R[off + col] = to_float(r_row[clampX]);
                            ch_G[off + col] = to_float(g_row[clampX]);
                            ch_B[off + col] = to_float(b_row[clampX]);
                        }
                    }
                };
            };

            template<typename T, bit_depth R, typename CellDataType, uint32_t OutputMask = 0xFFFFFFFFu>
            struct PlanarSaver {

                using Base                                = PlanarBase<T, CellDataType>;
                using SaveBlock                           = typename Base::BaseBlock;
                static constexpr unsigned int cell_width  = CellDataType::config_type::CellSize::X;
                static constexpr unsigned int cell_height = CellDataType::config_type::CellSize::Y;

                __forceinline__ __device__ static T to_sample(float value, const bool clip) {
                    return static_cast<T>(static_cast<T>(clip ? fclampf<R>(value) : value) & OutputMask);
                }

                __forceinline__ __device__ static void save(uint8_t** dst, const size_t* stride, const bool clip,
                                                            const CellDataType& cell, const int32_t cell_x,
                                                            const int32_t cell_y, const int32_t image_width,
                                                            const int32_t image_height) {
                    save(dst[0], dst[1], dst[2], stride[0], stride[1], stride[2], clip, cell, cell_x, cell_y, image_width, image_height);
                }

            private:

                __forceinline__ __device__ static void save(uint8_t* r_plane, uint8_t* g_plane, uint8_t* b_plane,
                                                            const size_t r_stride, const size_t g_stride, const size_t b_stride,
                                                            const bool clip, const CellDataType& cell,
                                                            const int32_t cell_x, const int32_t cell_y,
                                                            const int32_t image_width, const int32_t image_height) {
                    int32_t base_pixel_x = cell_x * cell_width;
                    int32_t base_pixel_y = cell_y * cell_height;

                    const float* ch_R = cell.channel_ptr(0);
                    const float* ch_G = cell.channel_ptr(1);
                    const float* ch_B = cell.channel_ptr(2);

                    if (Base::use_fast_path(r_plane, r_stride, g_plane, g_stride, b_plane, b_stride, base_pixel_x, base_pixel_y, image_width, image_height)) {
                        fast_save(r_plane, g_plane, b_plane, r_stride, g_stride, b_stride, clip, base_pixel_x,
                                  base_pixel_y, ch_R, ch_G, ch_B);
                    }
                    else {
                        safe_save(r_plane, g_plane, b_plane, r_stride, g_stride, b_stride, clip, base_pixel_x,
                                  base_pixel_y, image_width, image_height, ch_R, ch_G, ch_B);
                    }
                };

                __forceinline__ __device__ static void fast_save(uint8_t* r_plane, uint8_t* g_plane, uint8_t* b_plane,
                                                                 const size_t r_stride, const size_t g_stride, const size_t b_stride,
                                                                 const bool clip, const int32_t base_pixel_x, const int32_t base_pixel_y,
                                                                 const float* ch_R, const float* ch_G, const float* ch_B) {
                    // Cell fits entirely inside the image: fill one aligned BaseBlock (cell_width
                    // samples) per plane per row and store it. use_fast_path has verified alignment.
                    const size_t r_base = static_cast<size_t>(base_pixel_y) * r_stride + static_cast<size_t>(base_pixel_x) * sizeof(T);
                    const size_t g_base = static_cast<size_t>(base_pixel_y) * g_stride + static_cast<size_t>(base_pixel_x) * sizeof(T);
                    const size_t b_base = static_cast<size_t>(base_pixel_y) * b_stride + static_cast<size_t>(base_pixel_x) * sizeof(T);
#pragma unroll
                    for (int row = 0; row < cell_height; ++row) {
                        SaveBlock store_R, store_G, store_B;
                        const int off = row * cell_width;
#pragma unroll
                        for (int col = 0; col < cell_width; ++col) {
                            store_R.value[col] = to_sample(ch_R[off + col], clip);
                            store_G.value[col] = to_sample(ch_G[off + col], clip);
                            store_B.value[col] = to_sample(ch_B[off + col], clip);
                        }
                        *reinterpret_cast<SaveBlock*>(r_plane + r_base + static_cast<size_t>(row) * r_stride) = store_R;
                        *reinterpret_cast<SaveBlock*>(g_plane + g_base + static_cast<size_t>(row) * g_stride) = store_G;
                        *reinterpret_cast<SaveBlock*>(b_plane + b_base + static_cast<size_t>(row) * b_stride) = store_B;
                    }
                };

                __forceinline__ __device__ static void safe_save(uint8_t* r_plane, uint8_t* g_plane, uint8_t* b_plane,
                                                                 const size_t r_stride, const size_t g_stride, const size_t b_stride,
                                                                 const bool clip, const int32_t base_pixel_x, const int32_t base_pixel_y,
                                                                 const int32_t image_width, const int32_t image_height,
                                                                 const float* ch_R, const float* ch_G, const float* ch_B) {
                    // Write only the samples that fall inside the image.
#pragma unroll
                    for (int row = 0; row < cell_height; ++row) {
                        const int32_t pixelY = base_pixel_y + row;
                        if (pixelY < 0 || pixelY >= image_height)
                            continue;

                        T* r_row = reinterpret_cast<T*>(r_plane + static_cast<size_t>(pixelY) * r_stride);
                        T* g_row = reinterpret_cast<T*>(g_plane + static_cast<size_t>(pixelY) * g_stride);
                        T* b_row = reinterpret_cast<T*>(b_plane + static_cast<size_t>(pixelY) * b_stride);
                        const int off = row * cell_width;
#pragma unroll
                        for (int col = 0; col < cell_width; ++col) {
                            const int32_t imgX = base_pixel_x + col;
                            if (imgX < 0 || imgX >= image_width)
                                continue;
                            r_row[imgX] = to_sample(ch_R[off + col], clip);
                            g_row[imgX] = to_sample(ch_G[off + col], clip);
                            b_row[imgX] = to_sample(ch_B[off + col], clip);
                        }
                    }
                };
            };

            // planar formats

            template<>
            struct format_backend<packing_format::bgrp> {
                static constexpr bool is_implemented = true;

                // Planar BGR, 8-bit: plane order in memory is B, G, R. The internal cell follows the RGB
                // convention (channel 0 = R, 1 = G, 2 = B), so we reorder the planes into {R, G, B}.
                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                 int32_t imageWidth, int32_t imageHeight) {
                    const uint8_t* planes[3]  = {src[2], src[1], src[0]};
                    const size_t   strides[3] = {stride[2], stride[1], stride[0]};
                    PlanarLoader<uint8_t, CellDataType>::load(planes, strides, cell, nCellX, nCellY, imageWidth,
                                                              imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell, int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    uint8_t*     planes[3]  = {dst[2], dst[1], dst[0]};
                    const size_t strides[3] = {stride[2], stride[1], stride[0]};
                    PlanarSaver<uint8_t, bit_depth::bpp_8u, CellDataType>::save(planes, strides, clip, cell, nCellX,
                                                                               nCellY, imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::rgbp> {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                 int32_t imageWidth, int32_t imageHeight) {
                    PlanarLoader<uint8_t, CellDataType>::load(src, stride, cell, nCellX, nCellY, imageWidth,
                                                              imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell, int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    PlanarSaver<uint8_t, bit_depth::bpp_8u, CellDataType>::save(dst, stride, clip, cell, nCellX,
                                                                               nCellY, imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::yuv444p> {
                static constexpr bool is_implemented = true;

                // Fully planar YUV 4:4:4, 8-bit: planes Y, U, V map directly to cell channels 0, 1, 2.
                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                 int32_t imageWidth, int32_t imageHeight) {
                    PlanarLoader<uint8_t, CellDataType>::load(src, stride, cell, nCellX, nCellY, imageWidth,
                                                              imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell, int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    PlanarSaver<uint8_t, bit_depth::bpp_8u, CellDataType>::save(dst, stride, clip, cell, nCellX, nCellY,
                                                                               imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::yuv444p10> {
                static constexpr bool is_implemented = true;

                // Fully planar YUV 4:4:4, 10-bit data stored in the LSBs of uint16_t (bits [9:0]),
                // matching FFmpeg's yuv444p10le. Planes Y, U, V map directly to cell channels;
                // InputMask / OutputMask = 0x3FF keep the cell in canonical 10-bit values [0, 1023].
                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                 int32_t imageWidth, int32_t imageHeight) {
                    PlanarLoader<uint16_t, CellDataType, /*InputMask=*/0x3FF>::load(src, stride, cell, nCellX, nCellY,
                                                                                   imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell, int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    PlanarSaver<uint16_t, bit_depth::bpp_10u, CellDataType, /*OutputMask=*/0x3FF>::save(
                        dst, stride, clip, cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_INGEST_EXGEST_PLANAR_FAMILY_HPP