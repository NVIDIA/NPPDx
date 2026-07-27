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

#ifndef NPPDX_DETAIL_BACKEND_INGEST_EXGEST_NV12_FAMILY_HPP
#define NPPDX_DETAIL_BACKEND_INGEST_EXGEST_NV12_FAMILY_HPP

#include <cstdint>
#include "nppdx/detail/utils/force_align.hpp"
#include "nppdx/detail/backend/constants.hpp"
#include "nppdx/detail/backend/formats_impl/format_backend.hpp"
#include "nppdx/detail/backend/formats_impl/clamp_utils.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {

            // NV12 backend 

            template<typename T, bool VDecimate, typename CellDataType>
            struct NV12Base {

                static constexpr unsigned int cell_width  = CellDataType::config_type::CellSize::X;
                static constexpr unsigned int cell_height = CellDataType::config_type::CellSize::Y;

                static_assert(cell_width > 0 && cell_height > 0, "Cell dimensions must be positive");
                static_assert((cell_width % 2) == 0 && (VDecimate ? (cell_height % 2) == 0 : true), "NV12 assumes even cell dimensions");

                struct BaseBlock {
                    union {
                        T                                  value[cell_width];
                        ForceAlign<cell_width * sizeof(T)> _align;
                    };
                };

                __forceinline__ __device__ static bool use_fast_path(
                    const uint8_t* y_plane, const size_t y_stride,
                    const uint8_t* uv_plane, const size_t uv_stride,
                    const int32_t base_pixel_x, const int32_t base_pixel_y,
                    const int32_t image_width, const int32_t image_height
                ) {
                    
                    if (// use safe path if cell's start pixel (upper-left corner) is out-of-bound
                        base_pixel_x < 0 || base_pixel_y < 0 ||
                        // use safe path if cell's end pixel (bottom-right corner) is out-of-bound
                        base_pixel_x + static_cast<int32_t>(cell_width) > image_width ||
                        base_pixel_y + static_cast<int32_t>(cell_height) > image_height) {
                        return false;
                    }

                    constexpr size_t block_align = alignof(BaseBlock);
                    if (y_stride % block_align != 0 || 
                        uv_stride % block_align != 0 || 
                        reinterpret_cast<uintptr_t>(y_plane) % block_align != 0 ||
                        reinterpret_cast<uintptr_t>(uv_plane) % block_align != 0) {
                        return false;
                    }

                    return true; 
                }
            };

            template<typename T, bool VDecimate, typename CellDataType, int InputShift = 0>
            struct NV12Loader {

                using Base                                = NV12Base<T, VDecimate, CellDataType>;
                using LoadBlock                           = typename Base::BaseBlock;
                static constexpr unsigned int cell_width  = CellDataType::config_type::CellSize::X;
                static constexpr unsigned int cell_height = CellDataType::config_type::CellSize::Y;

                __forceinline__ __device__ static float to_float(T value) {
                    return static_cast<float>(value >> InputShift);
                }

                __forceinline__ __device__ static void load(const uint8_t** src, const size_t* stride,
                                                            CellDataType& cell, const int32_t cell_x,
                                                            const int32_t cell_y, const int32_t image_width,
                                                            const int32_t image_height) {
                    load(src[0], src[1], stride[0], stride[1], cell, cell_x, cell_y, image_width, image_height);
                }

            private:
                __forceinline__ __device__ static void load(const uint8_t* y_plane, const uint8_t* uv_plane,
                                                            const size_t y_stride, const size_t uv_stride,
                                                            CellDataType& cell, const int32_t cell_x,
                                                            const int32_t cell_y, const int32_t image_width,
                                                            const int32_t image_height) {
                    int32_t base_pixel_x = cell_x * cell_width;
                    int32_t base_pixel_y = cell_y * cell_height;

                    float* ch_Y = cell.channel_ptr(0);
                    float* ch_U = cell.channel_ptr(1);
                    float* ch_V = cell.channel_ptr(2);

                    if (Base::use_fast_path(y_plane, y_stride, uv_plane, uv_stride, base_pixel_x, base_pixel_y, image_width, image_height)) {
                        fast_load(y_plane, uv_plane, y_stride, uv_stride, base_pixel_x, base_pixel_y, ch_Y, ch_U, ch_V);
                    } else {
                        safe_load(y_plane, uv_plane, y_stride, uv_stride, base_pixel_x, base_pixel_y, image_width,
                                  image_height, ch_Y, ch_U, ch_V);
                    }
                }

                __forceinline__ __device__ static void fast_load(const uint8_t* y_plane, const uint8_t* uv_plane,
                                                                 const size_t y_stride, const size_t uv_stride,
                                                                 const int32_t base_pixel_x, const int32_t base_pixel_y,
                                                                 float* ch_Y, float* ch_U, float* ch_V) {
                    // Pointers where the cell starts (upper-left corner)
                    const uint8_t* y_base_ptr = y_plane + static_cast<size_t>(base_pixel_y) * y_stride + base_pixel_x * sizeof(T);
                    const uint8_t* uv_base_ptr =
                        uv_plane + static_cast<size_t>(VDecimate ? base_pixel_y / 2 : base_pixel_y) * uv_stride + base_pixel_x * sizeof(T);

#pragma unroll
                    for (int row = 0; row < cell_height; ++row) {
                        float* row_Y = &ch_Y[row * cell_width];
                        float* row_U = &ch_U[row * cell_width];
                        float* row_V = &ch_V[row * cell_width];

                        const uint8_t* y_row  = y_base_ptr + static_cast<size_t>(row) * y_stride;
                        const uint8_t* uv_row = uv_base_ptr + static_cast<size_t>(VDecimate ? row / 2 : row) * uv_stride;

                        // Load Y component
                        LoadBlock loaded_Y = *reinterpret_cast<const LoadBlock*>(y_row);
#pragma unroll
                        for (int i = 0; i < cell_width; ++i) {
                            row_Y[i] = to_float(loaded_Y.value[i]);
                        }

                        if constexpr (VDecimate) {
                            // Load UV component if row is even
                            // Copy UV component from the previous row if row is odd
                            if ((row & 1) == 0) {
                                LoadBlock loaded_UV = *reinterpret_cast<const LoadBlock*>(uv_row);
    #pragma unroll
                                for (unsigned int i = 0; i < cell_width; i += 2) {
                                    float u      = to_float(loaded_UV.value[i]);
                                    float v      = to_float(loaded_UV.value[i + 1]);
                                    row_U[i]     = u;
                                    row_U[i + 1] = u;
                                    row_V[i]     = v;
                                    row_V[i + 1] = v;
                                }
                            } else {
    #pragma unroll
                                for (unsigned int i = 0; i < cell_width; ++i) {
                                    row_U[i] = ch_U[(row - 1) * cell_width + i];
                                    row_V[i] = ch_V[(row - 1) * cell_width + i];
                                }
                            }
                        }
                        else { // VDecimate == false
                            // Just read a row from memory
                            LoadBlock loaded_UV = *reinterpret_cast<const LoadBlock*>(uv_row);
#pragma unroll
                            for (unsigned int i = 0; i < cell_width; i += 2) {
                                float u      = to_float(loaded_UV.value[i]);
                                float v      = to_float(loaded_UV.value[i + 1]);
                                row_U[i]     = u;
                                row_U[i + 1] = u;
                                row_V[i]     = v;
                                row_V[i + 1] = v;
                            }
                        }
                    }
                }

                __forceinline__ __device__ static void safe_load(const uint8_t* y_plane, const uint8_t* uv_plane,
                                                                 const size_t y_stride, const size_t uv_stride,
                                                                 const int32_t base_pixel_x, const int32_t base_pixel_y,
                                                                 const int32_t image_width, const int32_t image_height,
                                                                 float* ch_Y, float* ch_U, float* ch_V) {
                    // If an image doesn't have any size, the cell should still be in proper state
                    // (or in other words, should contain zeros)
                    if (image_width <= 0 || image_height <= 0) {
#pragma unroll
                        for (int i = 0; i < cell_width * cell_height; ++i) {
                            ch_Y[i] = 0.0f;
                            ch_U[i] = 0.0f;
                            ch_V[i] = 0.0f;
                        }
                        return;
                    }

                    // Handle case when there are no UV pairs in the image
                    const int uv_width  = image_width / 2;
                    const int uv_height = VDecimate ? image_height / 2 : image_height;

                    if (uv_width <= 0 || uv_height <= 0) {
#pragma unroll
                        for (int row = 0; row < cell_height; ++row) {
                            float* row_Y = &ch_Y[row * cell_width];
                            float* row_U = &ch_U[row * cell_width];
                            float* row_V = &ch_V[row * cell_width];

                            const int32_t  luma_y    = nppdx::clamp_value(base_pixel_y + row, 0, image_height - 1);
                            const T* y_row_ptr = reinterpret_cast<const T*>(y_plane + static_cast<size_t>(luma_y) * y_stride);

#pragma unroll
                            for (int i = 0; i < cell_width; ++i) {
                                const int32_t luma_x = nppdx::clamp_value(base_pixel_x + i, 0, image_width - 1);

                                row_Y[i] = to_float(y_row_ptr[luma_x]);
                                row_U[i] = 0.0f;
                                row_V[i] = 0.0f;
                            }
                        }

                        return;
                    }

                    // Handle normal case, when at least on UV pair is present
#pragma unroll
                    for (int row = 0; row < cell_height; ++row) {
                        float* row_Y = &ch_Y[row * cell_width];
                        float* row_U = &ch_U[row * cell_width];
                        float* row_V = &ch_V[row * cell_width];

                        const int32_t luma_y   = nppdx::clamp_value(base_pixel_y + row, 0, image_height - 1);
                        const int32_t chroma_y = nppdx::clamp_value(VDecimate ? luma_y / 2 : luma_y, 0, uv_height - 1);

                        const T* y_row_ptr  = reinterpret_cast<const T*>(y_plane + static_cast<size_t>(luma_y) * y_stride);
                        const T* uv_row_ptr = reinterpret_cast<const T*>(uv_plane + static_cast<size_t>(chroma_y) * uv_stride);

#pragma unroll
                        for (int i = 0; i < cell_width; ++i) {
                            const int32_t luma_x   = nppdx::clamp_value(base_pixel_x + i, 0, image_width - 1);
                            const int32_t chroma_x = nppdx::clamp_value(luma_x / 2, 0, uv_width - 1);

                            row_Y[i] = to_float(y_row_ptr[luma_x]);
                            row_U[i] = to_float(uv_row_ptr[2 * chroma_x + 0]);
                            row_V[i] = to_float(uv_row_ptr[2 * chroma_x + 1]);
                        }
                    }
                }
            }; // NV12Loader

            template<typename T, bit_depth R, bool VDecimate, typename CellDataType, int OutputShift = 0>
            struct NV12Saver {

                using Base                                = NV12Base<T, VDecimate, CellDataType>;
                using SaveBlock                           = typename Base::BaseBlock;
                static constexpr unsigned int cell_width  = CellDataType::config_type::CellSize::X;
                static constexpr unsigned int cell_height = CellDataType::config_type::CellSize::Y;

                __forceinline__ __device__ static T to_sample(float value, const bool clip) {
                    return static_cast<T>(clip ? fclampf<R>(value) : value) << OutputShift;
                }

                __forceinline__ __device__ static void save(uint8_t** dst, const size_t* stride, const bool clip,
                                                            const CellDataType& cell, const int32_t cell_x,
                                                            const int32_t cell_y, const int32_t image_width,
                                                            const int32_t image_height) {
                    save(dst[0], dst[1], stride[0], stride[1], clip, cell, cell_x, cell_y, image_width, image_height);
                }

            private:
                __forceinline__ __device__ static void save(uint8_t* y_plane, uint8_t* uv_plane, const size_t y_stride,
                                                            const size_t uv_stride, const bool clip,
                                                            const CellDataType& cell, const int32_t cell_x,
                                                            const int32_t cell_y, const int32_t image_width,
                                                            const int32_t image_height) {
                    int32_t base_pixel_x = cell_x * cell_width;
                    int32_t base_pixel_y = cell_y * cell_height;

                    const float* ch_Y = cell.channel_ptr(0);
                    const float* ch_U = cell.channel_ptr(1);
                    const float* ch_V = cell.channel_ptr(2);

                    if (Base::use_fast_path(y_plane, y_stride, uv_plane, uv_stride, base_pixel_x, base_pixel_y, image_width, image_height)) {
                        fast_save(y_plane, uv_plane, y_stride, uv_stride, clip, base_pixel_x, base_pixel_y, ch_Y, ch_U,
                                  ch_V);
                    } else {
                        safe_save(y_plane, uv_plane, y_stride, uv_stride, clip, base_pixel_x, base_pixel_y, image_width,
                                  image_height, ch_Y, ch_U, ch_V);
                    }
                }

                __forceinline__ __device__ static void fast_save(uint8_t* y_plane, uint8_t* uv_plane,
                                                                 const size_t y_stride, const size_t uv_stride,
                                                                 const bool clip, const int32_t base_pixel_x,
                                                                 const int32_t base_pixel_y, const float* ch_Y,
                                                                 const float* ch_U, const float* ch_V) {
                    // Pointers where the cell starts (upper-left corner)
                    uint8_t* y_base_ptr  = y_plane + static_cast<size_t>(base_pixel_y) * y_stride + base_pixel_x * sizeof(T);
                    uint8_t* uv_base_ptr = uv_plane +
                                           static_cast<size_t>(VDecimate ? base_pixel_y / 2 : base_pixel_y) * uv_stride +
                                           base_pixel_x * sizeof(T);

#pragma unroll
                    for (int row = 0; row < cell_height; ++row) {
                        const float* row_Y = &ch_Y[row * cell_width];
                        const float* row_U = &ch_U[row * cell_width];
                        const float* row_V = &ch_V[row * cell_width];

                        uint8_t* y_row  = y_base_ptr + static_cast<size_t>(row) * y_stride;
                        uint8_t* uv_row = uv_base_ptr + static_cast<size_t>(VDecimate ? row / 2 : row) * uv_stride;

                        // Fill Y Block and save
                        SaveBlock tosave_Y;
#pragma unroll
                        for (int i = 0; i < cell_width; ++i) {
                            tosave_Y.value[i] = to_sample(row_Y[i], clip);
                        }
                        *reinterpret_cast<SaveBlock*>(y_row) = tosave_Y;

                        if constexpr (VDecimate) {
                            if ((row & 1) == 0) {
                                SaveBlock tosave_UV;
#pragma unroll
                                for (int i = 0; i < cell_width / 2; ++i) {
                                    // Just an average values from current and the next row
                                    float U = (row_U[2 * i] + row_U[2 * i + 1] + row_U[cell_width + 2 * i] +
                                               row_U[cell_width + 2 * i + 1]) /
                                              4;
                                    float V = (row_V[2 * i] + row_V[2 * i + 1] + row_V[cell_width + 2 * i] +
                                               row_V[cell_width + 2 * i + 1]) /
                                              4;

                                    tosave_UV.value[2 * i]     = to_sample(U, clip);
                                    tosave_UV.value[2 * i + 1] = to_sample(V, clip);
                                }
                                *reinterpret_cast<SaveBlock*>(uv_row) = tosave_UV;
                            }
                        } else {
                            SaveBlock tosave_UV;
#pragma unroll
                            for (int i = 0; i < cell_width / 2; ++i) {
                                float U = (row_U[2 * i] + row_U[2 * i + 1]) / 2;
                                float V = (row_V[2 * i] + row_V[2 * i + 1]) / 2;

                                tosave_UV.value[2 * i]     = to_sample(U, clip);
                                tosave_UV.value[2 * i + 1] = to_sample(V, clip);
                            }
                            *reinterpret_cast<SaveBlock*>(uv_row) = tosave_UV;
                        }
                    }
                }

                __forceinline__ __device__ static void safe_save(uint8_t* y_plane, uint8_t* uv_plane,
                                                                 const size_t y_stride, const size_t uv_stride,
                                                                 const bool clip, const int32_t base_pixel_x,
                                                                 const int32_t base_pixel_y, const int32_t image_width,
                                                                 const int32_t image_height, const float* ch_Y,
                                                                 const float* ch_U, const float* ch_V) {

                    const int uv_width  = image_width / 2;
                    const int uv_height = VDecimate ? image_height / 2 : image_height;

                    // Pointers where the cell starts (upper-left corner)
                    uint8_t* y_base_ptr  = y_plane + static_cast<size_t>(base_pixel_y) * y_stride + base_pixel_x * sizeof(T);
                    uint8_t* uv_base_ptr = uv_plane +
                                           static_cast<size_t>(VDecimate ? base_pixel_y / 2 : base_pixel_y) * uv_stride +
                                           base_pixel_x * sizeof(T);
#pragma unroll
                    for (int row = 0; row < cell_height; ++row) {
                        const float* row_Y = &ch_Y[row * cell_width];
                        const float* row_U = &ch_U[row * cell_width];
                        const float* row_V = &ch_V[row * cell_width];

                        T* y_row  = reinterpret_cast<T*>(y_base_ptr + static_cast<size_t>(row) * y_stride);
                        T* uv_row = reinterpret_cast<T*>(uv_base_ptr +
                                                         static_cast<size_t>(VDecimate ? row / 2 : row) * uv_stride);

                        const int32_t luma_y   = base_pixel_y + row;
                        const int32_t chroma_y = VDecimate ? luma_y / 2 : luma_y;

                        // Try to write something only when current row is in the image size range
                        if (0 <= luma_y && luma_y < image_height) {
                            // Save Luma pixels
                            for (int i = 0; i < cell_width; ++i) {
                                if (0 <= base_pixel_x + i && base_pixel_x + i < image_width) {
                                    y_row[i] = to_sample(row_Y[i], clip);
                                }
                            }

                            // Save Chroma pixels. For 4:2:0 (VDecimate) only on even rows (a 2x2 block
                            // shares one chroma sample); for 4:2:2 on every row (vertical full res).
                            const bool write_chroma = VDecimate ? ((row & 1) == 0) : true;
                            if (write_chroma && (0 <= chroma_y && chroma_y < uv_height)) {
                                for (int i = 0; i < cell_width / 2; ++i) {
                                    if (0 <= base_pixel_x / 2 + i && base_pixel_x / 2 + i < uv_width) {
                                        // If we know that we should save UV, we just average values from the cell
                                        float U, V;
                                        if constexpr (VDecimate) {
                                            U = (row_U[2 * i] + row_U[2 * i + 1] + row_U[cell_width + 2 * i] +
                                                 row_U[cell_width + 2 * i + 1]) /
                                                4;
                                            V = (row_V[2 * i] + row_V[2 * i + 1] + row_V[cell_width + 2 * i] +
                                                 row_V[cell_width + 2 * i + 1]) /
                                                4;
                                        } else {
                                            U = (row_U[2 * i] + row_U[2 * i + 1]) / 2;
                                            V = (row_V[2 * i] + row_V[2 * i + 1]) / 2;
                                        }
                                        uv_row[2 * i]     = to_sample(U, clip);
                                        uv_row[2 * i + 1] = to_sample(V, clip);
                                    }
                                }
                            }
                        }
                    }
                }

            }; // NV12Saver
            
            // Formats adapters that use NV12 backend

            template<>
            struct format_backend<packing_format::nv12> {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, const int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    NV12Loader<uint8_t, /*VDecimate=*/true, CellDataType>::load(src, stride, cell, nCellX, nCellY, imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    NV12Saver<uint8_t, bit_depth::bpp_8u, /*VDecimate=*/true, CellDataType>::save(dst, stride, clip, cell, nCellX, nCellY,
                                                                              imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::p010> {
                static constexpr bool is_implemented = true;

                // The same format as NV12, but 10-bit data stored in the upper bits of uint16_t (bits
                // [15:6]). InputShift / OutputShift = 6 push the int<->float conversion down into the
                // generic NV12 helpers, so the cell always holds canonical 10-bit values [0, 1023] — no
                // wrapper pass, no temp_cell scratch.
                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, const int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    NV12Loader<uint16_t, /*VDecimate=*/true, CellDataType, /*InputShift=*/6>::load(src, stride, cell, nCellX, nCellY,
                                                                              imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    NV12Saver<uint16_t, bit_depth::bpp_10u, /*VDecimate=*/true, CellDataType, /*OutputShift=*/6>::save(
                        dst, stride, clip, cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::nv16> {
                static constexpr bool is_implemented = true;

                // Same semi-planar layout as NV12, but 4:2:2: chroma keeps full vertical resolution
                // (VDecimate = false), so the UV plane has the same height as the Y plane.
                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, const int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    NV12Loader<uint8_t, /*VDecimate=*/false, CellDataType>::load(src, stride, cell, nCellX, nCellY,
                                                                                imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    NV12Saver<uint8_t, bit_depth::bpp_8u, /*VDecimate=*/false, CellDataType>::save(
                        dst, stride, clip, cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::p216> {
                static constexpr bool is_implemented = true;

                // 16-bit counterpart of nv16: semi-planar YUV 4:2:2 with full 16-bit samples (the data
                // fills the whole word, so no InputShift/OutputShift unlike p010). VDecimate = false.
                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, const int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    NV12Loader<uint16_t, /*VDecimate=*/false, CellDataType>::load(src, stride, cell, nCellX, nCellY,
                                                                                 imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    NV12Saver<uint16_t, bit_depth::bpp_16u, /*VDecimate=*/false, CellDataType>::save(
                        dst, stride, clip, cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_INGEST_EXGEST_NV12_FAMILY_HPP
