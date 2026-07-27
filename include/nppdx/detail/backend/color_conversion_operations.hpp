/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NPPDX_DETAIL_BACKEND_COLOR_CONVERSION_OPERATIONS_HPP
#define NPPDX_DETAIL_BACKEND_COLOR_CONVERSION_OPERATIONS_HPP

#include "nppdx/operators/function.hpp"
#include "nppdx/detail/backend/constants.hpp"
#include "nppdx/detail/database/cell_data.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {

            // "Reference" = the fixed numeric convention encoded by scale_* symbols in constants.hpp
            // Both overloads are constexpr: no-arg returns the factor only;
            // float overload multiplies a sample (identical math to factor * x for 8/10/16).
            template<bit_depth InputDepth>
            __forceinline__ __host__ __device__ constexpr float scale_to_reference(float value) {
                if constexpr (InputDepth == bit_depth::bpp_8u) {
                    return value * scale_8bit_to_10bit;
                } else if constexpr (InputDepth == bit_depth::bpp_10u) {
                    return value;
                } else if constexpr (InputDepth == bit_depth::bpp_16u) {
                    return value * scale_16bit_to_10bit;
                } else {
                    return value;
                }
            }

            template<bit_depth Depth>
            __forceinline__ __host__ __device__ constexpr float scale_to_reference() {
                if constexpr (Depth == bit_depth::bpp_8u) {
                    return scale_8bit_to_10bit;
                } else if constexpr (Depth == bit_depth::bpp_10u) {
                    return 1.0f;
                } else if constexpr (Depth == bit_depth::bpp_16u) {
                    return scale_16bit_to_10bit;
                } else {
                    return 1.0f;
                }
            }

            template<bit_depth OutputDepth>
            __forceinline__ __host__ __device__ constexpr float scale_from_reference(float value) {
                if constexpr (OutputDepth == bit_depth::bpp_8u) {
                    return value * scale_10bit_to_8bit;
                } else if constexpr (OutputDepth == bit_depth::bpp_10u) {
                    return value;
                } else if constexpr (OutputDepth == bit_depth::bpp_16u) {
                    return value * scale_10bit_to_16bit;
                } else {
                    return value;
                }
            }

            template<bit_depth Depth>
            __forceinline__ __host__ __device__ constexpr float scale_from_reference() {
                if constexpr (Depth == bit_depth::bpp_8u) {
                    return scale_10bit_to_8bit;
                } else if constexpr (Depth == bit_depth::bpp_10u) {
                    return 1.0f;
                } else if constexpr (Depth == bit_depth::bpp_16u) {
                    return scale_10bit_to_16bit;
                } else {
                    return 1.0f;
                }
            }

            // Gray-axis midpoint in coded float space for this bit depth (U/V bias for YUV paths).
            template<bit_depth Depth>
            struct midpoint_gray;

            template<>
            struct midpoint_gray<bit_depth::bpp_8u> {
                static constexpr float value = uv_offset_10bit * scale_10bit_to_8bit; // 128.0f
            };
            template<>
            struct midpoint_gray<bit_depth::bpp_10u> {
                static constexpr float value = uv_offset_10bit; // 512.0f
            };
            template<>
            struct midpoint_gray<bit_depth::bpp_16u> {
                static constexpr float value = 32768.0f;
            };

            // Native depth → native depth linear rescale (one constexpr multiply). Same as
            // scale_from_reference<Out>(scale_to_reference<In>(x)) for 8/10/16.
            template<bit_depth InputDepth, bit_depth OutputDepth>
            __forceinline__ __host__ __device__ constexpr float scale_depth() {
                return scale_to_reference<InputDepth>() * scale_from_reference<OutputDepth>();
            }


            namespace color_conversion {

                template<bit_depth InD, bit_depth OutD, color_space CS>
                __forceinline__ __device__ void yuv_to_rgb_scale_cell(float y, float u, float v, float& r, float& g,
                                                                      float& b) {
                    constexpr float s_in  = scale_to_reference<InD>();
                    constexpr float s_out = scale_from_reference<OutD>();
                    constexpr float S     = s_in * s_out;
                    constexpr float m     = midpoint_gray<InD>::value;
                    const float     du    = u - m;
                    const float     dv    = v - m;

                    if constexpr (CS == color_space::yuv_bt601) {
                        r = S * (y + 1.4020000f * dv);
                        g = S * (y - 0.3441363f * du - 0.7141363f * dv);
                        b = S * (y + 1.7720000f * du);
                    } else if constexpr (CS == color_space::yuv_bt709) {
                        r = S * (y + 1.5748000f * dv);
                        g = S * (y - 0.1873240f * du - 0.4681240f * dv);
                        b = S * (y + 1.8556000f * du);
                    } else if constexpr (CS == color_space::yuv_bt2020) {
                        r = S * (y + 1.4746000f * dv);
                        g = S * (y - 0.1645531f * du - 0.5713531f * dv);
                        b = S * (y + 1.8814000f * du);
                    }
                }

                template<bit_depth InD, bit_depth OutD, color_space CS>
                __forceinline__ __device__ void rgb_to_yuv_scale_cell(float r, float g, float b, float& y, float& u,
                                                                      float& v) {
                    constexpr float s_in   = scale_to_reference<InD>();
                    constexpr float s_out  = scale_from_reference<OutD>();
                    constexpr float S      = s_in * s_out;
                    constexpr float uv_add = midpoint_gray<OutD>::value;

                    if constexpr (CS == color_space::yuv_bt601) {
                        y = S * (0.29900f * r + 0.58700f * g + 0.11400f * b);
                        u = S * (-0.16873600f * r - 0.33126399f * g + 0.50000000f * b) + uv_add;
                        v = S * (0.50000000f * r - 0.41868800f * g - 0.08131200f * b) + uv_add;
                    } else if constexpr (CS == color_space::yuv_bt709) {
                        y = S * (0.212600f * r + 0.715200f * g + 0.072200f * b);
                        u = S * (-0.114572f * r - 0.385428f * g + 0.500000f * b) + uv_add;
                        v = S * (0.500000f * r - 0.454153f * g - 0.045847f * b) + uv_add;
                    } else if constexpr (CS == color_space::yuv_bt2020) {
                        y = S * (0.26270f * r + 0.67800f * g + 0.05930f * b);
                        u = S * (-0.1396301f * r - 0.3603699f * g + 0.5000000f * b) + uv_add;
                        v = S * (0.5000000f * r - 0.4597857f * g - 0.0402143f * b) + uv_add;
                    }
                }

                // Generic conversion function using explicit dispatch with bit depth scaling
                template<bit_depth InputBitDepth, bit_depth OutputBitDepth, color_space InputColorSpace,
                         color_space OutputColorSpace, typename CellCfg, typename T>
                __forceinline__ __device__ void convert(NPPCellData<CellCfg, T> npp_cell_data) {
                    static_assert(CellCfg::value.channel_count == 3, "Color conversion requires 3 channels");

                    // Note: color_conversion_supported in nppdx/operators/color_space.hpp must match the
                    // combinations handled by convert<> (and the scale_cell helpers) in this file.
                    static_assert(color_conversion_supported(InputColorSpace, OutputColorSpace),
                                  "Unsupported color space conversion");

                    [[maybe_unused]] static constexpr unsigned int CellPixels = CellCfg::value.cell_pixels();

                    float* c0 = npp_cell_data.channel_ptr(0);
                    float* c1 = npp_cell_data.channel_ptr(1);
                    float* c2 = npp_cell_data.channel_ptr(2);

#pragma unroll
                    for (unsigned int i = 0; i < CellPixels; ++i) {
                        float a = c0[i];
                        float b = c1[i];
                        float c = c2[i];

                        if constexpr (InputColorSpace == OutputColorSpace) {
                            if constexpr (InputBitDepth == OutputBitDepth) {
                                // no-op
                            } else {
                                constexpr float depth_scale = scale_depth<InputBitDepth, OutputBitDepth>();
                                a *= depth_scale;
                                b *= depth_scale;
                                c *= depth_scale;
                            }
                        } else if constexpr (InputColorSpace == color_space::yuv_bt601 &&
                                             OutputColorSpace == color_space::rgb) {
                            yuv_to_rgb_scale_cell<InputBitDepth, OutputBitDepth, color_space::yuv_bt601>(a, b, c, a, b,
                                                                                                         c);
                        } else if constexpr (InputColorSpace == color_space::yuv_bt709 &&
                                             OutputColorSpace == color_space::rgb) {
                            yuv_to_rgb_scale_cell<InputBitDepth, OutputBitDepth, color_space::yuv_bt709>(a, b, c, a, b,
                                                                                                         c);
                        } else if constexpr (InputColorSpace == color_space::yuv_bt2020 &&
                                             OutputColorSpace == color_space::rgb) {
                            yuv_to_rgb_scale_cell<InputBitDepth, OutputBitDepth, color_space::yuv_bt2020>(a, b, c, a, b,
                                                                                                          c);
                        } else if constexpr (InputColorSpace == color_space::rgb &&
                                             OutputColorSpace == color_space::yuv_bt601) {
                            rgb_to_yuv_scale_cell<InputBitDepth, OutputBitDepth, color_space::yuv_bt601>(a, b, c, a, b,
                                                                                                         c);
                        } else if constexpr (InputColorSpace == color_space::rgb &&
                                             OutputColorSpace == color_space::yuv_bt709) {
                            rgb_to_yuv_scale_cell<InputBitDepth, OutputBitDepth, color_space::yuv_bt709>(a, b, c, a, b,
                                                                                                         c);
                        } else if constexpr (InputColorSpace == color_space::rgb &&
                                             OutputColorSpace == color_space::yuv_bt2020) {
                            rgb_to_yuv_scale_cell<InputBitDepth, OutputBitDepth, color_space::yuv_bt2020>(a, b, c, a, b,
                                                                                                          c);
                        }

                        c0[i] = a;
                        c1[i] = b;
                        c2[i] = c;
                    }

                } // convert

                // RGB to Grayscale conversion (3 channels)
                template<typename CellSize, class T>
                __forceinline__ __device__ void rgb_to_gray(NPPCellData<CellConfig<CellSize, 3>, T>& npp_cell_data) {
                    static constexpr unsigned int CellPixels = CellSize::value.x * CellSize::value.y;
#pragma unroll
                    for (int i = 0; i < CellPixels; ++i) {
                        float gray = 0.299f * npp_cell_data.channel_ptr(0)[i] +
                                     0.587f * npp_cell_data.channel_ptr(1)[i] +
                                     0.114f * npp_cell_data.channel_ptr(2)[i];
                        npp_cell_data.channel_ptr(0)[i] = gray; // R = gray
                        npp_cell_data.channel_ptr(1)[i] = gray; // G = gray
                        npp_cell_data.channel_ptr(2)[i] = gray; // B = gray
                        // ch3 (alpha) is not used in 3-channel processing
                    }
                }

            } // namespace color_conversion

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_COLOR_CONVERSION_OPERATIONS_HPP
