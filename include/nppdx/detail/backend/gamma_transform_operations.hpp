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

#ifndef NPPDX_DETAIL_BACKEND_GAMMA_TRANSFORM_OPERATIONS_HPP
#define NPPDX_DETAIL_BACKEND_GAMMA_TRANSFORM_OPERATIONS_HPP

#include "nppdx/detail/config.hpp"
#include "nppdx/utils.hpp"
#include "nppdx/operators/function.hpp"
#include "constants.hpp"

#include NPPDX_STD_INCLUDE_CSTDINT

namespace nppdx {
    namespace detail {
        namespace backend {

            template<bit_depth depth>
            __forceinline__ __device__ float scale_to_zerone(float value) {
                if constexpr (depth == bit_depth::bpp_8u) {
                    return value * scale_8bit_to_zerone;
                } else if constexpr (depth == bit_depth::bpp_10u) {
                    return value * scale_10bit_to_zerone;
                } else if constexpr (depth == bit_depth::bpp_16u) {
                    return value * scale_16bit_to_zerone;
                } else {
                    static_assert(depth != depth, "Unknown bit depth.");
                    return value;
                }
            }

            template<bit_depth depth>
            __forceinline__ __device__ float scale_from_zerone(float value) {
                if constexpr (depth == bit_depth::bpp_8u) {
                    return value * scale_zerone_to_8bit;
                } else if constexpr (depth == bit_depth::bpp_10u) {
                    return value * scale_zerone_to_10bit;
                } else if constexpr (depth == bit_depth::bpp_16u) {
                    return value * scale_zerone_to_16bit;
                } else {
                    static_assert(depth != depth, "Unknown bit depth.");
                    return value;
                }
            }

            namespace gamma_transform_operations {

                // ======================================================================
                // Base implementation
                // ======================================================================

                // valid function assumes that value is in the rage [0, 1]
                template<gamma_dir TransformDirection, gamma_transfer_function TransferFunction>
                __forceinline__ __device__ float interpolate_valid(float value) {
                    static_assert(TransformDirection != TransformDirection, "Configuration is not supported");
                    return value;
                }

                //
                template<gamma_dir TransformDirection, gamma_transfer_function TransferFunction>
                __forceinline__ __device__ float interpolate(float value) {
                    // return the value itself if it's out of bound
                    if (value <= 0.0f || value >= 1.0f) {
                        return value;
                    }

                    return interpolate_valid<TransformDirection, TransferFunction>(value);
                }

                // ======================================================================
                // SDR - standard dynamic range
                // ======================================================================

                // constants defined in the exact formula
                inline constexpr float SDR_beta  = 0.018053968510807f;
                inline constexpr float SDR_alpha = 1.09929682680944f;

                // helper constants
                inline constexpr float SDR_slope    = 4.5f;
                inline constexpr float SDR_exponent = 0.45f;
                inline constexpr float SDR_Y_beta   = SDR_slope * SDR_beta;

                // SDR forward
                template<>
                __forceinline__ __device__ float interpolate_valid<gamma_dir::forward, gamma_transfer_function::SDR>(
                    float value) {
                    // compute SDR forward gamma transform
                    if (value < SDR_beta) {
                        return SDR_slope * value;
                    } else {
                        return SDR_alpha * __powf(value, SDR_exponent) - (SDR_alpha - 1);
                    }
                }

                // SDR inverse
                template<>
                __forceinline__ __device__ float interpolate_valid<gamma_dir::inverse, gamma_transfer_function::SDR>(
                    float value) {
                    // compute SDR inverse gamma transform
                    if (value < SDR_Y_beta) {
                        return value / SDR_slope;
                    } else {
                        return __powf((value + (SDR_alpha - 1)) / SDR_alpha, 1.0f / SDR_exponent);
                    }
                }

                // ======================================================================
                // HLG - hybrid log-gamma
                // ======================================================================

                // constants defined in the exact formula
                inline constexpr float HLG_a = 0.17883277f;
                inline constexpr float HLG_b = 0.28466892f;    // 1 - 4 * a
                inline constexpr float HLG_c = 0.55991072953f; // 0.5 - a * ln(4a)

                // helper constants
                inline constexpr float HLG_slope = 3.0f;
                inline constexpr float HLG_range = 12.0f;
                inline constexpr float HLG_half  = 0.5f;

                // HLG forward
                template<>
                __forceinline__ __device__ float interpolate_valid<gamma_dir::forward, gamma_transfer_function::HLG>(
                    float value) {
                    // compute HLG forward gamma transform
                    if (value <= 1.0f / HLG_range) {
                        return sqrtf(HLG_slope * value);
                    } else {
                        return HLG_a * __logf(HLG_range * value - HLG_b) + HLG_c;
                    }
                }

                // HLG inverse
                template<>
                __forceinline__ __device__ float interpolate_valid<gamma_dir::inverse, gamma_transfer_function::HLG>(
                    float value) {
                    // compute HLG inverse gamma transform
                    if (value < HLG_half) {
                        return (value * value) / HLG_slope;
                    } else {
                        return (__expf((value - HLG_c) / HLG_a) + HLG_b) / HLG_range;
                    }
                }

                // ======================================================================
                // PQ - perceptual quantizer
                // ======================================================================

                // constants defined in the exact formula
                // _m - represent exponents
                // _c - represent coeficient constants
                inline constexpr float PQ_m1 = 0.1593017578125f;
                inline constexpr float PQ_m2 = 78.84375f;
                inline constexpr float PQ_c3 = 18.6875f;
                inline constexpr float PQ_c2 = 18.8515625f;
                inline constexpr float PQ_c1 = PQ_c3 - PQ_c2 + 1.0f;

                // PQ forward
                template<>
                __forceinline__ __device__ float interpolate_valid<gamma_dir::forward, gamma_transfer_function::PQ>(
                    float value) {
                    // compute PQ forward gamma transform

                    const float val_m1 = __powf(value, PQ_m1);
                    const float base   = (PQ_c1 + PQ_c2 * val_m1) / (1.0f + PQ_c3 * val_m1);
                    return __powf(base, PQ_m2);
                }

                // PQ inverse
                template<>
                __forceinline__ __device__ float interpolate_valid<gamma_dir::inverse, gamma_transfer_function::PQ>(
                    float value) {
                    // compute PQ inverse gamma transform

                    const float val_m2 = __powf(value, 1.0f / PQ_m2);
                    const float base   = nppdx_max(val_m2 - PQ_c1, 0.0f) / (PQ_c2 - PQ_c3 * val_m2);
                    return __powf(base, 1.0f / PQ_m1);
                }

            } // namespace gamma_transform_operations

        } // namespace backend
    } // namespace detail
} // namespace nppdx
#endif // NPPDX_DETAIL_BACKEND_GAMMA_TRANSFORM_OPERATIONS_HPP
