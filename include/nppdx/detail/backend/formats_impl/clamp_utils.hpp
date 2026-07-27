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

#ifndef NPPDX_DETAIL_BACKEND_INGEST_CLAMP_UTILS_HPP
#define NPPDX_DETAIL_BACKEND_INGEST_CLAMP_UTILS_HPP

#include "nppdx/operators/function.hpp"
#include "nppdx/detail/backend/constants.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {

            // Template function for clamping with different bit depths
            // Primary template - should not be instantiated
            template<bit_depth Depth>
            __forceinline__ __device__ float fclampf(const float value) {
                static_assert(Depth == bit_depth::bpp_8u || Depth == bit_depth::bpp_10u || Depth == bit_depth::bpp_16u,
                              "Unsupported bit depth");
                return value; // This should never be reached due to static_assert
            }

            // For 8-bit formats: just clamp to 8-bit range, no scaling (work in native 8-bit)
            template<>
            __forceinline__ __device__ float fclampf<bit_depth::bpp_8u>(const float value) {
                // Input and output are both 8-bit (0-255), no scaling needed
                return fminf(fmaxf(value + 0.5f, 0.0f), 255.0f);
            }

            // For 10-bit formats: just clamp, no scaling
            template<>
            __forceinline__ __device__ float fclampf<bit_depth::bpp_10u>(const float value) {
                return fminf(fmaxf(value + 0.5f, 0.0f), 1023.0f);
            }

            // For 16-bit formats: just clamp to 16-bit range, no scaling (work in native 16-bit)
            template<>
            __forceinline__ __device__ float fclampf<bit_depth::bpp_16u>(const float value) {
                // Input and output are both 16-bit (0-65535), no scaling needed
                return fminf(fmaxf(value + 0.5f, 0.0f), 65535.0f);
            }

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_INGEST_CLAMP_UTILS_HPP
