/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NPPDX_DETAIL_BACKEND_CONSTANTS_HPP
#define NPPDX_DETAIL_BACKEND_CONSTANTS_HPP

namespace nppdx {
    namespace detail {
        namespace backend {

            // Scaling constants for bit depth conversions
            inline constexpr float scale_8bit_to_16bit = 257.0f;          // Maps 255 to 65535 exactly
            inline constexpr float scale_16bit_to_8bit = 1.0f / 257.0f;   // Inverse for round trip
            inline constexpr float scale_10bit_to_16bit = 64.06158358f;   // Maps 1023 to 65535 exactly
            inline constexpr float scale_16bit_to_10bit = 1.0f / 64.06158358f; // Inverse for round trip
            inline constexpr float scale_8bit_to_10bit = 4.0f;            // Keep exact for YUV conversions
            inline constexpr float scale_10bit_to_8bit = 0.25f;           // Keep exact for YUV conversions

            // Scaling constants for value normalization to the range [0, 1] 
            inline constexpr float scale_8bit_to_zerone = 1.0f / 255.0f;
            inline constexpr float scale_10bit_to_zerone = 1.0f / 1023.0f;
            inline constexpr float scale_16bit_to_zerone = 1.0f / 65535.0f;
            inline constexpr float scale_zerone_to_8bit = 255.0f;
            inline constexpr float scale_zerone_to_10bit = 1023.0f;
            inline constexpr float scale_zerone_to_16bit = 65535.0f;

            // Color conversion constants for 10-bit range
            inline constexpr float y_offset_10bit  = 0.0f;   // No offset for Y (luma) in 10-bit range
            inline constexpr float uv_offset_10bit = 512.0f; // Mid-point for UV in 10-bit range


        } // namespace backend
    }     // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_CONSTANTS_HPP
