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

#ifndef NPPDX_DETAIL_BACKEND_AFFINE_CHANNEL_MAP_OPERATIONS_HPP
#define NPPDX_DETAIL_BACKEND_AFFINE_CHANNEL_MAP_OPERATIONS_HPP

#include "nppdx/detail/config.hpp"
#include "nppdx/detail/decl.hpp"

#include NPPDX_STD_INCLUDE_CMATH
#include NPPDX_STD_INCLUDE_CSTDINT

// Sentinel for AffineChannelMap clip_min / clip_max template parameters: bound disabled.
#define NPPDX_AFFINE_NO_CLIP static_cast<int32_t>(0xDEADBEEF)

namespace nppdx {
    namespace detail {
        namespace backend {
            namespace affine_channel_map_operations {

                template<int32_t ClipBound>
                NPPDX_DECL_NCFHD bool clip_bound_active() {
                    return ClipBound != NPPDX_AFFINE_NO_CLIP;
                }

                template<int32_t ClipMin, int32_t ClipMax>
                NPPDX_DECL_NCFHD bool both_clip_bounds_active() {
                    return clip_bound_active<ClipMin>() && clip_bound_active<ClipMax>();
                }

                template<int32_t PreOffset, int32_t Numerator, int32_t Denominator, int32_t PostOffset, int32_t ClipMin,
                         int32_t ClipMax>
                NPPDX_DECL_NCFHD float apply(float value) {
                    if constexpr (PreOffset != 0) {
                        value += static_cast<float>(PreOffset);
                    }
                    if constexpr (Numerator != Denominator) {
                        value *= static_cast<float>(Numerator) / static_cast<float>(Denominator);
                    }
                    if constexpr (PostOffset != 0) {
                        value += static_cast<float>(PostOffset);
                    }
                    if constexpr (clip_bound_active<ClipMin>()) {
                        value = fmaxf(value, static_cast<float>(ClipMin));
                    }
                    if constexpr (clip_bound_active<ClipMax>()) {
                        value = fminf(value, static_cast<float>(ClipMax));
                    }
                    return value;
                }

            } // namespace affine_channel_map_operations
        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_AFFINE_CHANNEL_MAP_OPERATIONS_HPP
