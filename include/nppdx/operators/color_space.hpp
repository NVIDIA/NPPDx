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

#ifndef NPPDX_OPERATORS_COLOR_SPACE_HPP
#define NPPDX_OPERATORS_COLOR_SPACE_HPP

#include <nppdx/utils.hpp>
#include "nppdx/detail/config.hpp"

#include NPPDX_STD_INCLUDE_UTILITY

namespace nppdx {
    enum class color_space
    {
        rgb,
        yuv_bt601, // BT.601 - Standard Definition TV (SDTV)
        yuv_bt709, // BT.709 - High Definition TV (HDTV)
        yuv_bt2020 // BT.2020 - Ultra High Definition TV (UHDTV)
    };

    // Note for developers: Whenever adding or removing color_space enumerators,
    // update NPPDX_FOR_COLOR_SPACES at the bottom of this header accordingly,
    // as well as color_conversion_supported() below.

    constexpr __forceinline__ __host__ __device__ bool color_conversion_supported(color_space input,
                                                                                  color_space output) {
        return (input == output || input == color_space::rgb || output == color_space::rgb);
    }

    // Apply a custom macro to all color spaces.
    // See description of NPPDX_FOR_PACKING_FORMATS in formats.hpp for more
    // details and a list of caveats.
#define NPPDX_FOR_COLOR_SPACES(MACRO, ...) \
    MACRO(rgb, __VA_ARGS__)                \
    MACRO(yuv_bt601, __VA_ARGS__)          \
    MACRO(yuv_bt709, __VA_ARGS__)          \
    MACRO(yuv_bt2020, __VA_ARGS__)

    template<typename ColorSpaceCarrierT, typename FunctorT, typename... ArgsT>
    constexpr auto dispatch_color_space(const ColorSpaceCarrierT& colorSpaceCarrier, FunctorT&& functor,
                                        ArgsT&&... args) {
        static_assert(has_carried_type<color_space, ColorSpaceCarrierT>);

#define NPPDX_PRIVATE_CASE_BODY_CS(colorSpace) \
    NPPDX_STD::forward<FunctorT>(functor)(colorSpace, NPPDX_STD::forward<ArgsT>(args)...)

        if constexpr (is_cval<ColorSpaceCarrierT>) {
            return NPPDX_PRIVATE_CASE_BODY_CS(colorSpaceCarrier);
        } else {
#define NPPDX_PRIVATE_CASE_CS(colorSpace, ...) \
    case color_space::colorSpace: return NPPDX_PRIVATE_CASE_BODY_CS(C<color_space::colorSpace>);
            switch (colorSpaceCarrier) {
                NPPDX_FOR_COLOR_SPACES(NPPDX_PRIVATE_CASE_CS)
                default: fatal_error();
            }

#undef NPPDX_PRIVATE_CASE_CS
        }

        // Even though the code below is unreachable, it appears to be a
        // necessary workaround for GCC 7 to correctly infer the return type.
        return NPPDX_PRIVATE_CASE_BODY_CS(C<color_space::rgb>);

#undef NPPDX_PRIVATE_CASE_BODY_CS
    }

} // namespace nppdx

#endif // NPPDX_OPERATORS_COLOR_SPACE_HPP
