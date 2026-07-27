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

#ifndef NPPDX_DETAIL_NPPDX_IS_SUPPORTED_HPP
#define NPPDX_DETAIL_NPPDX_IS_SUPPORTED_HPP

#include "commondx/detail/stl/type_traits.hpp"

#include "../operators.hpp"
#include "../traits.hpp"
#include "nppdx_checks.hpp"
#include "nppdx/detail/database/cell_config.hpp"
#include "nppdx/detail/decl.hpp"


namespace nppdx {
    namespace detail {
        // Host+device safe: composed entirely of constexpr __host__ __device__ helpers, so
        // `is_supported_v<...>` can also be evaluated from device-side template metaprogramming
        // (e.g. `static_assert` inside a kernel).
        NPPDX_DECL_NCHD bool is_supported_impl(uint2 tile_size, int sm, packing_format format) {
            // For NPPDx, we focus on tile size and architecture support
            return is_valid_tile_v(tile_size, sm, format) && (sm >= 700);
        }
    } // namespace detail

    // Check if a description is supported on a given CUDA architecture
    template<class Description, unsigned int Architecture>
    struct is_supported:
        public NPPDX_STD::bool_constant<detail::is_supported_impl(
            uint2 {tile_size_of_v<Description>.x, tile_size_of_v<Description>.y}, int(Architecture),
            packing_format_of_v<Description>)> {
    };

    template<class Description, unsigned int Architecture>
    inline constexpr bool is_supported_v = is_supported<Description, Architecture>::value;
} // namespace nppdx

#endif // NPPDX_DETAIL_NPPDX_IS_SUPPORTED_HPP
