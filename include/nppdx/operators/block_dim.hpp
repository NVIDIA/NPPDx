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

#ifndef NPPDX_OPERATORS_BLOCKDIM_HPP
#define NPPDX_OPERATORS_BLOCKDIM_HPP

#include "commondx/operators/block_dim.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "nppdx/operators/operator_type.hpp"
#include "nppdx/detail/config.hpp"

namespace nppdx {
    // Import BlockDim operator from commonDx
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct BlockDim: public commondx::BlockDim<X, Y, Z> {
        static_assert(Y == 1 && Z == 1, "NPPDx only supports 1D thread blocks, use BlockDim<X, 1, 1>");
        static constexpr dim3 value = dim3 {X, Y, Z};
    };
} // namespace nppdx

namespace commondx::detail {
    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::block_dim, nppdx::BlockDim<X, Y, Z>>:
        NPPDX_STD::true_type {
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct get_operator_type<nppdx::operator_type, nppdx::BlockDim<X, Y, Z>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::block_dim;
    };
} // namespace commondx::detail

#endif // NPPDX_OPERATORS_BLOCKDIM_HPP
