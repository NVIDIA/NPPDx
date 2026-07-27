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

#ifndef NPPDX_OPERATORS_TILE_SIZE_HPP
#define NPPDX_OPERATORS_TILE_SIZE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"
#include "nppdx/operators/operator_type.hpp"
#include "nppdx/detail/config.hpp"
#include "nppdx/types.hpp"

namespace nppdx {

    // TileSize operator
    template<uint X, uint Y>
    struct TileSize: public UInt2D<X, Y>, public commondx::detail::operator_expression {
        static_assert(X > 0, "First dimension must be greater than 0");
        static_assert(Y > 0, "Second dimension size must be greater than 0");
    };

#define NPPDX_MAKE_TILE_SIZE(Val) NPPDX_MAKE_TP_VEC2D(::nppdx::TileSize, Val)

} // namespace nppdx

namespace commondx::detail {
    // TileSize operator specializations
    template<unsigned int X, unsigned int Y>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::tile_size, nppdx::TileSize<X, Y>>:
        NPPDX_STD::true_type {
    };

    template<unsigned int X, unsigned int Y>
    struct get_operator_type<nppdx::operator_type, nppdx::TileSize<X, Y>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::tile_size;
    };

} // namespace commondx::detail

#endif // NPPDX_OPERATORS_TILE_SIZE_HPP
