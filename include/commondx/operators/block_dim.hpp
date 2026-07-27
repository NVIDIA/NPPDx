/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMONDX_OPERATORS_BLOCK_DIM_HPP
#define COMMONDX_OPERATORS_BLOCK_DIM_HPP

#include "commondx/detail/expressions.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct BlockDim: public detail::block_operator_expression {
        static constexpr unsigned int x     = X;
        static constexpr unsigned int y     = Y;
        static constexpr unsigned int z     = Z;
        static constexpr dim3         value = dim3(x, y, z);

        static constexpr unsigned int flat_size = x * y * z;
        static constexpr unsigned int rank      = (z != 1) ? 3 : ((y != 1) ? 2 : 1);
    };

    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct GridDim: public BlockDim<X, Y, Z> {
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int BlockDim<X, Y, Z>::x;
    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int BlockDim<X, Y, Z>::y;
    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int BlockDim<X, Y, Z>::z;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr dim3 BlockDim<X, Y, Z>::value;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int BlockDim<X, Y, Z>::flat_size;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int BlockDim<X, Y, Z>::rank;

} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_BLOCK_DIM_HPP
