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

#ifndef COMMONDX_OPERATORS_SIZE_HPP
#define COMMONDX_OPERATORS_SIZE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct Size3D: public detail::operator_expression {
        static_assert(X > 0, "First dimension must be greater than 0");
        static_assert(Y > 0, "Second dimension size must be greater than 0");
        static_assert(Z > 0, "Third dimension size must be greater than 0");

        static constexpr unsigned int x = X;
        static constexpr unsigned int y = Y;
        static constexpr unsigned int z = Z;

        static constexpr unsigned int flat_size = x * y * z;
        static constexpr unsigned int rank      = (x != 1) + (y != 1) + (z != 1);
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int Size3D<X, Y, Z>::x;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int Size3D<X, Y, Z>::y;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int Size3D<X, Y, Z>::z;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int Size3D<X, Y, Z>::flat_size;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int Size3D<X, Y, Z>::rank;
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_SIZE_HPP
