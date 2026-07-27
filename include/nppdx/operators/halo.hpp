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

#ifndef NPPDX_OPERATORS_HALO_HPP
#define NPPDX_OPERATORS_HALO_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"
#include "nppdx/operators/operator_type.hpp"
#include "nppdx/detail/config.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/types.hpp"
#include "nppdx/utils.hpp"

namespace nppdx {
    struct Halo4 {
        int2 left_top;
        int2 right_bottom;

        // Horizontal computes the total number of pixel that belong to halo inside the tile in the x direction
        NPPDX_DECL_NCFHD int horizontal() const { return left_top.x + right_bottom.x; }
        // Vertical computes the total number of pixel that belong to halo inside the tile in the y direction
        NPPDX_DECL_NCFHD int vertical() const { return left_top.y + right_bottom.y; }

        NPPDX_DECL_NCFHD int2 total() const { return left_top + right_bottom; }

        NPPDX_DECL_NCFHD bool empty() const { return left_top.is_zero() && right_bottom.is_zero(); }

        NPPDX_DECL_NSCFHD Halo4 make_uniform(int value) { return Halo4 {int2::uniform(value), int2::uniform(value)}; }

        NPPDX_DECL_NSCFHD Halo4 make_empty() { return make_uniform(0); }

        [[nodiscard]] friend NPPDX_DECL_CFHD Halo4 operator+(const Halo4& a, const Halo4& b) {
            return {a.left_top + b.left_top, a.right_bottom + b.right_bottom};
        }

        [[nodiscard]] friend NPPDX_DECL_CFHD Halo4 operator-(const Halo4& a, const Halo4& b) {
            return {a.left_top - b.left_top, a.right_bottom - b.right_bottom};
        }

        [[nodiscard]] friend NPPDX_DECL_CFHD bool operator==(const Halo4& a, const Halo4& b) {
            return all(a.left_top == b.left_top) && all(a.right_bottom == b.right_bottom);
        }

        [[nodiscard]] friend NPPDX_DECL_CFHD bool operator!=(const Halo4& a, const Halo4& b) { return !(a == b); }
    };

    namespace detail {
        NPPDX_DECL_NCFHD int2 scale_halo_component(int2 value, int num, int den) {
            using ll2 = vec2<long long>;
            return int2((ll2(value) * ll2(int2::uniform(num))).dir_quot<-1>(ll2(int2::uniform(den))));
        }
    } // namespace detail

    NPPDX_DECL_NCFHD Halo4 scale_halo(Halo4 halo, int num, int den) {
        return Halo4 {detail::scale_halo_component(halo.left_top, num, den),
                      detail::scale_halo_component(halo.right_bottom, num, den)};
    }

    namespace detail {
        template<typename TagT, typename LeftTopT, typename RightBottomT>
        struct GenericHalo: public commondx::detail::operator_expression {
            static_assert(IsInt2D<LeftTopT>);
            static_assert(IsInt2D<RightBottomT>);

            NPPDX_DECL_SC Halo4 value = {LeftTopT::value, RightBottomT::value};

            using LeftTop     = LeftTopT;
            using RightBottom = RightBottomT;
        };

        template<typename TagT>
        using EmptyGenericHalo = GenericHalo<TagT, Int2D<0, 0>, Int2D<0, 0>>;

        template<typename T, typename TagT>
        NPPDX_DECL_CI bool IsHalo = false;
        template<typename TagT, typename LeftTopT, typename RightBottomT>
        NPPDX_DECL_CI bool IsHalo<GenericHalo<TagT, LeftTopT, RightBottomT>, TagT> = true;

        struct MemoryHaloTag {
        };
        struct CumulativeHaloTag {
        };
    } // namespace detail


    template<typename LeftTopT, typename RightBottomT>
    using MemoryHalo      = detail::GenericHalo<detail::MemoryHaloTag, LeftTopT, RightBottomT>;
    using EmptyMemoryHalo = detail::EmptyGenericHalo<detail::MemoryHaloTag>;
    template<typename T>
    NPPDX_DECL_CI bool IsMemoryHalo = detail::IsHalo<T, detail::MemoryHaloTag>;

    template<typename LeftTopT, typename RightBottomT>
    using CumulativeHalo      = detail::GenericHalo<detail::CumulativeHaloTag, LeftTopT, RightBottomT>;
    using EmptyCumulativeHalo = detail::EmptyGenericHalo<detail::CumulativeHaloTag>;
    template<typename T>
    NPPDX_DECL_CI bool IsCumulativeHalo = detail::IsHalo<T, detail::CumulativeHaloTag>;

#define NPPDX_MAKE_HALO(HaloOpType, Halo4Val) \
    HaloOpType<NPPDX_MAKE_INT2D((Halo4Val).left_top), NPPDX_MAKE_INT2D((Halo4Val).right_bottom)>

} // namespace nppdx

namespace commondx::detail {
    template<typename LeftTop, typename RightBottom>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::memory_halo,
                       nppdx::MemoryHalo<LeftTop, RightBottom>>: NPPDX_STD::true_type {
    };

    template<typename LeftTop, typename RightBottom>
    struct get_operator_type<nppdx::operator_type, nppdx::MemoryHalo<LeftTop, RightBottom>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::memory_halo;
    };

    template<typename LeftTop, typename RightBottom>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::cumulative_halo,
                       nppdx::CumulativeHalo<LeftTop, RightBottom>>: NPPDX_STD::true_type {
    };

    template<typename LeftTop, typename RightBottom>
    struct get_operator_type<nppdx::operator_type, nppdx::CumulativeHalo<LeftTop, RightBottom>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::cumulative_halo;
    };
} // namespace commondx::detail

#endif // NPPDX_OPERATORS_HALO_HPP
