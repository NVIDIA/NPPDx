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

#ifndef NPPDX_DETAIL_DATABASE_REGION_HPP
#define NPPDX_DETAIL_DATABASE_REGION_HPP

#include "nppdx/detail/decl.hpp"
#include "nppdx/operators/tile_layout.hpp"

namespace nppdx {
    namespace detail {

        NPPDX_DECLARE_TAGGED_TYPE(OuterPos, int2);
        NPPDX_DECLARE_TAGGED_TYPE(InnerPos, int2);

        // The region stucture represents a 2-dimensional inner subregion of a larger outer region.
        // It specifies the offset in pixels of the inner region's top-left-most pixel relative to the outer region's
        // top-left-most pixel, and the size of the inner region.

        struct region {
            int2  offset;
            uint2 size;

            NPPDX_DECL_NCHD OuterPos outer_left_top_pos() const { return OuterPos(offset); }
            NPPDX_DECL_NCHD OuterPos outer_right_bottom_pos() const { return OuterPos(offset + int2(size)); }
            NPPDX_DECL_NCHD bool     contains(OuterPos pos) const {
                return all(pos >= outer_left_top_pos() && pos < outer_right_bottom_pos());
            }

            NPPDX_DECL_NCHD region operator+(Halo4 halo) const {
                return {offset - halo.left_top, uint2(int2(size) + halo.total())};
            }

            NPPDX_DECL_NCHD region operator-(Halo4 halo) const {
                return {offset + halo.left_top, uint2(int2(size) - halo.total())};
            }

            NPPDX_DECL_NCHD OuterPos outer_pos(InnerPos pos) const { return OuterPos(offset + pos); }

            NPPDX_DECL_NCHD InnerPos inner_pos(OuterPos pos) const { return InnerPos(pos - offset); }
        };

        NPPDX_DECL_NCHD region make_input_region(const layout_props& props) {
            return {props.padding_halo().left_top, props.extended_size()};
        }

        NPPDX_DECL_NCHD region make_output_region(const layout_props& props, const Halo4& local_halo) {
            return make_input_region(props) - local_halo;
        }

        namespace tp {
            template<typename OffsetT, typename SizeT>
            struct Region {
                static_assert(IsInt2D<OffsetT>);
                static_assert(IsUInt2D<SizeT>);

                NPPDX_DECL_SC region value = {OffsetT::value, SizeT::value};

                using Offset = OffsetT;
                using Size = SizeT;
            };

            template<typename T>
            NPPDX_DECL_CI bool IsRegion = false;
            template<typename OffsetT, typename SizeT>
            NPPDX_DECL_CI bool IsRegion<Region<OffsetT, SizeT>> = true;

#define NPPDX_MAKE_REGION(RegionVal) \
            ::nppdx::detail::tp::Region<NPPDX_MAKE_INT2D(RegionVal.offset), NPPDX_MAKE_UINT2D(RegionVal.size)>
        }
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_DATABASE_REGION_HPP
