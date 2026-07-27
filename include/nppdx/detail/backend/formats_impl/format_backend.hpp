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

#ifndef NPPDX_DETAIL_BACKEND_INGEST_EXGEST_FORMAT_BACKEND_HPP
#define NPPDX_DETAIL_BACKEND_INGEST_EXGEST_FORMAT_BACKEND_HPP

#include "nppdx/operators/formats.hpp"
#include "nppdx/detail/database/cell_data.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {

            template<packing_format Format>
            struct format_backend {
                static constexpr bool is_implemented = false;
            };

            template<>
            struct format_backend<packing_format::none> {
                static constexpr bool is_implemented = false;
            };

            template<>
            struct format_backend<packing_format::internal> {
                static constexpr bool is_implemented = false;
            };

            using cell_4x2_3ch_float = NPPCellData<CellConfig<UInt2D<4, 2>, 3>, float>;
            using cell_4x2_3ch_const_float = NPPCellData<CellConfig<UInt2D<4, 2>, 3>, const float>;

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_INGEST_EXGEST_FORMAT_BACKEND_HPP