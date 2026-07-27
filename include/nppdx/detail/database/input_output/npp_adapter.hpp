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

#ifndef NPPDX_DETAIL_DATABASE_NPP_ADAPTER_HPP
#define NPPDX_DETAIL_DATABASE_NPP_ADAPTER_HPP

#include "nppdx/operators/formats.hpp"
#include "nppdx/detail/database/region.hpp"
#include "nppdx/detail/database/cell_data.hpp"
#include "nppdx/detail/backend/ingest_exgest_operations.hpp"
#include "nppdx/detail/database/cell_config.hpp"
#include "nppdx/detail/config.hpp"

#include NPPDX_STD_INCLUDE_CSTDINT
#include NPPDX_STD_INCLUDE_TYPE_TRAITS

namespace nppdx {
    namespace detail {

        template<packing_format Format>
        struct nppdx_safe_functions {
            template<typename CellCfg>
            __forceinline__ __device__ static void safe_load_cell(const uint8_t** src, const size_t* stride,
                                                                  NPPCellData<CellCfg, float>& cell_data,
                                                                  int2 cell_coord, uint2 image_size) {

                static_assert(
                    backend::format_backend<Format>::is_implemented,
                    "Backend not implemented for this format - check which packing_format enum value is being used");

                backend::format_backend<Format>::safe_load(src, stride, cell_data, cell_coord.x, cell_coord.y,
                                                           image_size.x, image_size.y);
            }

            template<typename CellCfg>
            __forceinline__ __device__ static void safe_save_cell(uint8_t** dst, const size_t* stride,
                                                                  const NPPCellData<CellCfg, const float>& cell_data,
                                                                  int2 cell_coord, uint2 image_size) {

                static_assert(
                    backend::format_backend<Format>::is_implemented,
                    "Backend not implemented for this format - check which packing_format enum value is being used");

                // clip=true: 8-bit exgest applies fclampf (same contract as NPP color paths that saturate to valid
                // range). Used for all benchmarks including rgb24 identity vs NPP.
                backend::format_backend<Format>::safe_save(dst, stride, true, 1.0f, cell_data, cell_coord.x,
                                                           cell_coord.y, image_size.x, image_size.y);
            }
        };

    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_DATABASE_NPP_ADAPTER_HPP