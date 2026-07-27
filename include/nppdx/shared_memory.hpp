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

#ifndef NPPDX_SHARED_MEMORY_HPP
#define NPPDX_SHARED_MEMORY_HPP


#include "nppdx/detail/config.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/tile_storage.hpp"
#include "nppdx/traits.hpp"

namespace nppdx {
    namespace shared_memory {
        // Trait-aware shared-memory helpers.
        //
        // The leaf types (ChannelSlice, TileStorage) and the value-/type-based
        // byte/alignment helpers live in nppdx/tile_storage.hpp. This header layers the
        // Description-aware byte/alignment lookups on top of that
        namespace detail {

            // ========================================================================
            // Trait memory helpers (uses inputs_of_t, outputs_of_t, temp_of_t from nppdx_traits.hpp)
            // ========================================================================

            template<class Description>
            NPPDX_DECL_CFHD unsigned int get_inputs_bytes() {
                return tuple_storage_bytes<inputs_of_t<Description>>();
            }

            template<class Description>
            NPPDX_DECL_CFHD unsigned int get_outputs_bytes() {
                return tuple_storage_bytes<outputs_of_t<Description>>();
            }

            template<class Description>
            NPPDX_DECL_CFHD unsigned int get_temp_bytes() {
                return tuple_storage_bytes<temp_of_t<Description>>();
            }

            template<class Description>
            NPPDX_DECL_CFHD unsigned int get_inputs_alignment() {
                return tuple_alignment<inputs_of_t<Description>>();
            }

            template<class Description>
            NPPDX_DECL_CFHD unsigned int get_outputs_alignment() {
                return tuple_alignment<outputs_of_t<Description>>();
            }

            template<class Description>
            NPPDX_DECL_CFHD unsigned int get_temp_alignment() {
                return tuple_alignment<temp_of_t<Description>>();
            }

            template<class Description>
            NPPDX_DECL_CFHD unsigned int get_io_alignment() {
                return max_alignment(get_inputs_alignment<Description>(), get_outputs_alignment<Description>());
            }

            template<class... Descriptions>
            NPPDX_DECL_CFHD unsigned int get_max_io_alignment() {
                if constexpr (sizeof...(Descriptions) == 0) {
                    return 1;
                } else {
                    return max_alignment(get_io_alignment<Descriptions>()...);
                }
            }

            template<class... Descriptions>
            NPPDX_DECL_CFHD unsigned int get_max_temp_bytes() {
                if constexpr (sizeof...(Descriptions) == 0) {
                    return 0;
                } else {
                    constexpr unsigned int temps[]  = {get_temp_bytes<Descriptions>()...};
                    unsigned int           max_temp = 0;
                    for (unsigned int i = 0; i < sizeof...(Descriptions); i++) {
                        if (temps[i] > max_temp) {
                            max_temp = temps[i];
                        }
                    }
                    return max_temp;
                }
            }

            template<class... Descriptions>
            NPPDX_DECL_CFHD unsigned int get_max_temp_alignment() {
                if constexpr (sizeof...(Descriptions) == 0) {
                    return 1;
                } else {
                    return max_alignment(get_temp_alignment<Descriptions>()...);
                }
            }

            template<class... Descriptions>
            NPPDX_DECL_CFHD unsigned int get_max_io_bytes() {
                if constexpr (sizeof...(Descriptions) == 0) {
                    return 0;
                } else {
                    constexpr unsigned int io_sizes[] = {
                        (get_inputs_bytes<Descriptions>() > get_outputs_bytes<Descriptions>()
                             ? get_inputs_bytes<Descriptions>()
                             : get_outputs_bytes<Descriptions>())...};
                    unsigned int max_io = 0;
                    for (unsigned int i = 0; i < sizeof...(Descriptions); i++) {
                        if (io_sizes[i] > max_io) {
                            max_io = io_sizes[i];
                        }
                    }
                    return max_io;
                }
            }

            template<class Inputs, class Outputs, class Temp>
            NPPDX_DECL_CFHD unsigned int compute_operation_storage_from_tuples() {
                constexpr unsigned int in_bytes  = tuple_storage_bytes<Inputs>();
                constexpr unsigned int out_bytes = tuple_storage_bytes<Outputs>();
                constexpr unsigned int tmp_bytes = tuple_storage_bytes<Temp>();
                constexpr unsigned int alignment = max_alignment(
                    max_alignment(tuple_alignment<Inputs>(), tuple_alignment<Outputs>()), tuple_alignment<Temp>());
                return compute_aligned_storage(alignment, in_bytes, out_bytes, tmp_bytes);
            }

        } // namespace detail

        template<class Description>
        NPPDX_DECL_CFHD unsigned int compute_operation_storage() {
            return detail::compute_operation_storage_from_tuples<inputs_of_t<Description>, outputs_of_t<Description>,
                                                                 temp_of_t<Description>>();
        }

        template<class... Descriptions>
        NPPDX_DECL_CFHD unsigned int get_pipeline_shared_storage() {
            constexpr unsigned int io_bytes   = detail::get_max_io_bytes<Descriptions...>() * 2;
            constexpr unsigned int temp_bytes = detail::get_max_temp_bytes<Descriptions...>();
            constexpr unsigned int alignment  = detail::max_alignment(detail::get_max_io_alignment<Descriptions...>(),
                                                                      detail::get_max_temp_alignment<Descriptions...>());
            return detail::compute_aligned_storage(alignment, io_bytes, temp_bytes);
        }
    } // namespace shared_memory
} // namespace nppdx

#endif // NPPDX_SHARED_MEMORY_HPP
