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

#ifndef NPPDX_DETAIL_DATABASE_OPERATION_COMMON_HPP
#define NPPDX_DETAIL_DATABASE_OPERATION_COMMON_HPP

#include "nppdx/detail/database/cell_config.hpp"
#include "nppdx/detail/database/cell_data.hpp"
#include "nppdx/detail/database/region.hpp"
#include "nppdx/detail/database/cell_executor.hpp"
#include "nppdx/operators/operation_operator_traits.hpp"
#include "nppdx/operators/formats.hpp"

namespace nppdx {
    namespace detail {

        template<typename OpDesc, typename Layout, typename CellCfg, int SM, int BlockThreads, typename Enable = void>
        class op_impl
        {
            static_assert(!NPPDX_STD::is_same_v<OpDesc, OpDesc>, "unspecialized op_impl instantiated");
        };

        struct OpImplHelperBase {
            using processing_type                   = float;
            NPPDX_DECL_SC unsigned int num_channels = packing_format_traits<packing_format::internal>::channels;

            template<typename T>
            NPPDX_DECL_SC void check_data_type() {
                static_assert(NPPDX_STD::is_same_v<NPPDX_STD::remove_cv_t<T>, processing_type>,
                              "T must be the same as processing_type");
            }
        };

        // Base traits - common to ALL operations
        template<typename Operation, typename Layout, typename CellCfg>
        struct OpImplHelper: public OpImplHelperBase {
            using op_traits = operation_operator_traits<typename Operation::tag_type>;

            NPPDX_DECL_SC cell_config cell_cfg = CellCfg::value;

            NPPDX_DECL_SC uint2 input_nominal_size = Layout::value.nominal_size;
            NPPDX_DECL_SC uint2 output_nominal_size =
                op_traits::get_output_nominal_tile(Operation::value, input_nominal_size);
            NPPDX_DECL_SC storage_config storage = op_traits::get_storage_config(Operation::value, Layout::value);

            using inputs = NPPDX_MAKE_SLOT_STORAGE(
                processing_type, op_traits::get_storage_config(Operation::value, Layout::value).input);
            using outputs = NPPDX_MAKE_SLOT_STORAGE(
                processing_type, op_traits::get_storage_config(Operation::value, Layout::value).output);
            using temp = NPPDX_MAKE_SLOT_STORAGE(processing_type,
                                                 op_traits::get_storage_config(Operation::value, Layout::value).temp);

            using TileStorageType = NPPDX_STD::tuple_element_t<0, inputs>;
        };
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_DATABASE_OPERATION_COMMON_HPP
