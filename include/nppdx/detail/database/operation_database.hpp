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

#ifndef NPPDX_DETAIL_DATABASE_OPERATION_DATABASE_HPP
#define NPPDX_DETAIL_DATABASE_OPERATION_DATABASE_HPP


#include "nppdx/operators/formats.hpp"
#include "commondx/detail/stl/type_traits.hpp"
#include "nppdx/detail/database/operation_implementation.hpp"
#include "nppdx/detail/nppdx_exec_op.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/detail/database/function/function_operation_implementation.hpp"
#include "nppdx/operators/tile_size.hpp"

namespace nppdx::detail {

    struct operation_config {
        cell_config  cell_cfg;
        unsigned int block_threads;
    };

    NPPDX_DECL_NCHD operation_config make_op_config(cell_config cell_cfg, uint2 tile_size) {
        unsigned int suggested_block_threads = get_number_of_cells(tile_size, cell_cfg.size);
        return operation_config {cell_cfg, suggested_block_threads};
    }

    template<typename TagType>
    NPPDX_DECL_NCHD operation_config get_recommended_op_config(const ExecOperator<TagType>& exec_op) {
        using TagEnumType = typename TagType::type;
        if constexpr (NPPDX_STD::is_same_v<TagEnumType, input_output_direction>) {
            const packing_format format = exec_op.op_data;
            return make_op_config(get_cell_config(exec_op.sm, format), exec_op.layout.nominal_size);
        } else {
            const layout_props& layout = exec_op.layout;
            const uint2         tile_size =
                exec_op.local_halo().empty() ? uint2(make_input_region(layout).size) : layout.memory_size();
            return make_op_config(get_cell_config(exec_op.sm, packing_format::internal), tile_size);
        }
    }

    template<typename TagType>
    NPPDX_DECL_NCHD auto normalize_exec_operator(const ExecOperator<TagType>& exec_op) {
        if (exec_op.block_dim_opt.has_value()) {
            return exec_op;
        }

        operation_config rec_op_cfg = get_recommended_op_config(exec_op);
        if (rec_op_cfg.block_threads == exec_op.block_threads()) {
            return exec_op;
        }

        return ExecOperator<TagType> {
            exec_op.sm,
            dim3(rec_op_cfg.block_threads),
            exec_op.layout,
            exec_op.op_data,
        };
    }

} // namespace nppdx::detail

#endif // NPPDX_DETAIL_DATABASE_OPERATION_DATABASE_HPP
