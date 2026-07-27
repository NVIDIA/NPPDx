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

#ifndef NPPDX_DETAIL_NPPDX_EXEC_OP_HPP
#define NPPDX_DETAIL_NPPDX_EXEC_OP_HPP



#include "nppdx/operators/operation_operator_traits.hpp"
#include "nppdx/utils.hpp"
#include "nppdx/operators/function.hpp"
#include "nppdx/operators/input_output.hpp"
#include "nppdx/operators/halo.hpp"
#include "nppdx/operators/tile_layout.hpp"
#include "nppdx/traits/nppdx_traits.hpp"
#include "nppdx/detail/database/storage_config.hpp"

namespace nppdx {


    // Note: The extra typename template parameter is a workaround for gcc7's
    // defective handling of enum template parameter.
    template<typename TagType>
    struct ExecOperator {
        int                   sm;
        simple_optional<dim3> block_dim_opt;

        layout_props layout;

        using tag_type     = TagType;
        using op_op_traits = check_t<operation_operator_traits_checker, operation_operator_traits<TagType>>;
        using op_data_type = typename op_op_traits::value_type;
        op_data_type op_data;

        NPPDX_DECL_NCHD Halo4 local_halo() const { return op_op_traits::get_local_halo(op_data); }

        NPPDX_DECL_NCHD detail::storage_config storage_config() const {
            return op_op_traits::get_storage_config(op_data, layout);
        }

        NPPDX_DECL_NCHD dim3 block_dim() const { return block_dim_opt.has_value() ? block_dim_opt.value() : dim3(256); }

        NPPDX_DECL_NCHD unsigned int block_threads() const {
            const dim3& bd = block_dim();
            return bd.x * bd.y * bd.z;
        }
    };

} // namespace nppdx

#endif // NPPDX_DETAIL_NPPDX_EXEC_OP_HPP