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

#ifndef NPPDX_EXAMPLE_COMMON_PRINT_HPP
#define NPPDX_EXAMPLE_COMMON_PRINT_HPP

#include <stdio.h>

#include <nppdx.hpp>
#include <numeric>

namespace common {

    __forceinline__ __device__ bool block0() {
        return (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) == 0;
    }

    __forceinline__ __device__ bool thread0() {
        return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) == 0;
    }


    template<typename NPPDX>
    __forceinline__ __device__ void print(NPPDX) {
        if (thread0() && block0()) {
            printf("\nOPERATORS =======================================================\n");
            printf("\n");
            printf("BlockDim: \n");
            if constexpr (nppdx::detail::has_operator_v<nppdx::operator_type::block_dim, NPPDX>) {
                auto bd = nppdx::block_dim_of_v<NPPDX>;
                printf("Value: %u %u %u \n\n", bd.x, bd.y, bd.z);
            } else {
                printf("Value: Absent\n\n");
            }

            printf("InputFormat: \n");
            if constexpr (nppdx::detail::has_operator_v<nppdx::operator_type::input_format, NPPDX>) {
                auto input_format = nppdx::input_format_of_v<NPPDX>;
                printf("Value: %d \n\n", static_cast<int>(input_format));
            } else {
                printf("Value: Absent\n\n");
            }

            printf("OutputFormat: \n");
            if constexpr (nppdx::detail::has_operator_v<nppdx::operator_type::output_format, NPPDX>) {
                auto output_format = nppdx::output_format_of_v<NPPDX>;
                printf("Value: %d \n\n", static_cast<int>(output_format));
            } else {
                printf("Value: Absent\n\n");
            }

            printf("InputOutput: \n");
            if constexpr (nppdx::detail::has_operator_v<nppdx::operator_type::input_output, NPPDX>) {
                auto input_output = nppdx::input_output_direction_of_v<NPPDX>;
                printf("Value: %d \n\n", static_cast<int>(input_output));
            } else {
                printf("Value: Absent\n\n");
            }

            printf("TileSize: \n");
            if constexpr (nppdx::detail::has_operator_v<nppdx::operator_type::tile_size, NPPDX>) {
                auto tile_size = nppdx::tile_size_of_v<NPPDX>;
                printf("Value: %u %u \n\n", tile_size.x, tile_size.y);
            } else {
                printf("Value: Absent\n\n");
            }


            // Size operator was removed in simplified implementation

            printf("SM: \n");
            if constexpr (nppdx::detail::has_operator_v<nppdx::operator_type::sm, NPPDX>) {
                auto sm_v = nppdx::sm_of_v<NPPDX>;
                printf("Value: %u \n\n", sm_v);
            } else {
                printf("Value: Absent\n\n");
            }

            printf("Type: \n");
            if constexpr (nppdx::detail::has_operator_v<nppdx::operator_type::type, NPPDX>) {
                auto opt = nppdx::type_of_v<NPPDX>;
                auto msg = opt == nppdx::type::real ? "real" : "complex";
                printf("Value: %s \n\n", msg);
            } else {
                printf("Value: Absent\n\n");
            }

            printf("Is type complete? \n");
            auto msg = NPPDX::is_complete_v ? "Yes" : "No";
            printf("Value: %s \n \n", msg);

            printf("END =============================================================\n\n");
        }
    }

    __forceinline__ __host__ __device__ const char* interpolation_method_to_string(
        nppdx::interpolation_method interpolation_method) {
        switch (interpolation_method) {
#define NPPDX_CASE_INTERPOLATION_TO_STRING(method, ...) \
    case nppdx::interpolation_method::method: return #method;
            NPPDX_FOR_INTERPOLATION_METHODS(NPPDX_CASE_INTERPOLATION_TO_STRING)
#undef NPPDX_CASE_INTERPOLATION_TO_STRING
            default: return "unknown";
        }
    }
} // namespace common

#endif // NPPDX_EXAMPLE_COMMON_PRINT_HPP
