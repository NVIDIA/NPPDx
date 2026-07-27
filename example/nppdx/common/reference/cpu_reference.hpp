/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NPPDX_EXAMPLE_NPPDX_COMMON_REFERENCE_CPU_REFERENCE_HPP
#define NPPDX_EXAMPLE_NPPDX_COMMON_REFERENCE_CPU_REFERENCE_HPP

#include <type_traits>

#include <thrust/host_vector.h>

#include "../numeric.hpp"

namespace common {
    template<typename NPPDX, class XTransform, class YTransform, class ZTransform, typename TX, typename TY,
             typename TZ>
    auto run_cpu_reference_correctness(const thrust::host_vector<TX>& input_x, const thrust::host_vector<TY>& input_y,
                                       const thrust::host_vector<TZ>& input_z, const int batches = 1) {

        return std::make_tuple(input_x, input_y, input_z);
    }
} // namespace common

#endif // NPPDX_EXAMPLE_NPPDX_COMMON_REFERENCE_CPU_REFERENCE_HPP
