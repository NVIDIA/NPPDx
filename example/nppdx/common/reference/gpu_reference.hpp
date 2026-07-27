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

#ifndef NPPDX_EXAMPLE_NPPDX_COMMON_REFERENCE_GPU_REFERENCE_HPP
#define NPPDX_EXAMPLE_NPPDX_COMMON_REFERENCE_GPU_REFERENCE_HPP

#include <type_traits>

#include <thrust/device_vector.h>

#include "../numeric.hpp"

namespace common {

    template<typename OutType, class Op>
    struct apply_and_convert {
        template<class T>
        __host__ __device__ __forceinline__ OutType operator()(const T& val) {
            Op op;
            return op(convert<OutType>(val));
        }
    };

    template<typename NPPDX, class XTransform, class YTransform, class ZTransform, typename TX, typename TY,
             typename TZ>
    auto run_gpu_reference_correctness(const thrust::device_vector<TX>& input_x,
                                       const thrust::device_vector<TY>& input_y,
                                       const thrust::device_vector<TZ>& input_z,
                                       [[maybe_unused]] const int       batches = 1) {


        return std::make_tuple(thrust::host_vector<TX>(input_x), thrust::host_vector<TY>(input_y),
                               thrust::host_vector<TZ>(input_z));
    }
} // namespace common

#endif // NPPDX_TEST_COMMON_REFERENCE_GPU_REFERENCE_HPP
