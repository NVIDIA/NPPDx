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

#ifndef NPPDX_EXAMPLE_COMMON_DEVICE_IO_HPP
#define NPPDX_EXAMPLE_COMMON_DEVICE_IO_HPP

namespace common {
    // Perform cooperative threadblock-wide copy from src
    // (arbitrary memory space) to dst (also arbitrary). This is not
    // an efficient implementation, used just for correctness.
    template<int Size, typename Prec>
    __device__ __forceinline__ void block_copy(const Prec* src, Prec* dst) {
        const auto threads = blockDim.x * blockDim.y * blockDim.z;
        const auto tid     = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * (blockDim.x) + threadIdx.x;

#pragma unroll
        for (int i = tid; i < Size; i += threads) {
            dst[i] = src[i];
        }
    }

    // Perform single-thread copy from src
    // (arbitrary memory space) to dst (also arbitrary). This is a
    // very efficient implementation, for better results stage through
    // shared memory to achieve higher global memory coalescing.
    template<int Size, typename Prec>
    __device__ __forceinline__ void thread_copy(const Prec* src, Prec* dst) {
#pragma unroll
        for (int i = 0; i < Size; ++i) {
            dst[i] = src[i];
        }
    }
} // namespace common

#endif // NPPDX_EXAMPLE_COMMON_KERNELS_HPP
