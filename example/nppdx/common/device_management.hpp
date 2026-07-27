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

#ifndef DEVICE_MANAGEMENT_HPP
#define DEVICE_MANAGEMENT_HPP

#ifdef NPPDX_CLANG_CUDA_COMPAT

#    include "cuda_compat/device_management.hpp"

#else

#    include <cstddef>
#    include <cstdint>

#    include <cuda_runtime.h>

#    include "error_checking.hpp"
#    include "macros.hpp"

namespace common {
    inline unsigned int get_cuda_device_arch() {
        int device;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));

        int major = 0;
        int minor = 0;
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

        return static_cast<unsigned int>(major) * 100 + static_cast<unsigned int>(minor) * 10;
    }

    namespace cuda_runtime {
        struct DevBuf {
            uint8_t* d_buf;

            // Prevent double-free by making non-copyable and non-movable.
            // For more complex workflows, the move constructor/assignment operator should be defined.
            DevBuf(const DevBuf&)            = delete;
            DevBuf& operator=(const DevBuf&) = delete;
            DevBuf(DevBuf&&)                 = delete;
            DevBuf& operator=(DevBuf&&)      = delete;

            DevBuf(std::size_t size): d_buf(nullptr) { CUDA_CHECK_AND_EXIT(cudaMalloc(&d_buf, size)); }

            ~DevBuf() { CUDA_CHECK_AND_EXIT(cudaFree(d_buf)); }
        };

        struct CudaStream {
            cudaStream_t stream;

            // Prevent double-free by making non-copyable and non-movable.
            // For more complex workflows, the move constructor/assignment operator should be defined.
            CudaStream(const CudaStream&)            = delete;
            CudaStream& operator=(const CudaStream&) = delete;
            CudaStream(CudaStream&&)                 = delete;
            CudaStream& operator=(CudaStream&&)      = delete;

            CudaStream() { CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream)); }

            ~CudaStream() { CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream)); }

            operator cudaStream_t() const { return stream; }
        };
    } // namespace cuda_runtime

    using DevBuf     = cuda_runtime::DevBuf;
    using CudaStream = cuda_runtime::CudaStream;
} // namespace common

#endif

#endif // DEVICE_MANAGEMENT_HPP
