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

#ifndef COMMONDX_DEVICE_INFO_SHARED_MEMORY_HPP
#define COMMONDX_DEVICE_INFO_SHARED_MEMORY_HPP

#include "commondx/detail/macros.hpp"


// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {

    // Source for these chip memory numbers:
    // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#compute-capabilities-table-memory-information-per-compute-capability
    COMMONDX_HOST_DEVICE constexpr unsigned int sm_shared_memory_size(const unsigned int sm) {
        return [sm]() {
            switch (sm) {
                case 1210: return 99;
                case 1200: return 99;
                case 1030: return 227;
                case 1100: return 227;
                case 1010: return 227;
                case 1000: return 227;
                case 900: return 227;
                case 890: return 99;
                case 870: return 163;
                case 860: return 99;
                case 800: return 163;
                case 750: return 64;
                case 720: return 96;
                case 700: return 96;
                default: return 48;
            }
        }() * 1024;
    }
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_DEVICE_INFO_SHARED_MEMORY_HPP
