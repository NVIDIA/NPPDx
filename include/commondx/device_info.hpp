/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMONDX_DEVICE_INFO_HPP
#define COMMONDX_DEVICE_INFO_HPP

#include <cuda_fp16.h>

#include "commondx/detail/macros.hpp"
#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/device_info_shared_memory.hpp"
#include "commondx/complex_types.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {

    template<unsigned int SM>
    struct device_info {
        template<typename DataType>
        inline static constexpr bool is_mma_available() {
            // 7X0
            if constexpr (SM < 800) {
                return COMMONDX_STL_NAMESPACE::is_same_v<DataType, __half>;
            // 8X0
            } else if constexpr (SM < 900) {
                return COMMONDX_STL_NAMESPACE::is_same_v<DataType, __half> ||
                       COMMONDX_STL_NAMESPACE::is_same_v<DataType, double> ||
                       COMMONDX_STL_NAMESPACE::is_same_v<DataType, complex<double>>;
            // 9X0
            } else if constexpr (SM < 1000) {
                return COMMONDX_STL_NAMESPACE::is_same_v<DataType, __half> ||
                       COMMONDX_STL_NAMESPACE::is_same_v<DataType, double> ||
                       COMMONDX_STL_NAMESPACE::is_same_v<DataType, complex<double>>;
            // 1X00
            } else if constexpr (SM < 2000) {
                return COMMONDX_STL_NAMESPACE::is_same_v<DataType, __half> ||
                    COMMONDX_STL_NAMESPACE::is_same_v<DataType, double> ||
                    COMMONDX_STL_NAMESPACE::is_same_v<DataType, complex<double>>;
            }
            // Next SM
            return false;
        }

        inline static constexpr unsigned int shared_memory() {
            return sm_shared_memory_size(SM);
        }

    };
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_DEVICE_INFO_HPP
