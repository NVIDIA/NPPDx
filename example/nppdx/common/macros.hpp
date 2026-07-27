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

#ifndef NPPDX_EXAMPLE_COMMON_MACROS_HPP
#define NPPDX_EXAMPLE_COMMON_MACROS_HPP

#include <iostream>

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif // CUDA_CHECK_AND_EXIT

#ifndef CUDA_CHECK
#    define CUDA_CHECK(error)                                                                               \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
            }                                                                                               \
        }
#endif // CUDA_CHECK

#ifndef NPP_CHECK
#    define NPP_CHECK(error)                                                                            \
        {                                                                                               \
            auto status = static_cast<NppStatus>(error);                                                \
            if (status != NPP_SUCCESS) {                                                                \
                std::cout << "NPP error:" << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(1);                                                                           \
            }                                                                                           \
        }

#endif // NPP_CHECK


#endif // NPPDX_EXAMPLE_COMMON_MACROS_HPP
