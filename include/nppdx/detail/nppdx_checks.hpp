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

#ifndef NPPDX_DETAIL_NPPDX_CHECKS_HPP
#define NPPDX_DETAIL_NPPDX_CHECKS_HPP

#include "commondx/device_info_shared_memory.hpp"
#include "nppdx/detail/decl.hpp"

namespace nppdx {
    namespace detail {


        // Runtime SM encoding (e.g. 860 for 8.6) -> max shared memory per block.
        NPPDX_DECL_NCHD inline unsigned int sm_shared_memory_size(int sm) {
            return commondx::sm_shared_memory_size(static_cast<unsigned int>(sm));
        }

        // NPPDx-specific validation utilities
        // Tile size and format validation is handled by the operation database
        NPPDX_DECL_NCHD bool is_valid_sm(int sm) {
            switch (sm) {
                case 700:
                case 720:
                case 750:
                case 800:
                case 860:
                case 870:
                case 890:
                case 900:
                case 1000:
                case 1010:
                case 1030:
                case 1100:
                case 1200:
                case 1210: return true;
                default: return false;
            }
        }

    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_NPPDX_CHECKS_HPP
