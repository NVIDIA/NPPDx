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

#ifndef NPPDX_OPERATORS_SM_HPP
#define NPPDX_OPERATORS_SM_HPP

#include "commondx/operators/sm.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

#include "nppdx/operators/operator_type.hpp"

namespace nppdx {
    // Import SM operator from commonDx
    template<unsigned int Architecture>
    struct SM: public commondx::SM<Architecture> {};
} // namespace nppdx

namespace commondx::detail {
    // Thread specializations
    template<unsigned int Architecture>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::sm, nppdx::SM<Architecture>>:
        NPPDX_STD::true_type {};

    template<unsigned int Architecture>
    struct get_operator_type<nppdx::operator_type, nppdx::SM<Architecture>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::sm;
    };
} // namespace commondx::detail

#endif // NPPDX_OPERATORS_SM_HPP
