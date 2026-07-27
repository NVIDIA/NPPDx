/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMONDX_OPERATORS_EXECUTION_OPERATORS_HPP
#define COMMONDX_OPERATORS_EXECUTION_OPERATORS_HPP

#include "commondx/detail/expressions.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    struct Thread:  detail::operator_expression {};
    struct Warp:    detail::operator_expression {};
    struct Block:   detail::operator_expression {};
    struct Cluster: detail::operator_expression {};
    struct Device:  detail::operator_expression {};
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_EXECUTION_OPERATORS_HPP
