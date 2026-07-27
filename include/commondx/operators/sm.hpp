/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMONDX_OPERATORS_SM_HPP
#define COMMONDX_OPERATORS_SM_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

// Namespace wrappers
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {

    namespace detail {
        template<unsigned Architecture>
        __device__ __host__
        constexpr unsigned forward_sm_if_compatible() {
        #ifdef __CUDACC_VER_MAJOR__
            static_assert(Architecture != 1010 || __CUDACC_VER_MAJOR__ <= 12, "Incorrect SM<1010> value, use SM<1100> instead. In CUDA 13 sm_101 (Thor) was renamed to sm_110");
            static_assert(Architecture != 1100 || __CUDACC_VER_MAJOR__ >= 13, "Incorrect SM<1100> value, use SM<1010> instead. In CUDA 13 sm_101 (Thor) was renamed to sm_110");
        #endif

            return Architecture;
        }
    }

    template<unsigned int Architecture, class = void>
    struct SM;

    template<>
    struct SM<700>: public detail::constant_operator_expression<unsigned int, 700> {};

    template<>
    struct SM<720>: public detail::constant_operator_expression<unsigned int, 720> {
        [[deprecated("Use of SM<720> (Xavier architecture) has been deprecated and support will be removed in the future.")]] SM(){}
    };

    template<>
    struct SM<750>: public detail::constant_operator_expression<unsigned int, 750> {};

    template<>
    struct SM<800>: public detail::constant_operator_expression<unsigned int, 800> {};

    template<>
    struct SM<860>: public detail::constant_operator_expression<unsigned int, 860> {};

    template<>
    struct SM<870>: public detail::constant_operator_expression<unsigned int, 870> {};

    template<>
    struct SM<890>: public detail::constant_operator_expression<unsigned int, 890> {};

    template<>
    struct SM<900>: public detail::constant_operator_expression<unsigned int, 900> {};

    template<>
    struct SM<1000>: public detail::constant_operator_expression<unsigned int, 1000> {};

    // SFINAE as C++ Thor rename WAR to run static_assert only when this is instantiated
    template<unsigned Architecture>
    struct SM<Architecture, COMMONDX_STL_NAMESPACE::enable_if_t<(Architecture == 1010) || (Architecture == 1100)>>
        : public detail::constant_operator_expression<unsigned int, detail::forward_sm_if_compatible<Architecture>()> {};

    template<>
    struct SM<1030>: public detail::constant_operator_expression<unsigned int, 1030> {};

    template<>
    struct SM<1200>: public detail::constant_operator_expression<unsigned int, 1200> {};

    template<>
    struct SM<1210>: public detail::constant_operator_expression<unsigned int, 1210> {};
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_SM_HPP
