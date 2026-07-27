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

#ifndef NPPDX_DETAIL_NPPDX_DESCRIPTION_ARITHMETIC_HPP
#define NPPDX_DETAIL_NPPDX_DESCRIPTION_ARITHMETIC_HPP

#include "nppdx/detail/nppdx_execution.hpp"
#include "nppdx/traits/detail/is_complete.hpp"

namespace nppdx {
    namespace detail {
        template<class... Operators>
        struct make_description {
        private:
            using description_type = nppdx_description<Operators...>;

            static constexpr bool is_complete = is_complete_description<description_type>::value;


            // Workaround (NVRTC/MSVC)
            //
            // For NVRTC we need to utilize conditional types to avoid compilation errors
            // when Block() is added to description before description is complete.
            //
            // This workaround disables some useful diagnostics based on static_asserts.
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
            using operator_wrapper_type = nppdx_operator_wrapper<Operators...>;
            using execution_type =
                NPPDX_STD::conditional_t<is_complete_description<operator_wrapper_type>::value,
                                         nppdx_execution<Operators...>, nppdx_execution_partial<Operators...>>;
#else
            using execution_type = nppdx_execution<Operators...>;
#endif

        public:
            using type = typename NPPDX_STD::conditional_t<is_complete, execution_type, description_type>;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;
    } // namespace detail

    template<class Operator1, class Operator2>
    NPPDX_HOST_DEVICE_FORCEINLINE_FUNC auto operator+(const Operator1&, const Operator2&) //
        -> NPPDX_STD::enable_if_t<commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
                                  detail::make_description_t<Operator1, Operator2>> {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    NPPDX_HOST_DEVICE_FORCEINLINE_FUNC auto operator+(const detail::nppdx_description<Operators1...>&,
                                                      const Operator2&) //
        -> NPPDX_STD::enable_if_t<commondx::detail::is_operator_expression<Operator2>::value,
                                  detail::make_description_t<Operators1..., Operator2>> {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    NPPDX_HOST_DEVICE_FORCEINLINE_FUNC auto operator+(const Operator1&,
                                                      const detail::nppdx_description<Operators2...>&) //
        -> NPPDX_STD::enable_if_t<commondx::detail::is_operator_expression<Operator1>::value,
                                  detail::make_description_t<Operator1, Operators2...>> {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    NPPDX_HOST_DEVICE_FORCEINLINE_FUNC auto operator+(const detail::nppdx_description<Operators1...>&,
                                                      const detail::nppdx_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }

} // namespace nppdx
#endif // NPPDX_INCLUDE_NPPDX_DETAIL_DESCRIPTION_ARITHMETIC_HPP
