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

#ifndef COMMONDX_TRAITS_DX_TRAITS_HPP
#define COMMONDX_TRAITS_DX_TRAITS_HPP

#include "commondx/operators/block_dim.hpp"
#include "commondx/operators/execution_operators.hpp"
#include "commondx/operators/sm.hpp"
#include "commondx/operators/type.hpp"
#include "commondx/traits/detail/get.hpp"
#include "commondx/traits/detail/description_traits.hpp"
#include "commondx/detail/stl/type_traits.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    template<class OperatorTypeClass, class Description, class Default>
    struct precision_of {
        using type = typename detail::get_or_default_t<OperatorTypeClass, OperatorTypeClass::precision, Description, Default>::type;
    };

    template<class OperatorTypeClass, class Description, class Default = DataType<data_type::real>>
    struct data_type_of {
        using value_type                  = data_type;
        static constexpr value_type value = detail::get_or_default_t<OperatorTypeClass, OperatorTypeClass::type, Description, Default>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description, class Default>
    constexpr data_type data_type_of<OperatorTypeClass, Description, Default>::value;

    template<class OperatorTypeClass, class Description>
    inline constexpr data_type data_type_of_v = data_type_of<OperatorTypeClass, Description>::value;

    template<class OperatorTypeClass, class Description>
    struct sm_of {
    private:
        static constexpr bool has_sm = detail::has_operator<OperatorTypeClass, OperatorTypeClass::sm, Description>::value;
        static_assert(has_sm, "Description does not have CUDA architecture defined");

    public:
        using value_type                  = unsigned int;
        static constexpr value_type value = detail::get_t<OperatorTypeClass, OperatorTypeClass::sm, Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description>
    constexpr unsigned int sm_of<OperatorTypeClass, Description>::value;

    template<class OperatorTypeClass, class Description>
    inline constexpr unsigned int sm_of_v = sm_of<OperatorTypeClass, Description>::value;

    template<class OperatorTypeClass, class Description>
    struct block_dim_of {
    private:
        static constexpr bool has_block_dim = detail::has_operator<OperatorTypeClass, OperatorTypeClass::block_dim, Description>::value;
        static_assert(has_block_dim, "Description does not have block dimensions");

    public:
        using value_type                  = dim3;
        static constexpr value_type value = detail::get_t<OperatorTypeClass, OperatorTypeClass::block_dim, Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description>
    constexpr dim3 block_dim_of<OperatorTypeClass, Description>::value;

    template<class OperatorTypeClass, class Description>
    inline constexpr dim3 block_dim_of_v = block_dim_of<OperatorTypeClass, Description>::value;

    template<class OperatorTypeClass, class Description>
    struct grid_dim_of {
    private:
        static constexpr bool has_grid_dim = detail::has_operator<OperatorTypeClass, OperatorTypeClass::grid_dim, Description>::value;
        static_assert(has_grid_dim, "Description does not have grid dimensions");

    public:
        using value_type                  = dim3;
        static constexpr value_type value = detail::get_t<OperatorTypeClass, OperatorTypeClass::grid_dim, Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description>
    constexpr dim3 grid_dim_of<OperatorTypeClass, Description>::value;

    template<class OperatorTypeClass, class Description>
    inline constexpr dim3 grid_dim_of_v = grid_dim_of<OperatorTypeClass, Description>::value;

    template<class Description>
    struct is_dx_expression {
        using value_type                  = bool;
        static constexpr value_type value = detail::is_expression<Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr bool is_dx_expression<Description>::value;

    namespace detail {
#define COMMONDX_DEFINE_HAS_OP(name)                                                          \
        template<class OTC, class Desc, class = void>                                         \
        struct has_##name##_operator : COMMONDX_STL_NAMESPACE::false_type {};                 \
        template<class OTC, class Desc>                                                       \
        struct has_##name##_operator<OTC, Desc, decltype(void(OTC::name))> :                  \
            has_operator<OTC, OTC::name, Desc> {};
        COMMONDX_DEFINE_HAS_OP(thread)
        COMMONDX_DEFINE_HAS_OP(warp)
        COMMONDX_DEFINE_HAS_OP(block)
        COMMONDX_DEFINE_HAS_OP(cluster)
        COMMONDX_DEFINE_HAS_OP(device)
#undef COMMONDX_DEFINE_HAS_OP
    } // namespace detail

    template<class OperatorTypeClass, class Description>
    struct is_dx_execution_expression {
        static constexpr auto thread        = detail::has_thread_operator<OperatorTypeClass, Description>::value;
        static constexpr auto is_thread_op  = COMMONDX_STL_NAMESPACE::is_same<Description, Thread>::value;
        static constexpr auto warp          = detail::has_warp_operator<OperatorTypeClass, Description>::value;
        static constexpr auto is_warp_op    = COMMONDX_STL_NAMESPACE::is_same<Description, Warp>::value;
        static constexpr auto block         = detail::has_block_operator<OperatorTypeClass, Description>::value;
        static constexpr auto is_block_op   = COMMONDX_STL_NAMESPACE::is_same<Description, Block>::value;
        static constexpr auto cluster       = detail::has_cluster_operator<OperatorTypeClass, Description>::value;
        static constexpr auto is_cluster_op = COMMONDX_STL_NAMESPACE::is_same<Description, Cluster>::value;
        static constexpr auto device        = detail::has_device_operator<OperatorTypeClass, Description>::value;
        static constexpr auto is_device_op  = COMMONDX_STL_NAMESPACE::is_same<Description, Device>::value;

    public:
        using value_type                  = bool;
        static constexpr value_type value = is_dx_expression<Description>::value
                                            && ((thread && !is_thread_op) || (warp && !is_warp_op) || (block && !is_block_op)
                                                || (cluster && !is_cluster_op) || (device && !is_device_op));
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description>
    constexpr bool is_dx_execution_expression<OperatorTypeClass, Description>::value;

    template<class Description, template<class> class IsCompleteDescriptionCheck>
    struct is_complete_dx_expression {
        using value_type                  = bool;
        static constexpr value_type value = is_dx_expression<Description>::value && IsCompleteDescriptionCheck<Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description, template<class> class IsCompleteDescriptionCheck>
    constexpr bool is_complete_dx_expression<Description, IsCompleteDescriptionCheck>::value;

    template<class OperatorTypeClass, class Description, template<class> class IsCompleteDescriptionCheck>
    struct is_complete_dx_execution_expression {
        using value_type = bool;
        static constexpr value_type value =
            is_dx_execution_expression<OperatorTypeClass, Description>::value && is_complete_dx_expression<Description, IsCompleteDescriptionCheck>::value;
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description, template<class> class IsCompleteDescriptionCheck>
    constexpr bool is_complete_dx_execution_expression<OperatorTypeClass, Description, IsCompleteDescriptionCheck>::value;

    namespace detail {
        // Concatenates OperatorType to the description (faster than using decltype and adding operators)
        template<class OperatorType, class Description>
        struct concatenate_description;

        template<class OperatorType, template<class...> class Description, class... Operators>
        struct concatenate_description<OperatorType, Description<Operators...>> {
            using type = Description<OperatorType, Operators...>;
        };

        template<class OperatorType, class Description>
        using concatenate_description_t = typename concatenate_description<OperatorType, Description>::type;

        // Removes given OperatorType from a Description
        template<class Description, class OperatorTypeClass, OperatorTypeClass OperatorType>
        struct filter {
            using type = void;
        };

        template<class Description, class OperatorTypeClass, OperatorTypeClass OperatorType>
        using filter_t = typename filter<Description, OperatorTypeClass, OperatorType>::type;

        template<template<class...> class Description, class OperatorTypeClass, OperatorTypeClass OperatorType>
        struct filter<Description<>, OperatorTypeClass, OperatorType> {
            using type = Description<>;
        };

        template<template<class...> class Description, class OperatorTypeClass, OperatorTypeClass OperatorType, class Head, class... Tail>
        struct filter<Description<Head, Tail...>,  OperatorTypeClass, OperatorType> {
            using type =
                typename COMMONDX_STL_NAMESPACE::conditional<is_operator<OperatorTypeClass, OperatorType, Head>::value,
                                          filter_t<Description<Tail...>, OperatorTypeClass, OperatorType>,
                                          concatenate_description_t<Head, typename filter<Description<Tail...>, OperatorTypeClass, OperatorType>::type> //
                                          >::type;
        };

        template<template<class...> class DescriptionType, class Description>
        struct convert_to_dx_description {
            using type = void;
        };

        template<template<class...> class DescriptionType, template<class...> class Description, class... Types>
        struct convert_to_dx_description<DescriptionType, Description<Types...>> {
            using type = DescriptionType<Types...>;
        };

        // Guarded filter_t wrappers — pass-through when enum value absent
#define COMMONDX_DEFINE_FILTER_OP(name)                                                                            \
        template<class Desc, class OTC, class = void>                                                             \
        struct filter_##name { using type = Desc; };                                                              \
        template<class Desc, class OTC>                                                                           \
        struct filter_##name<Desc, OTC, decltype(void(OTC::name))> { using type = filter_t<Desc, OTC, OTC::name>; };
        COMMONDX_DEFINE_FILTER_OP(thread)
        COMMONDX_DEFINE_FILTER_OP(warp)
        COMMONDX_DEFINE_FILTER_OP(block)
        COMMONDX_DEFINE_FILTER_OP(cluster)
        COMMONDX_DEFINE_FILTER_OP(device)
#undef COMMONDX_DEFINE_FILTER_OP

        template<class Desc, class OTC> using filter_thread_t  = typename filter_thread<Desc, OTC>::type;
        template<class Desc, class OTC> using filter_warp_t    = typename filter_warp<Desc, OTC>::type;
        template<class Desc, class OTC> using filter_block_t   = typename filter_block<Desc, OTC>::type;
        template<class Desc, class OTC> using filter_cluster_t = typename filter_cluster<Desc, OTC>::type;
        template<class Desc, class OTC> using filter_device_t  = typename filter_device<Desc, OTC>::type;
    } // namespace detail

    template<template<class...> class DescriptionType, class Description, class OperatorTypeClass>
    struct extract_dx_description {
    private:
        using description_removed_cvref = COMMONDX_STL_NAMESPACE::remove_cv_t<COMMONDX_STL_NAMESPACE::remove_reference_t<Description>>;
        // Converts execution description to simple description; filters out Thread, Block, Cluster, Warp, Device operators
        using dx_description_type = typename detail::convert_to_dx_description<DescriptionType, description_removed_cvref>::type;

    public:
        static_assert(is_dx_expression<description_removed_cvref >::value, "Description type is not a Dx description");
        using type = typename COMMONDX_STL_NAMESPACE::conditional<
            detail::is_operator_expression<description_removed_cvref >::value,
            description_removed_cvref , // For single operator or if Description just return Description
            detail::filter_device_t<
                detail::filter_cluster_t<
                    detail::filter_block_t<
                        detail::filter_warp_t<
                            detail::filter_thread_t<dx_description_type, OperatorTypeClass>,
                        OperatorTypeClass>,
                    OperatorTypeClass>,
                OperatorTypeClass>,
            OperatorTypeClass> //
            >::type;
    };

    template<template<class...> class DescriptionType, class Description, class OperatorTypeClass>
    using extract_dx_description_t = typename extract_dx_description<DescriptionType, Description, OperatorTypeClass>::type;
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TRAITS_DX_TRAITS_HPP
