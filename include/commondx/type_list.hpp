/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMONDX_TYPE_LIST_HPP
#define COMMONDX_TYPE_LIST_HPP

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {

    template<int Index, class T>
    struct type_list_element;

    template<class... Elements>
    struct type_list {};

    template<int Index, class Head, class... Tail>
    struct type_list_element<Index, type_list<Head, Tail...>>: type_list_element<Index - 1, type_list<Tail...>> {};

    template<class Head, class... Tail>
    struct type_list_element<0, type_list<Head, Tail...>> {
        using type = Head;
    };

} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TYPE_LIST_HPP
