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

#ifndef COMMONDX_TENSOR_HPP
#define COMMONDX_TENSOR_HPP

#include "commondx/detail/macros.hpp"
#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"

#include "commondx/detail/cute/tensor.hpp"

#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        struct gmem_tag {};
        struct smem_tag {};
        struct rmem_tag {};

        template<class PointerTag, class Layout>
        struct pointer_layout {
            static_assert(cute::is_layout<Layout>::value, "Layout must be a CuTe layout type");

            using pointer_tag_type = PointerTag;
            using layout_type = Layout;

            PointerTag pointer_tag;
            Layout layout;

        private:
            template<class T, class K>
            static constexpr auto make_ptr_impl(K* ptr) {
                if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<PointerTag, gmem_tag>) {
                    return cute::make_gmem_ptr<T>(ptr);
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<PointerTag, smem_tag>) {
                    return cute::make_smem_ptr<T>(ptr);
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<PointerTag, rmem_tag>) {
                    return cute::make_rmem_ptr<T>(ptr);
                }
            }

        public:
            constexpr pointer_layout(PointerTag pt, Layout l) : pointer_tag(pt), layout(l) {}

            template<class T>
            static constexpr auto make_ptr(T* ptr) {
                return make_ptr_impl<T, T>(ptr);
            }

            template<class T>
            static constexpr auto make_ptr(void* ptr) {
                return make_ptr_impl<T, void>(ptr);
            }

            template<class T>
            static constexpr auto make_ptr(const void* ptr) {
                return make_ptr_impl<T, const void>(ptr);
            }
        };
    } // namespace detail

    template<class Engine, class Layout>
    using tensor = cute::Tensor<Engine, Layout>;

    template<class T>
    struct is_layout : cute::is_layout<T> {};

    template<class Tag, class Layout>
    struct is_layout<detail::pointer_layout<Tag, Layout>> : cute::true_type {};

    template<class T>
    inline constexpr bool is_layout_v = is_layout<T>::value;

    template<class Iterator, class... Args>
    COMMONDX_HOST_DEVICE constexpr auto make_tensor(const Iterator& iter, Args const&... args) {
        return cute::make_tensor(iter, args...);
    }

    template<class T, class PointerTag, class Layout>
    COMMONDX_HOST_DEVICE constexpr auto make_tensor(const cute::gmem_ptr<T>& iter, const detail::pointer_layout<PointerTag, Layout>& pl) {
        static_assert(COMMONDX_STL_NAMESPACE::is_same_v<PointerTag, detail::gmem_tag>, "Global memory pointer must remain a global memory pointer");
        return cute::make_tensor(iter, pl.layout);
    }

    template<class T, class PointerTag, class Layout>
    COMMONDX_HOST_DEVICE constexpr auto make_tensor(const cute::smem_ptr<T>& iter, const detail::pointer_layout<PointerTag, Layout>& pl) {
        static_assert(COMMONDX_STL_NAMESPACE::is_same_v<PointerTag, detail::smem_tag>, "Shared memory pointer must remain a shared memory pointer");
        return cute::make_tensor(iter, pl.layout);
    }

    template<class T, class PointerTag, class Layout>
    COMMONDX_HOST_DEVICE constexpr auto make_tensor(const cute::rmem_ptr<T>& iter, const detail::pointer_layout<PointerTag, Layout>& pl) {
        static_assert(COMMONDX_STL_NAMESPACE::is_same_v<PointerTag, detail::rmem_tag>, "Register file pointer must remain a register file pointer");
        return cute::make_tensor(iter, pl.layout);
    }

    template<class T, class PointerTag, class Layout>
    COMMONDX_HOST_DEVICE constexpr auto make_tensor(T* ptr, const detail::pointer_layout<PointerTag, Layout>& pl) {
        using pl_t = typename detail::pointer_layout<PointerTag, Layout>;
        return cute::make_tensor(pl_t::make_ptr(ptr), pl.layout);
    }

    template<class T, class PointerTag, class Layout>
    COMMONDX_HOST_DEVICE constexpr auto make_tensor(void* ptr, const detail::pointer_layout<PointerTag, Layout>& pl) {
        using pl_t = typename detail::pointer_layout<PointerTag, Layout>;
        return cute::make_tensor(pl_t::template make_ptr<T>(ptr), pl.layout);
    }

    template<class T, class PointerTag, class Layout>
    COMMONDX_HOST_DEVICE constexpr auto make_tensor(const void*                                        ptr,
                                                   const detail::pointer_layout<PointerTag, Layout>& pl) {
        using pl_t = typename detail::pointer_layout<PointerTag, Layout>;
        return cute::make_tensor(pl_t::template make_ptr<T>(ptr), pl.layout);
    }

    template<class Layout>
    COMMONDX_HOST_DEVICE constexpr auto size(Layout const& layout) {
        if constexpr(is_layout_v<Layout> && not cute::is_layout<Layout>::value) {
            return cute::size(layout.layout);
        } else {
            return cute::size(layout);
        }

        CUTE_GCC_UNREACHABLE;
    }

    template<class Layout>
    COMMONDX_HOST_DEVICE constexpr auto cosize(Layout const& layout) {
        if constexpr(is_layout_v<Layout> && not cute::is_layout<Layout>::value) {
            return cute::cosize(layout.layout);
        } else {
            return cute::cosize(layout);
        }
        CUTE_GCC_UNREACHABLE;
    }

} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TENSOR_HPP
