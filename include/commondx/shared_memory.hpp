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

#ifndef COMMONDX_SHARED_MEMORY_HPP
#define COMMONDX_SHARED_MEMORY_HPP

#include "commondx/detail/macros.hpp"
#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/utility.hpp"
#include "commondx/tensor.hpp" // includes CuTe/CUTLASS, cutlass::round_up
#include "commondx/types.hpp"

#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        // Keep below functions external to allow for constexpr device
        // memory computation without code repetition
        template<class Layout>
        COMMONDX_HOST_DEVICE constexpr
        COMMONDX_STL_NAMESPACE::enable_if_t<commondx::is_layout_v<Layout>, unsigned>
        add_aligned_shared_memory_extent(unsigned current, unsigned alignment, unsigned elem_size, const Layout& layout) {
            return (cutlass::round_up(current, alignment) + commondx::cosize(layout) * elem_size);
        }

        COMMONDX_HOST_DEVICE constexpr
        unsigned add_aligned_shared_memory_extent(unsigned current, unsigned alignment, unsigned matrix_size_bytes) {
            return (cutlass::round_up(current, alignment) + matrix_size_bytes);
        }

        COMMONDX_HOST_DEVICE constexpr
        unsigned add_aligned_shared_memory_extent(unsigned current, unsigned alignment, unsigned elem_size, unsigned num_elements) {
            return (cutlass::round_up(current, alignment) + num_elements * elem_size);
        }

        struct shared_storage_calculator {
            private:
            unsigned current = 0;

            public:
            template<class ... Args>
            COMMONDX_HOST_DEVICE shared_storage_calculator&
            add(Args ... args) {
                current = add_aligned_shared_memory_extent(current, args...);
                return *this;
            }

            COMMONDX_HOST_DEVICE unsigned get() {
                return current;
            }
        };
    } // namespace detail

    COMMONDX_HOST_DEVICE detail::shared_storage_calculator make_shared_storage_calculator() {
        return detail::shared_storage_calculator{};
    }

    namespace detail {
        COMMONDX_HOST_DEVICE
        uintptr_t ptr_round_up(uintptr_t a, uintptr_t b) {
            return ((a + b - 1) / b) * b;
        }

        // Shared memory slicing helpers
        template<class T>
        struct pop_first_static;

        template<class T, class ... Ts>
        struct pop_first_static<cute::tuple<T, Ts...>> {
            using type = cute::tuple<Ts...>;
        };

        template<class T>
        using pop_first_static_t = typename pop_first_static<T>::type;

        template<class PointerTypeTuple, class Tuple, auto ... I>
        COMMONDX_HOST_DEVICE
        auto offset_pointers(byte* smem, Tuple const& offsets, cute::index_sequence<I...>) {
            return cute::make_tuple(reinterpret_cast<cute::tuple_element_t<I, PointerTypeTuple>*>(smem + cute::get<I>(offsets))...);
        }

        template<class PointerTypeTuple, class MD> // Memory Descriptor is: tuple<cosize, alignment>
        COMMONDX_HOST_DEVICE
        constexpr auto align_offsets(uintptr_t smem, MD const& md) {
            static_assert(cute::tuple_size<PointerTypeTuple>::value == 1);
            using current_ptr_t = cute::tuple_element_t<0, PointerTypeTuple>;

            auto last_int_ptr = ptr_round_up(smem, cute::get<1>(md));

            return cute::make_tuple(last_int_ptr);
        }

        template<class PointerTypeTuple, class MD, class ... MDS> // Memory Descriptor is: tuple<cosize, alignment>
        COMMONDX_HOST_DEVICE
        constexpr auto align_offsets(uintptr_t smem, MD const& md, MDS const& ... mds) {
            using current_ptr_t = cute::tuple_element_t<0, PointerTypeTuple>;
            constexpr auto elem_size = sizeof(current_ptr_t);

            const auto current_int_ptr = ptr_round_up(smem, cute::get<1>(md));
            using next_type_tuple = pop_first_static_t<PointerTypeTuple>;

            // next = current + tensor_cosize * elem_size
            const auto next_int_ptr = current_int_ptr + (cute::get<0>(md) * elem_size);
            return cute::prepend(align_offsets<next_type_tuple>(next_int_ptr, mds...), current_int_ptr);
        }
    } // namespace detail

    namespace shared_memory {
        namespace detail {
            //
            // Internal helper that, given a base pointer and current offset,
            // aligns the offset, returns a pointer of type T*, then updates offset.
            //
            // Requirements:
            //   - Alignment for a slice (pointer) must be multiple of the alignment of
            //     the corresponding type,
            //   - Each slice must contain at least one element, except for the last one,
            //     which may contain zero elements (see the shorthand slice API for details),
            //
            template <class T>
            COMMONDX_HOST_DEVICE
            T* slice_pointer(byte* base, uintptr_t& offset, const unsigned alignment, const unsigned size_in_elements) {
                assert(alignment % alignof(T) == 0);
                assert(alignment >= alignof(T));
                offset = commondx::detail::ptr_round_up(offset, alignment);
                T* ptr = reinterpret_cast<T*>(base + offset);
                offset += size_in_elements * sizeof(T);
                return ptr;
            }

            //
            // Internal helper for building a tuple of pointers from alignment/size pairs
            //
            template <class T, class... Ts, class... Args>
            COMMONDX_HOST_DEVICE
            auto slice_into_pointers_impl(byte* base, uintptr_t& offset,
                                          const unsigned alignment, const unsigned size_in_elements,
                                          const Args... args)
            {
                T* head = slice_pointer<T>(base, offset, alignment, size_in_elements);
                if constexpr (sizeof...(Ts) == 0) {
                    return cute::make_tuple(head);
                } else {
                    auto tail = slice_into_pointers_impl<Ts...>(base, offset, args...);
                    return cute::tuple_cat(cute::make_tuple(head), tail);
                }
            }
        } // namespace detail

        //
        // slice_into_pointers:
        //   Example usage (full form):
        //      auto [a_ptr, b_ptr] = slice_into_pointers<A,B>(
        //          smem,
        //          align_a, size_a,
        //          align_b, size_b        // <- size_b required
        //      );
        //
        //   New shorthand: size for the LAST pointer may be omitted
        //      auto [a_ptr, b_ptr] = slice_into_pointers<A,B>(
        //          smem,
        //          align_a, size_a,
        //          align_b                // <- size_b omitted
        //      );
        //

        // --------------------------------------------------------------------
        // 1.  Full form - expects (alignment, size) for every pointer
        // --------------------------------------------------------------------
        template <class... Ts, class... Args>
        COMMONDX_HOST_DEVICE
        auto
        slice_into_pointers(void* smem, const Args... args)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<
                   (sizeof...(Args) == 2 * sizeof...(Ts)),
                   cute::tuple<Ts*...>>
        {
            static_assert(sizeof...(Ts) * 2 == sizeof...(Args),
                          "slice_into_pointers<Ts...> requires (alignment,size) pairs for every pointer");
            uintptr_t offset = 0;
            return detail::slice_into_pointers_impl<Ts...>(
                static_cast<byte*>(smem), offset, args...);
        }

        // --------------------------------------------------------------------
        // 2.  Shorthand - last size omitted, i.e. (2*|Ts| − 1) parameters
        // --------------------------------------------------------------------
        template <class... Ts, class... Args>
        COMMONDX_HOST_DEVICE
        auto
        slice_into_pointers(void* smem, const Args... args)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<
                   (sizeof...(Args) + 1 == 2 * sizeof...(Ts)),
                   cute::tuple<Ts*...>>
        {
            // Provide a dummy size (0 elements) for the last pointer so that
            // alignment is honoured while no additional space is consumed.
            uintptr_t offset = 0;
            return detail::slice_into_pointers_impl<Ts...>(
                static_cast<byte*>(smem), offset, args..., 0u);
        }

        namespace detail {
            //
            // Internal helper for building a tuple of "tensors" from alignment/layout
            //
            // If "layout" is a layout, we do:
            //      offset = round_up(offset, alignment)
            //      T* ptr = base + offset
            //      offset += cosize(layout) * sizeof(T)
            //      return make_tensor(ptr, layout)
            //
            // We rely on commondx::cosize(layout).
            //
            // Requirements:
            //   - Alignment for a slice (tensor) must be multiple of the alignment of
            //     the corresponding type,
            //   - Each slice must contain at least one element, except for the last one,
            //     which may contain zero elements (see the shorthand slice API for details),
            //
            template <class T, class Layout, __CUTE_REQUIRES(commondx::is_layout_v<Layout>)>
            COMMONDX_HOST_DEVICE
            auto slice_tensor(byte* base, uintptr_t& offset, const unsigned alignment, const Layout& layout) {
                assert(alignment % alignof(T) == 0);
                assert(alignment >= alignof(T));
                offset = commondx::detail::ptr_round_up(offset, alignment);

                // Grab raw pointer, then wrap it with cute::make_smem_ptr
                auto* raw_ptr = reinterpret_cast<T*>(base + offset);

                // Increase the offset
                offset += commondx::cosize(layout) * sizeof(T);

                // Now create the tensor using the wrapped smem pointer
                return commondx::make_tensor(cute::make_smem_ptr(raw_ptr), layout);
            }

            //
            // Implementation that extracts each (alignment, Layout) pair for Ts...
            //
            template <class T, class... Ts, class Layout, class... Args>
            COMMONDX_HOST_DEVICE
            auto slice_into_tensors_impl(byte* base, uintptr_t& offset,
                                         const unsigned alignment, const Layout& layout,
                                         const Args... args)
            {
                auto head = slice_tensor<T>(base, offset, alignment, layout);
                if constexpr (sizeof...(Ts) == 0) {
                    return cute::make_tuple(head);
                } else {
                    auto tail = slice_into_tensors_impl<Ts...>(base, offset, args...);
                    return cute::tuple_cat(cute::make_tuple(head), tail);
                }
            }

            template<class T>
            COMMONDX_HOST_DEVICE
            constexpr bool verify_layouts_pointer_tag_impl() {
                if constexpr(commondx::is_layout_v<T>) {
                    if constexpr(cute::is_layout<T>::value) {
                        // Return true for CuTe Layout type
                        return true;
                    } else {
                        return COMMONDX_STL_NAMESPACE::is_same_v<typename T::pointer_tag_type, commondx::detail::smem_tag>;
                    }
                } else {
                    // Return true for types that are not layout
                    return true;
                }
                CUTE_GCC_UNREACHABLE;
            }

            template<class T>
            COMMONDX_HOST_DEVICE
            constexpr bool verify_layouts_pointer_tag() {
                return verify_layouts_pointer_tag_impl<T>();
            }

            template<class T, class... Ts>
            COMMONDX_HOST_DEVICE
            constexpr auto verify_layouts_pointer_tag() -> COMMONDX_STL_NAMESPACE::enable_if_t<sizeof...(Ts) != 0, bool> {
                return verify_layouts_pointer_tag_impl<T>() && verify_layouts_pointer_tag<Ts...>();
            }
        }

        //
        // slice_into_tensors:
        //   Example usage:
        //      auto [a_tensor, b_tensor] = slice_into_tensors<AValueType, BValueType>(
        //          smem,
        //          alignment_a, some_layout_a,
        //          alignment_b, some_layout_b
        //      );
        //
        // Each pair is (alignment, Layout).
        //
        template <class... Ts, class... Args>
        COMMONDX_HOST_DEVICE
        auto slice_into_tensors(void* smem, const Args... args) {
            static_assert(sizeof...(Ts) * 2 == sizeof...(Args),
                        "slice_into_tensors<Ts...> requires 2 * sizeof...(Ts) args: (alignment, Layout) pairs");
            static_assert(detail::verify_layouts_pointer_tag<Args...>(), "slice_into_tensors: if layout is tagged it must be for shared memory");
            uintptr_t offset = 0;
            return detail::slice_into_tensors_impl<Ts...>(static_cast<byte*>(smem), offset, args...);
        }



        namespace detail {
            // Overload with the base case of zero additional parameters
            COMMONDX_HOST_DEVICE
            auto slice_impl(byte*, uintptr_t&) {
                return cute::make_tuple();
            }

            // Overload that processes one [Type, alignment, nextParam], where
            // nextParam can be an integral size or a layout.
            template <class T, class... RestTypes, class NextParam, class... RestArgs>
            COMMONDX_HOST_DEVICE
            auto slice_impl(byte* base, uintptr_t& offset,
                            const unsigned alignment, const NextParam next_param,
                            const RestArgs... rest_args)
            {
                if constexpr (commondx::is_layout_v<NextParam>) {
                    auto head = slice_tensor<T>(base, offset, alignment, next_param);
                    if constexpr (sizeof...(RestTypes) == 0) {
                        return cute::make_tuple(head);
                    } else if constexpr (sizeof...(RestTypes) == 1) {
                        auto tail = slice_impl<RestTypes...>(base, offset, rest_args..., 0u);
                        return cute::tuple_cat(cute::make_tuple(head), tail);
                    } else {
                        auto tail = slice_impl<RestTypes...>(base, offset, rest_args...);
                        return cute::tuple_cat(cute::make_tuple(head), tail);
                    }
                } else {
                    // treat nextParam as size in elements
                    static_assert(cute::is_integral<NextParam>::value, "Slicing argument can be either a valid layout or integral signifying size in elements");
                    T* head = slice_pointer<T>(base, offset, alignment, static_cast<unsigned>(next_param));
                    if constexpr (sizeof...(RestTypes) == 0) {
                        return cute::make_tuple(head);
                    } else if constexpr (sizeof...(RestTypes) == 1) {
                        auto tail = slice_impl<RestTypes...>(base, offset, rest_args..., 0u);
                        return cute::tuple_cat(cute::make_tuple(head), tail);
                    } else {
                        auto tail = slice_impl<RestTypes...>(base, offset, rest_args...);
                        return cute::tuple_cat(cute::make_tuple(head), tail);
                    }
                }
            }
        }

        //
        // slice:
        //   Mix of pointers and tensors:
        //   - If the second arg is an integral, treat it as 'size_in_elements'
        //   - If the second arg is a layout, treat it as a layout
        //
        // Requirements:
        //   - Alignment for a slice (tensor or pointer) must be multiple of the alignment of
        //     the corresponding type,
        //   - Each slice must contain at least one element, except for the last one,
        //     which may contain zero elements (see the shorthand slice API for details),
        //
        // Example usage (full form):
        //   auto [a_tensor, b_tensor, c_ptr, d_ptr]
        //     = slice<AValueType, BValueType, CValueType, DValueType>(
        //         smem,
        //         alignment_of_a, layout_a,
        //         alignment_of_b, layout_b,
        //         alignment_of_c, size_in_elems,
        //         alignment_of_d, cosize(layout_c)
        //       );
        //
        // Example usage (shorthand form; size/layout for the LAST pointer may be omitted):
        //   auto [a_tensor, b_tensor, c_ptr, d_ptr]
        //     = slice<AValueType, BValueType, CValueType, DValueType>(
        //         smem,
        //         alignment_of_a, layout_a,
        //         alignment_of_b, layout_b,
        //         alignment_of_c, size_in_elems,
        //         alignment_of_d
        //       );
        //
        // Each chunk is (alignment, either layout or size).
        // We'll do a single pass over the arguments in sets of two.
        //
        // The size/layout argument can be omitted for the LAST pointer.
        // If it is omitted the corresponding return value is a pointer.

        // --------------------------------------------------------------------
        // slice : full form - expects (alignment , layout|size) for every T
        // --------------------------------------------------------------------
        template <class... Ts, class... Args>
        COMMONDX_HOST_DEVICE
        auto
        slice(void* smem, const Args... args)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<
                   (sizeof...(Args) == 2 * sizeof...(Ts)),               // full form
                   decltype(detail::slice_impl<Ts...>(                   // deduce return type
                              static_cast<byte*>(nullptr),
                              COMMONDX_STL_NAMESPACE::declval<uintptr_t&>(),
                              COMMONDX_STL_NAMESPACE::declval<Args>()...))>
        {
            static_assert(detail::verify_layouts_pointer_tag<Args...>(),
                          "slice: if layout is tagged it must be for shared memory");
            uintptr_t offset = 0;
            return detail::slice_impl<Ts...>(static_cast<byte*>(smem), offset, args...);
        }

        // --------------------------------------------------------------------
        // slice : shorthand - last pointer SIZE omitted (one arg less)
        // --------------------------------------------------------------------
        template <class... Ts, class... Args>
        COMMONDX_HOST_DEVICE
        auto
        slice(void* smem, const Args... args)
            -> COMMONDX_STL_NAMESPACE::enable_if_t<
                   (sizeof...(Args) + 1 == 2 * sizeof...(Ts)),           // size omitted
                   decltype(detail::slice_impl<Ts...>(                  // deduce return type
                              static_cast<byte*>(nullptr),
                              COMMONDX_STL_NAMESPACE::declval<uintptr_t&>(),
                              COMMONDX_STL_NAMESPACE::declval<Args>()..., unsigned{}))>
        {
            static_assert(detail::verify_layouts_pointer_tag<Args...>(),
                          "slice: if layout is tagged it must be for shared memory");
            uintptr_t offset = 0;
            // Forward a dummy size (0) for the final pointer so alignment is
            // honoured while no extra memory is consumed.
            return detail::slice_impl<Ts...>(static_cast<byte*>(smem), offset, args..., 0u);
        }

    } // namespace shared_memory
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_SHARED_MEMORY_HPP
