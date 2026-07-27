/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NPPDX_TILE_STORAGE_HPP
#define NPPDX_TILE_STORAGE_HPP

// Leaf shared-memory types and value-/type-based byte/alignment helpers.
//
// This header intentionally has no dependency on nppdx/traits.hpp so that
// it can be pulled in by storage_config.hpp (and therefore by the operator
// trait headers that storage_config.hpp serves) without forming an include
// cycle. The trait-aware byte/alignment helpers that look up a Description's
// inputs/outputs/temp tuples live in nppdx/shared_memory.hpp on top of this
// header.

#include "nppdx/detail/config.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/types.hpp"

#include NPPDX_STD_INCLUDE_CSTDLIB
#include NPPDX_STD_INCLUDE_ALGORITHM
#include NPPDX_STD_INCLUDE_NEW
#include NPPDX_STD_INCLUDE_TUPLE
#include NPPDX_STD_INCLUDE_TYPE_TRAITS
#include NPPDX_STD_INCLUDE_UTILITY

namespace nppdx {
    namespace shared_memory {

        namespace detail {

            // ========================================================================
            // Value-based computation functions
            // ========================================================================

            NPPDX_DECL_CFHD unsigned int round_up(unsigned int value, unsigned int alignment) {
                return ((value + alignment - 1) / alignment) * alignment;
            }

            // Variadic needed for pack expansion in type-based helpers
            template<typename... Values>
            NPPDX_DECL_CFHD unsigned int max_alignment(Values... alignments) {
                if constexpr (sizeof...(alignments) == 0) {
                    return 1;
                } else {
                    unsigned int result = 1;
                    ((result = (static_cast<unsigned int>(alignments) > result) ? static_cast<unsigned int>(alignments)
                                                                                : result),
                     ...);
                    return result;
                }
            }

            // Variadic needed for pack expansion in type-based helpers
            template<typename... Sizes>
            NPPDX_DECL_CFHD unsigned int compute_aligned_storage(unsigned int alignment, Sizes... item_sizes) {
                unsigned int offset = 0;
                ((offset = round_up(offset, alignment), offset += static_cast<unsigned int>(item_sizes)), ...);
                return offset;
            }

        } // namespace detail

        NPPDX_DECL_CFHD unsigned int compute_channels_storage(unsigned int width, unsigned int height,
                                                              unsigned int num_channels, unsigned int type_size,
                                                              unsigned int alignment) {
            if (num_channels == 0) {
                return 0;
            }
            unsigned int channel_size = width * height * type_size;
            return (num_channels - 1) * detail::round_up(channel_size, alignment) + channel_size;
        }

        template<typename T, unsigned int Width, unsigned int Height,
                 unsigned int Alignment = (NPPDX_STD::max)(4u, static_cast<unsigned int>(alignof(T)))>
        struct ChannelSlice {
            static_assert(Alignment % alignof(T) == 0, "Alignment must be a multiple of alignof(T)");

            using value_type = T;

            value_type* data;

            static constexpr unsigned int width      = Width;
            static constexpr unsigned int height     = Height;
            static constexpr unsigned int alignment  = Alignment;
            static constexpr unsigned int size       = Width * Height;
            static constexpr unsigned int size_bytes = size * sizeof(T);

            NPPDX_DECL_FD ChannelSlice(): data(nullptr) {}

            NPPDX_DECL_FD ChannelSlice(T* ptr): data(ptr) {}

            static NPPDX_DECL_FD ChannelSlice from_memory(unsigned char* ptr) {
                static_assert(NPPDX_STD::is_trivially_constructible_v<T>, "value_type must be trivially constructible");
                auto channel_data = new (ptr) T[size];
                return ChannelSlice(channel_data);
            }

            NPPDX_DECL_FD value_type& operator()(int2 pos) { return data[pos.y * Width + pos.x]; }

            NPPDX_DECL_FD const value_type& operator()(int2 pos) const { return data[pos.y * Width + pos.x]; }

            NPPDX_DECL_FD value_type& operator()(int y, int x) { return data[y * Width + x]; }

            NPPDX_DECL_FD const value_type& operator()(int y, int x) const { return data[y * Width + x]; }

            NPPDX_DECL_FD value_type& operator[](int idx) { return data[idx]; }

            NPPDX_DECL_FD const value_type& operator[](int idx) const { return data[idx]; }

            NPPDX_DECL_FD value_type*       ptr() { return data; }
            NPPDX_DECL_FD const value_type* ptr() const { return data; }
        };

        template<typename ChannelSliceType_, unsigned int NumChannels>
        struct TileStorage {
            using ChannelSliceType = ChannelSliceType_;

            ChannelSliceType channels[NumChannels];

            static constexpr unsigned int num_channels = NumChannels;

            NPPDX_DECL_FD TileStorage() {}

            NPPDX_DECL_FD TileStorage(const ChannelSliceType (&slices)[NumChannels]) {
                for (unsigned int i = 0; i < NumChannels; ++i) {
                    channels[i] = slices[i];
                }
            }

            template<typename... Slices,
                     typename = typename NPPDX_STD::enable_if_t<
                         (sizeof...(Slices) == NumChannels) && (NPPDX_STD::is_same_v<Slices, ChannelSliceType> && ...)>>
            NPPDX_DECL_FD TileStorage(Slices... slices) {
                ChannelSliceType temp[] = {slices...};
                for (unsigned int i = 0; i < NumChannels; ++i) {
                    channels[i] = temp[i];
                }
            }

            static NPPDX_DECL_FD TileStorage from_memory(unsigned char* ptr) {
                using value_type = typename ChannelSliceType::value_type;
                static_assert(NPPDX_STD::is_trivially_constructible_v<value_type>,
                              "value_type must be trivially constructible");

                TileStorage  result;
                unsigned int offset = 0;
                for (unsigned int i = 0; i < NumChannels; ++i) {
                    offset              = detail::round_up(offset, ChannelSliceType::alignment);
                    auto channel_values = new (ptr + offset) value_type[ChannelSliceType::size];
                    result.channels[i]  = ChannelSliceType(channel_values);
                    offset += ChannelSliceType::size_bytes;
                }
                return result;
            }

            template<unsigned int Channel>
            NPPDX_DECL_FD ChannelSliceType& channel() {
                static_assert(Channel < NumChannels, "Channel index out of bounds");
                return channels[Channel];
            }

            template<unsigned int Channel>
            NPPDX_DECL_FD const ChannelSliceType& channel() const {
                static_assert(Channel < NumChannels, "Channel index out of bounds");
                return channels[Channel];
            }

            NPPDX_DECL_FD ChannelSliceType& channel(unsigned int idx) { return channels[idx]; }

            NPPDX_DECL_FD const ChannelSliceType& channel(unsigned int idx) const { return channels[idx]; }

            NPPDX_DECL_FD void set_channel(unsigned int idx, const ChannelSliceType& slice) { channels[idx] = slice; }

            static NPPDX_DECL_FD constexpr unsigned int required_bytes() {
                return compute_channels_storage(ChannelSliceType::width, ChannelSliceType::height, NumChannels,
                                                sizeof(typename ChannelSliceType::value_type),
                                                ChannelSliceType::alignment);
            }

            NPPDX_DECL_FD auto& operator()(int2 pos, int ch_idx) { return channel(uint(ch_idx))(pos); }

            NPPDX_DECL_FD const auto& operator()(int2 pos, int ch_idx) const { return channel(uint(ch_idx))(pos); }
        };

        namespace detail {

            // ========================================================================
            // Type-based helpers (operate on TileStorage tuples, no Description traits)
            // ========================================================================

            template<typename TileStorageType>
            NPPDX_DECL_FD TileStorageType slice_tile_storage(unsigned char* base, unsigned int& offset) {
                constexpr unsigned int alignment = TileStorageType::ChannelSliceType::alignment;
                offset                           = round_up(offset, alignment);
                unsigned char* storage_ptr       = base + offset;
                offset += TileStorageType::required_bytes();
                return TileStorageType::from_memory(storage_ptr);
            }

            // Helper to compute alignment from tuple using index_sequence
            template<typename TupleType, NPPDX_STD::size_t... Is>
            NPPDX_DECL_CFHD unsigned int tuple_alignment_impl(NPPDX_STD::index_sequence<Is...>) {
                if constexpr (sizeof...(Is) == 0) {
                    return 1;
                } else {
                    return max_alignment(static_cast<unsigned int>(
                        NPPDX_STD::tuple_element_t<Is, TupleType>::ChannelSliceType::alignment)...);
                }
            }

            template<typename TupleType>
            NPPDX_DECL_CFHD unsigned int tuple_alignment() {
                return tuple_alignment_impl<TupleType>(
                    NPPDX_STD::make_index_sequence<NPPDX_STD::tuple_size_v<TupleType>> {});
            }

            // Helper to compute storage bytes from tuple using index_sequence
            template<typename TupleType, NPPDX_STD::size_t... Is>
            NPPDX_DECL_CFHD unsigned int tuple_storage_bytes_impl(NPPDX_STD::index_sequence<Is...>) {
                if constexpr (sizeof...(Is) == 0) {
                    return 0;
                } else {
                    return compute_aligned_storage(tuple_alignment<TupleType>(),
                                                   NPPDX_STD::tuple_element_t<Is, TupleType>::required_bytes()...);
                }
            }

            template<typename TupleType>
            NPPDX_DECL_CFHD unsigned int tuple_storage_bytes() {
                return tuple_storage_bytes_impl<TupleType>(
                    NPPDX_STD::make_index_sequence<NPPDX_STD::tuple_size_v<TupleType>> {});
            }

        } // namespace detail

        // Obtain a TileStorage tuple
        template<typename... TileStorageTypes>
        NPPDX_DECL_FD NPPDX_STD::tuple<TileStorageTypes...> slice_into_tile_storage(unsigned char* smem) {
            unsigned int offset = 0;
            return NPPDX_STD::tuple<TileStorageTypes...> {
                detail::slice_tile_storage<TileStorageTypes>(smem, offset)...};
        }

        template<typename... TileStorageTypes>
        NPPDX_DECL_CFHD unsigned int compute_total_tile_storage() {
            return detail::tuple_storage_bytes<NPPDX_STD::tuple<TileStorageTypes...>>();
        }

        // Must be called by all threads in a block: the copy ends with a block-wide barrier.
#if defined(__CUDA_ARCH__) || (defined(__CUDACC__) && !defined(NPPDX_CLANG_CUDA_COMPAT_HOST))
        template<typename TileStorage>
        NPPDX_DECL_FD void copy_tile_storage(const TileStorage& src, TileStorage& dst) {
            constexpr unsigned int elements_per_channel = TileStorage::ChannelSliceType::size;
            constexpr unsigned int total_elements       = TileStorage::num_channels * elements_per_channel;
            const unsigned int     threads              = blockDim.x * blockDim.y * blockDim.z;
            const unsigned int tid = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

            for (unsigned int linear_idx = tid; linear_idx < total_elements; linear_idx += threads) {
                const unsigned int channel_idx        = linear_idx / elements_per_channel;
                const unsigned int element_idx        = linear_idx % elements_per_channel;
                dst.channel(channel_idx)[element_idx] = src.channel(channel_idx)[element_idx];
            }

            __syncthreads();
        }
#endif

    } // namespace shared_memory
} // namespace nppdx

#endif // NPPDX_TILE_STORAGE_HPP
