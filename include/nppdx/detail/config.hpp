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

#ifndef NPPDX_DETAIL_CONFIG_HPP
#define NPPDX_DETAIL_CONFIG_HPP

// STL namespace alias
#include "commondx/detail/config.hpp"


#ifdef __CUDACC_RTC__
#    define NPPDX_DETAIL_USE_CUDA_STL
#endif

// clang-format off
#ifdef NPPDX_DETAIL_USE_CUDA_STL
#    define NPPDX_STD ::cuda::std
#    define NPPDX_STD_INCLUDE_ALGORITHM   <cuda/std/algorithm>
#    define NPPDX_STD_INCLUDE_ARRAY       <cuda/std/array>
#    define NPPDX_STD_INCLUDE_CASSERT     <cuda/std/cassert>
#    define NPPDX_STD_INCLUDE_CMATH       <cuda/std/cmath>
#    define NPPDX_STD_INCLUDE_CSTDDEF     <cuda/std/cstddef>
#    define NPPDX_STD_INCLUDE_CSTDINT     <cuda/std/cstdint>
#    define NPPDX_STD_INCLUDE_CSTDLIB     <cuda/std/cstdlib>
#    define NPPDX_STD_INCLUDE_CSTDIO      <cuda/std/cstdio>
#    define NPPDX_STD_INCLUDE_NEW         <cuda/std/new>
#    define NPPDX_STD_INCLUDE_NUMERIC     <cuda/std/numeric>
#    define NPPDX_STD_INCLUDE_TUPLE       <cuda/std/tuple>
#    define NPPDX_STD_INCLUDE_TYPE_TRAITS <cuda/std/type_traits>
#    define NPPDX_STD_INCLUDE_UTILITY     <cuda/std/utility>
#else
#    define NPPDX_STD ::std
#    ifdef NPPDX_CLANG_CUDA_COMPAT
#        define NPPDX_STD_INCLUDE_ALGORITHM "nppdx/detail/clang_cuda_compat/algorithm.hpp"
#    else
#        define NPPDX_STD_INCLUDE_ALGORITHM <algorithm>
#    endif
#    define NPPDX_STD_INCLUDE_ARRAY       <array>
#    define NPPDX_STD_INCLUDE_CASSERT     <cassert>
#    ifdef NPPDX_CLANG_CUDA_COMPAT
#        define NPPDX_STD_INCLUDE_CMATH "nppdx/detail/clang_cuda_compat/cmath.hpp"
#    else
#        define NPPDX_STD_INCLUDE_CMATH <cmath>
#    endif
#    define NPPDX_STD_INCLUDE_CSTDDEF     <cstddef>
#    define NPPDX_STD_INCLUDE_CSTDINT     <cstdint>
#    ifdef NPPDX_CLANG_CUDA_COMPAT
#        define NPPDX_STD_INCLUDE_CSTDLIB "nppdx/detail/clang_cuda_compat/cstdlib.hpp"
#    else
#        define NPPDX_STD_INCLUDE_CSTDLIB <cstdlib>
#    endif
#    define NPPDX_STD_INCLUDE_CSTDIO      <cstdio>
#    define NPPDX_STD_INCLUDE_NEW         <new>
#    define NPPDX_STD_INCLUDE_NUMERIC     <numeric>
#    define NPPDX_STD_INCLUDE_TUPLE       <tuple>
#    define NPPDX_STD_INCLUDE_TYPE_TRAITS <type_traits>
#    define NPPDX_STD_INCLUDE_UTILITY     <utility>
#endif
// clang-format on


#if defined(__CUDACC__) || defined(NPPDX_CLANG_CUDA_COMPAT)
#    define NPPDX_HOST_DEVICE_FORCEINLINE_FUNC __host__ __device__ __forceinline__
#else
#    define NPPDX_HOST_DEVICE_FORCEINLINE_FUNC
#endif


#endif // NPPDX_DETAIL_CONFIG_HPP
