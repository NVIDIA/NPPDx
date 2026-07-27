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

#ifndef NPPDX_DETAIL_SYSTEM_CHECKS_HPP
#define NPPDX_DETAIL_SYSTEM_CHECKS_HPP

// We require target architecture to be Volta+ (only checking on device)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#    error "NPPDx requires GPU architecture sm_70 or higher");
#endif

#ifdef __CUDACC_RTC__

// NVRTC version check
#    ifndef NPPDX_IGNORE_DEPRECATED_COMPILER
#        if !(__CUDACC_VER_MAJOR__ >= 12)
#            error NPPDx requires NVRTC from CUDA Toolkit 12.0 or newer
#        endif
#    endif // NPPDX_IGNORE_DEPRECATED_COMPILER

// NVRTC compilation checks
#    ifndef NPPDX_IGNORE_DEPRECATED_COMPILER
static_assert((__CUDACC_VER_MAJOR__ >= 12), "NPPDx requires CUDA Runtime 12.0 or newer to work with NVRTC");
#    endif // NPPDX_IGNORE_DEPRECATED_COMPILER

#elif defined(NPPDX_CLANG_CUDA_COMPAT)

// Clang device-only PTX path. This mode intentionally uses vendored CUDA
// compatibility headers and does not include CUDA Toolkit headers.
#    ifndef __clang__
#        error NPPDx Clang CUDA compatibility mode requires clang
#    endif
#    ifndef NPPDX_IGNORE_DEPRECATED_COMPILER
#        if __clang_major__ < 21
#            error NPPDx Clang CUDA compatibility mode requires clang 21 or newer
#        endif
#    endif

#else
#    include <cuda.h>

// NVCC compilation

static_assert(CUDART_VERSION >= 12000, "NPPDx requires CUDA Runtime 12.0 or newer");
static_assert(CUDA_VERSION >= 12000, "NPPDx requires CUDA Toolkit 12.0 or newer");
#    ifdef __NVCC__
static_assert((__CUDACC_VER_MAJOR__ >= 12), "NPPDx requires NVCC 12.0 or newer");
#    endif

// SM89 MMA's require NVCC 12.4
#    ifndef NPPDX_SUPPORTS_SM89_MMA
#        define NPPDX_SUPPORTS_SM89_MMA \
            ((__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4) || __CUDACC_VER_MAJOR__ >= 13)
#        if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
#            define NPPDX_ARCH_MMA_SM89_ENABLED
#        endif
#    endif // NPPDX_SUPPORTS_SM89_MMA

#    ifndef NPPDX_IGNORE_DEPRECATED_COMPILER

// Test for GCC 7+
#        if defined(__GNUC__) && !defined(__clang__)
#            if (__GNUC__ < 7)
#                error NPPDx requires GCC in version 7 or newer
#            endif
#        endif // __GNUC__

// Test for clang 9+
#        ifdef __clang__
#            if (__clang_major__ < 9)
#                error NPPDx requires clang in version 9 or newer (experimental support for clang as host compiler)
#            endif
#        endif // __clang__

// MSVC (Visual Studio) is not supported
//#        ifdef _MSC_VER
//#            error NPPDx does not support compilation with MSVC yet
//#        endif // _MSC_VER

#    endif // NPPDX_IGNORE_DEPRECATED_COMPILER

#endif // __CUDACC_RTC__

// C++ Version
#ifndef NPPDX_IGNORE_DEPRECATED_DIALECT
#    if (__cplusplus < 201703L)
#        error NPPDx requires C++17 (or newer) enabled
#    endif
#endif // NPPDX_IGNORE_DEPRECATED_DIALECT

#endif // NPPDX_DETAIL_SYSTEM_CHECKS_HPP
