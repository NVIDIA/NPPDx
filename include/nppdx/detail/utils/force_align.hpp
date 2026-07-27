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

#ifndef NPPDX_DETAIL_UTILS_FORCE_ALIGN_HPP
#define NPPDX_DETAIL_UTILS_FORCE_ALIGN_HPP


// Memory alignment helper for fast path (simplified version of NPP's ForceAlign)
// Alignment macro for table-like format
#ifdef __CUDA_ARCH__
#    define ALIGN_DECL(N, TYPE) __align__(N) TYPE _d
#else
#    define ALIGN_DECL(N, TYPE) alignas(N) TYPE _d
#endif

// Helper types for larger alignments
struct _16_bytes {
    uint64_t x, y;
}; // 16-byte struct
struct _32_bytes {
    uint64_t x, y, z, w;
}; // 32-byte struct

// Primary template - fallback to 1-byte alignment for odd/unsupported sizes
template<int BYTES>
struct ForceAlign {
#ifdef __CUDA_ARCH__
    __align__(1) uint8_t _d[BYTES];
#else
    alignas(1) uint8_t _d[BYTES];
#endif
};

// ===== EVEN SIZES: Optimized Alignment (Odd sizes fall back to __align__(1) =====
// Lowest Common 2^K       Size    | Alignment   | Native Type  (union member for alignment only)
template<>
struct ForceAlign<2> {
    ALIGN_DECL(2, uint16_t);
};
template<>
struct ForceAlign<4> {
    ALIGN_DECL(4, uint32_t);
};
template<>
struct ForceAlign<6> {
    ALIGN_DECL(2, uint16_t);
};
template<>
struct ForceAlign<8> {
    ALIGN_DECL(8, uint64_t);
};
template<>
struct ForceAlign<10> {
    ALIGN_DECL(2, uint16_t);
};
template<>
struct ForceAlign<12> {
#ifdef __CUDA_ARCH__
    __align__(4) uint32_t _d[3];
#else
    alignas(4) uint32_t _d[3];
#endif
};
template<>
struct ForceAlign<16> {
    ALIGN_DECL(16, _16_bytes);
};
template<>
struct ForceAlign<20> {
    ALIGN_DECL(4, uint32_t);
}; //UYVP
template<>
struct ForceAlign<24> {
    ALIGN_DECL(8, uint64_t);
}; //V210
template<>
struct ForceAlign<32> {
    ALIGN_DECL(32, _32_bytes);
};
template<>
struct ForceAlign<48> {
    ALIGN_DECL(16, _16_bytes);
};
template<>
struct ForceAlign<64> {
    ALIGN_DECL(64, _32_bytes);
};
template<>
struct ForceAlign<80> {
    ALIGN_DECL(16, _16_bytes);
};
template<>
struct ForceAlign<96> {
    ALIGN_DECL(32, _32_bytes);
};
template<>
struct ForceAlign<128> {
    ALIGN_DECL(128, _32_bytes);
};
template<>
struct ForceAlign<192> {
    ALIGN_DECL(64, _32_bytes);
};
template<>
struct ForceAlign<256> {
    ALIGN_DECL(128, _32_bytes);
};
#endif
