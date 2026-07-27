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

#ifndef NPPDX_DETAIL_BACKEND_INGEST_EXGEST_NCHOOSE3_FAMILY_HPP
#define NPPDX_DETAIL_BACKEND_INGEST_EXGEST_NCHOOSE3_FAMILY_HPP

#include <cstdint>
#include <type_traits>
#if defined(__CUDACC__) && (!defined(NPPDX_ENABLE_HALF_INGEST) || (NPPDX_ENABLE_HALF_INGEST != 0))
#    include <cuda_fp16.h>
#endif
#include "nppdx/detail/backend/constants.hpp"
#include "nppdx/detail/backend/formats_impl/format_backend.hpp"
#include "nppdx/detail/backend/formats_impl/clamp_utils.hpp"

// Fast path control (from NPP)
#ifndef USE_FAST_PATH
#    define USE_FAST_PATH 1
#endif

namespace nppdx {
    namespace detail {
        namespace backend {

            // Nchoose3 backend

// nvcc: store_GMEM_12_bytes_aligned uses three st.global.cs.u32 when enabled;
// otherwise three uchar4 writes. Keep both names so the old monolithic-header
// toggle and the split-header RGB24-specific toggle both work.
#ifndef NPPDX_ST_GLOBAL_12_BYTES_PTX
#    ifdef NPPDX_RGB24_ROW_FAST_STORE_PTX
#        define NPPDX_ST_GLOBAL_12_BYTES_PTX NPPDX_RGB24_ROW_FAST_STORE_PTX
#    else
#        define NPPDX_ST_GLOBAL_12_BYTES_PTX 1
#    endif
#endif
#ifndef NPPDX_RGB24_ROW_FAST_STORE_PTX
#    define NPPDX_RGB24_ROW_FAST_STORE_PTX NPPDX_ST_GLOBAL_12_BYTES_PTX
#endif

            // Swizzle: compile-time component select so vector stays in registers (no pointer/array).
            template<int I>
            __forceinline__ __device__ uint8_t channel_comp(const uchar4& p) {
                if constexpr (I == 0)
                    return p.x;
                else if constexpr (I == 1)
                    return p.y;
                else if constexpr (I == 2)
                    return p.z;
                else
                    return p.w;
            }
            template<int I>
            __forceinline__ __device__ uint16_t channel_comp(const ushort4& p) {
                if constexpr (I == 0)
                    return p.x;
                else if constexpr (I == 1)
                    return p.y;
                else if constexpr (I == 2)
                    return p.z;
                else
                    return p.w;
            }
            // Row loader: one row of CellW pixels (3ch). Specializations for CellW 2|4|6|8 and T uint8_t|uint16_t; else one-pixel-at-a-time.
            template<typename T, int Ch0, int Ch1, int Ch2, unsigned int CellW>
            struct load_Nchoose3_row_impl {
                static __forceinline__ __device__ void call(const uint8_t* rowBase, unsigned int dataOffset, float* ch0,
                                                            float* ch1, float* ch2) {
                    constexpr size_t pixelBytes = 3 * sizeof(T);
#pragma unroll
                    for (unsigned int col = 0; col < CellW; ++col) {
                        const T* pix          = (const T*)(rowBase + (size_t)col * pixelBytes);
                        ch0[dataOffset + col] = float(pix[Ch0]);
                        ch1[dataOffset + col] = float(pix[Ch1]);
                        ch2[dataOffset + col] = float(pix[Ch2]);
                    }
                }
            };

            // CellW=2 uint8_t: 2 uchar4 (a,b). Direct .x/.y/.z/.w - no comp templates.
#define NPPDX_NCHOOSE3_ROW_IMPL_U8_2(Ch0, Ch1, Ch2, t0, t1, t2, t3, t4, t5)                                      \
    template<>                                                                                                   \
    struct load_Nchoose3_row_impl<uint8_t, Ch0, Ch1, Ch2, 2> {                                                   \
        static __forceinline__ __device__ void call(const uint8_t* rowBase, unsigned int dataOffset, float* ch0, \
                                                    float* ch1, float* ch2) {                                    \
            const uchar4 a      = *(const uchar4*)(rowBase + 0);                                                 \
            const uchar2 b      = *(const uchar2*)(rowBase + 4);                                                 \
            ch0[dataOffset + 0] = float(t0);                                                                     \
            ch0[dataOffset + 1] = float(t1);                                                                     \
            ch1[dataOffset + 0] = float(t2);                                                                     \
            ch1[dataOffset + 1] = float(t3);                                                                     \
            ch2[dataOffset + 0] = float(t4);                                                                     \
            ch2[dataOffset + 1] = float(t5);                                                                     \
        }                                                                                                        \
    };
            NPPDX_NCHOOSE3_ROW_IMPL_U8_2(0, 1, 2, a.x, a.w, a.y, b.x, a.z, b.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_2(0, 2, 1, a.x, a.w, a.z, b.y, a.y, b.x)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_2(1, 0, 2, a.y, b.x, a.x, a.w, a.z, b.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_2(1, 2, 0, a.y, b.x, a.z, b.y, a.x, a.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_2(2, 0, 1, a.z, b.y, a.x, a.w, a.y, b.x)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_2(2, 1, 0, a.z, b.y, a.y, b.x, a.x, a.w)
#undef NPPDX_NCHOOSE3_ROW_IMPL_U8_2
            // CellW=4 uint8_t: 3 uchar4 (a,b,c). Direct .x/.y/.z/.w.
#define NPPDX_NCHOOSE3_ROW_IMPL_U8_4(Ch0, Ch1, Ch2, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11)            \
    template<>                                                                                                   \
    struct load_Nchoose3_row_impl<uint8_t, Ch0, Ch1, Ch2, 4> {                                                   \
        static __forceinline__ __device__ void call(const uint8_t* rowBase, unsigned int dataOffset, float* ch0, \
                                                    float* ch1, float* ch2) {                                    \
            const uchar4 a      = *(const uchar4*)(rowBase + 0);                                                 \
            const uchar4 b      = *(const uchar4*)(rowBase + 4);                                                 \
            const uchar4 c      = *(const uchar4*)(rowBase + 8);                                                 \
            ch0[dataOffset + 0] = float(t0);                                                                     \
            ch0[dataOffset + 1] = float(t1);                                                                     \
            ch0[dataOffset + 2] = float(t2);                                                                     \
            ch0[dataOffset + 3] = float(t3);                                                                     \
            ch1[dataOffset + 0] = float(t4);                                                                     \
            ch1[dataOffset + 1] = float(t5);                                                                     \
            ch1[dataOffset + 2] = float(t6);                                                                     \
            ch1[dataOffset + 3] = float(t7);                                                                     \
            ch2[dataOffset + 0] = float(t8);                                                                     \
            ch2[dataOffset + 1] = float(t9);                                                                     \
            ch2[dataOffset + 2] = float(t10);                                                                    \
            ch2[dataOffset + 3] = float(t11);                                                                    \
        }                                                                                                        \
    };
            NPPDX_NCHOOSE3_ROW_IMPL_U8_4(0, 1, 2, a.x, a.w, b.z, c.y, a.y, b.x, b.w, c.z, a.z, b.y, c.x, c.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_4(0, 2, 1, a.x, a.w, b.z, c.y, a.z, b.y, c.x, c.w, a.y, b.x, b.w, c.z)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_4(1, 0, 2, a.y, b.x, b.w, c.z, a.x, a.w, b.z, c.y, a.z, b.y, c.x, c.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_4(1, 2, 0, a.y, b.x, b.w, c.z, a.z, b.y, c.x, c.w, a.x, a.w, b.z, c.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_4(2, 0, 1, a.z, b.y, c.x, c.w, a.x, a.w, b.z, c.y, a.y, b.x, b.w, c.z)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_4(2, 1, 0, a.z, b.y, c.x, c.w, a.y, b.x, b.w, c.z, a.x, a.w, b.z, c.y)
#undef NPPDX_NCHOOSE3_ROW_IMPL_U8_4
            // CellW=6 uint8_t: 5 uchar4 (a..e). Direct .x/.y/.z/.w.
#define NPPDX_NCHOOSE3_ROW_IMPL_U8_6(Ch0, Ch1, Ch2, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, \
                                     t15, t16, t17)                                                                  \
    template<>                                                                                                       \
    struct load_Nchoose3_row_impl<uint8_t, Ch0, Ch1, Ch2, 6> {                                                       \
        static __forceinline__ __device__ void call(const uint8_t* rowBase, unsigned int dataOffset, float* ch0,     \
                                                    float* ch1, float* ch2) {                                        \
            const uchar4 a      = *(const uchar4*)(rowBase + 0);                                                     \
            const uchar4 b      = *(const uchar4*)(rowBase + 4);                                                     \
            const uchar4 c      = *(const uchar4*)(rowBase + 8);                                                     \
            const uchar4 d      = *(const uchar4*)(rowBase + 12);                                                    \
            const uchar2 e      = *(const uchar2*)(rowBase + 16);                                                    \
            ch0[dataOffset + 0] = float(t0);                                                                         \
            ch0[dataOffset + 1] = float(t1);                                                                         \
            ch0[dataOffset + 2] = float(t2);                                                                         \
            ch0[dataOffset + 3] = float(t3);                                                                         \
            ch0[dataOffset + 4] = float(t4);                                                                         \
            ch0[dataOffset + 5] = float(t5);                                                                         \
            ch1[dataOffset + 0] = float(t6);                                                                         \
            ch1[dataOffset + 1] = float(t7);                                                                         \
            ch1[dataOffset + 2] = float(t8);                                                                         \
            ch1[dataOffset + 3] = float(t9);                                                                         \
            ch1[dataOffset + 4] = float(t10);                                                                        \
            ch1[dataOffset + 5] = float(t11);                                                                        \
            ch2[dataOffset + 0] = float(t12);                                                                        \
            ch2[dataOffset + 1] = float(t13);                                                                        \
            ch2[dataOffset + 2] = float(t14);                                                                        \
            ch2[dataOffset + 3] = float(t15);                                                                        \
            ch2[dataOffset + 4] = float(t16);                                                                        \
            ch2[dataOffset + 5] = float(t17);                                                                        \
        }                                                                                                            \
    };
            // R at bytes 0,3,6,9,12,15: a.x,a.w,b.z,c.y,d.x,d.w - G/B similarly (see CellW=4 pattern extended).
            NPPDX_NCHOOSE3_ROW_IMPL_U8_6(0, 1, 2, a.x, a.w, b.z, c.y, d.x, d.w, a.y, b.x, b.w, c.z, d.y, e.x, a.z, b.y,
                                         c.x, c.w, d.z, e.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_6(0, 2, 1, a.x, a.w, b.z, c.y, d.x, d.w, a.z, b.y, c.x, c.w, d.z, e.y, a.y, b.x,
                                         b.w, c.z, d.y, e.x)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_6(1, 0, 2, a.y, b.x, b.w, c.z, d.y, e.x, a.x, a.w, b.z, c.y, d.x, d.w, a.z, b.y,
                                         c.x, c.w, d.z, e.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_6(1, 2, 0, a.y, b.x, b.w, c.z, d.y, e.x, a.z, b.y, c.x, c.w, d.z, e.y, a.x, a.w,
                                         b.z, c.y, d.x, d.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_6(2, 0, 1, a.z, b.y, c.x, c.w, d.z, e.y, a.x, a.w, b.z, c.y, d.x, d.w, a.y, b.x,
                                         b.w, c.z, d.y, e.x)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_6(2, 1, 0, a.z, b.y, c.x, c.w, d.z, e.y, a.y, b.x, b.w, c.z, d.y, e.x, a.x, a.w,
                                         b.z, c.y, d.x, d.w)
#undef NPPDX_NCHOOSE3_ROW_IMPL_U8_6
            // CellW=8 uint8_t: 6 uchar4 (a..f). Direct .x/.y/.z/.w.
#define NPPDX_NCHOOSE3_ROW_IMPL_U8_8(Ch0, Ch1, Ch2, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, \
                                     t15, t16, t17, t18, t19, t20, t21, t22, t23)                                    \
    template<>                                                                                                       \
    struct load_Nchoose3_row_impl<uint8_t, Ch0, Ch1, Ch2, 8> {                                                       \
        static __forceinline__ __device__ void call(const uint8_t* rowBase, unsigned int dataOffset, float* ch0,     \
                                                    float* ch1, float* ch2) {                                        \
            const uchar4 a      = *(const uchar4*)(rowBase + 0);                                                     \
            const uchar4 b      = *(const uchar4*)(rowBase + 4);                                                     \
            const uchar4 c      = *(const uchar4*)(rowBase + 8);                                                     \
            const uchar4 d      = *(const uchar4*)(rowBase + 12);                                                    \
            const uchar4 e      = *(const uchar4*)(rowBase + 16);                                                    \
            const uchar4 f      = *(const uchar4*)(rowBase + 20);                                                    \
            ch0[dataOffset + 0] = float(t0);                                                                         \
            ch0[dataOffset + 1] = float(t1);                                                                         \
            ch0[dataOffset + 2] = float(t2);                                                                         \
            ch0[dataOffset + 3] = float(t3);                                                                         \
            ch0[dataOffset + 4] = float(t4);                                                                         \
            ch0[dataOffset + 5] = float(t5);                                                                         \
            ch0[dataOffset + 6] = float(t6);                                                                         \
            ch0[dataOffset + 7] = float(t7);                                                                         \
            ch1[dataOffset + 0] = float(t8);                                                                         \
            ch1[dataOffset + 1] = float(t9);                                                                         \
            ch1[dataOffset + 2] = float(t10);                                                                        \
            ch1[dataOffset + 3] = float(t11);                                                                        \
            ch1[dataOffset + 4] = float(t12);                                                                        \
            ch1[dataOffset + 5] = float(t13);                                                                        \
            ch1[dataOffset + 6] = float(t14);                                                                        \
            ch1[dataOffset + 7] = float(t15);                                                                        \
            ch2[dataOffset + 0] = float(t16);                                                                        \
            ch2[dataOffset + 1] = float(t17);                                                                        \
            ch2[dataOffset + 2] = float(t18);                                                                        \
            ch2[dataOffset + 3] = float(t19);                                                                        \
            ch2[dataOffset + 4] = float(t20);                                                                        \
            ch2[dataOffset + 5] = float(t21);                                                                        \
            ch2[dataOffset + 6] = float(t22);                                                                        \
            ch2[dataOffset + 7] = float(t23);                                                                        \
        }                                                                                                            \
    };
            // R at 0,3,6,9,12,15,18,21: ... d.x,d.w,e.z,f.y - last uchar4 reads 2 padding bytes if row is exactly 24 B.
            NPPDX_NCHOOSE3_ROW_IMPL_U8_8(0, 1, 2, a.x, a.w, b.z, c.y, d.x, d.w, e.z, f.y, a.y, b.x, b.w, c.z, d.y, e.x,
                                         e.w, f.z, a.z, b.y, c.x, c.w, d.z, e.y, f.x, f.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_8(0, 2, 1, a.x, a.w, b.z, c.y, d.x, d.w, e.z, f.y, a.z, b.y, c.x, c.w, d.z, e.y,
                                         f.x, f.w, a.y, b.x, b.w, c.z, d.y, e.x, e.w, f.z)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_8(1, 0, 2, a.y, b.x, b.w, c.z, d.y, e.x, e.w, f.z, a.x, a.w, b.z, c.y, d.x, d.w,
                                         e.z, f.y, a.z, b.y, c.x, c.w, d.z, e.y, f.x, f.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_8(1, 2, 0, a.y, b.x, b.w, c.z, d.y, e.x, e.w, f.z, a.z, b.y, c.x, c.w, d.z, e.y,
                                         f.x, f.w, a.x, a.w, b.z, c.y, d.x, d.w, e.z, f.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_8(2, 0, 1, a.z, b.y, c.x, c.w, d.z, e.y, f.x, f.w, a.x, a.w, b.z, c.y, d.x, d.w,
                                         e.z, f.y, a.y, b.x, b.w, c.z, d.y, e.x, e.w, f.z)
            NPPDX_NCHOOSE3_ROW_IMPL_U8_8(2, 1, 0, a.z, b.y, c.x, c.w, d.z, e.y, f.x, f.w, a.y, b.x, b.w, c.z, d.y, e.x,
                                         e.w, f.z, a.x, a.w, b.z, c.y, d.x, d.w, e.z, f.y)
#undef NPPDX_NCHOOSE3_ROW_IMPL_U8_8

            // CellW=2 uint16_t: 2 ushort4 (a,b). Direct .x/.y/.z/.w.
#define NPPDX_NCHOOSE3_ROW_IMPL_U16_2(Ch0, Ch1, Ch2, t0, t1, t2, t3, t4, t5)                                     \
    template<>                                                                                                   \
    struct load_Nchoose3_row_impl<uint16_t, Ch0, Ch1, Ch2, 2> {                                                  \
        static __forceinline__ __device__ void call(const uint8_t* rowBase, unsigned int dataOffset, float* ch0, \
                                                    float* ch1, float* ch2) {                                    \
            const ushort4 a     = *(const ushort4*)(rowBase + 0);                                                \
            const ushort2 b     = *(const ushort2*)(rowBase + 8);                                                \
            ch0[dataOffset + 0] = float(t0);                                                                     \
            ch0[dataOffset + 1] = float(t1);                                                                     \
            ch1[dataOffset + 0] = float(t2);                                                                     \
            ch1[dataOffset + 1] = float(t3);                                                                     \
            ch2[dataOffset + 0] = float(t4);                                                                     \
            ch2[dataOffset + 1] = float(t5);                                                                     \
        }                                                                                                        \
    };
            NPPDX_NCHOOSE3_ROW_IMPL_U16_2(0, 1, 2, a.x, a.w, a.y, b.x, a.z, b.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_2(0, 2, 1, a.x, a.w, a.z, b.y, a.y, b.x)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_2(1, 0, 2, a.y, b.x, a.x, a.w, a.z, b.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_2(1, 2, 0, a.y, b.x, a.z, b.y, a.x, a.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_2(2, 0, 1, a.z, b.y, a.x, a.w, a.y, b.x)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_2(2, 1, 0, a.z, b.y, a.y, b.x, a.x, a.w)
#undef NPPDX_NCHOOSE3_ROW_IMPL_U16_2
#define NPPDX_NCHOOSE3_ROW_IMPL_U16_4(Ch0, Ch1, Ch2, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11)           \
    template<>                                                                                                   \
    struct load_Nchoose3_row_impl<uint16_t, Ch0, Ch1, Ch2, 4> {                                                  \
        static __forceinline__ __device__ void call(const uint8_t* rowBase, unsigned int dataOffset, float* ch0, \
                                                    float* ch1, float* ch2) {                                    \
            const ushort4 a     = *(const ushort4*)(rowBase + 0);                                                \
            const ushort4 b     = *(const ushort4*)(rowBase + 8);                                                \
            const ushort4 c     = *(const ushort4*)(rowBase + 16);                                               \
            ch0[dataOffset + 0] = float(t0);                                                                     \
            ch0[dataOffset + 1] = float(t1);                                                                     \
            ch0[dataOffset + 2] = float(t2);                                                                     \
            ch0[dataOffset + 3] = float(t3);                                                                     \
            ch1[dataOffset + 0] = float(t4);                                                                     \
            ch1[dataOffset + 1] = float(t5);                                                                     \
            ch1[dataOffset + 2] = float(t6);                                                                     \
            ch1[dataOffset + 3] = float(t7);                                                                     \
            ch2[dataOffset + 0] = float(t8);                                                                     \
            ch2[dataOffset + 1] = float(t9);                                                                     \
            ch2[dataOffset + 2] = float(t10);                                                                    \
            ch2[dataOffset + 3] = float(t11);                                                                    \
        }                                                                                                        \
    };
            NPPDX_NCHOOSE3_ROW_IMPL_U16_4(0, 1, 2, a.x, a.w, b.z, c.y, a.y, b.x, b.w, c.z, a.z, b.y, c.x, c.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_4(0, 2, 1, a.x, a.w, b.z, c.y, a.z, b.y, c.x, c.w, a.y, b.x, b.w, c.z)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_4(1, 0, 2, a.y, b.x, b.w, c.z, a.x, a.w, b.z, c.y, a.z, b.y, c.x, c.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_4(1, 2, 0, a.y, b.x, b.w, c.z, a.z, b.y, c.x, c.w, a.x, a.w, b.z, c.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_4(2, 0, 1, a.z, b.y, c.x, c.w, a.x, a.w, b.z, c.y, a.y, b.x, b.w, c.z)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_4(2, 1, 0, a.z, b.y, c.x, c.w, a.y, b.x, b.w, c.z, a.x, a.w, b.z, c.y)
#undef NPPDX_NCHOOSE3_ROW_IMPL_U16_4
#define NPPDX_NCHOOSE3_ROW_IMPL_U16_6(Ch0, Ch1, Ch2, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, \
                                      t15, t16, t17)                                                                  \
    template<>                                                                                                        \
    struct load_Nchoose3_row_impl<uint16_t, Ch0, Ch1, Ch2, 6> {                                                       \
        static __forceinline__ __device__ void call(const uint8_t* rowBase, unsigned int dataOffset, float* ch0,      \
                                                    float* ch1, float* ch2) {                                         \
            const ushort4 a     = *(const ushort4*)(rowBase + 0);                                                     \
            const ushort4 b     = *(const ushort4*)(rowBase + 8);                                                     \
            const ushort4 c     = *(const ushort4*)(rowBase + 16);                                                    \
            const ushort4 d     = *(const ushort4*)(rowBase + 24);                                                    \
            const ushort2 e     = *(const ushort2*)(rowBase + 32);                                                    \
            ch0[dataOffset + 0] = float(t0);                                                                          \
            ch0[dataOffset + 1] = float(t1);                                                                          \
            ch0[dataOffset + 2] = float(t2);                                                                          \
            ch0[dataOffset + 3] = float(t3);                                                                          \
            ch0[dataOffset + 4] = float(t4);                                                                          \
            ch0[dataOffset + 5] = float(t5);                                                                          \
            ch1[dataOffset + 0] = float(t6);                                                                          \
            ch1[dataOffset + 1] = float(t7);                                                                          \
            ch1[dataOffset + 2] = float(t8);                                                                          \
            ch1[dataOffset + 3] = float(t9);                                                                          \
            ch1[dataOffset + 4] = float(t10);                                                                         \
            ch1[dataOffset + 5] = float(t11);                                                                         \
            ch2[dataOffset + 0] = float(t12);                                                                         \
            ch2[dataOffset + 1] = float(t13);                                                                         \
            ch2[dataOffset + 2] = float(t14);                                                                         \
            ch2[dataOffset + 3] = float(t15);                                                                         \
            ch2[dataOffset + 4] = float(t16);                                                                         \
            ch2[dataOffset + 5] = float(t17);                                                                         \
        }                                                                                                             \
    };
            NPPDX_NCHOOSE3_ROW_IMPL_U16_6(0, 1, 2, a.x, a.w, b.z, c.y, d.x, d.w, a.y, b.x, b.w, c.z, d.y, e.x, a.z, b.y,
                                          c.x, c.w, d.z, e.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_6(0, 2, 1, a.x, a.w, b.z, c.y, d.x, d.w, a.z, b.y, c.x, c.w, d.z, e.y, a.y, b.x,
                                          b.w, c.z, d.y, e.x)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_6(1, 0, 2, a.y, b.x, b.w, c.z, d.y, e.x, a.x, a.w, b.z, c.y, d.x, d.w, a.z, b.y,
                                          c.x, c.w, d.z, e.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_6(1, 2, 0, a.y, b.x, b.w, c.z, d.y, e.x, a.z, b.y, c.x, c.w, d.z, e.y, a.x, a.w,
                                          b.z, c.y, d.x, d.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_6(2, 0, 1, a.z, b.y, c.x, c.w, d.z, e.y, a.x, a.w, b.z, c.y, d.x, d.w, a.y, b.x,
                                          b.w, c.z, d.y, e.x)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_6(2, 1, 0, a.z, b.y, c.x, c.w, d.z, e.y, a.y, b.x, b.w, c.z, d.y, e.x, a.x, a.w,
                                          b.z, c.y, d.x, d.w)
#undef NPPDX_NCHOOSE3_ROW_IMPL_U16_6
#define NPPDX_NCHOOSE3_ROW_IMPL_U16_8(Ch0, Ch1, Ch2, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, \
                                      t15, t16, t17, t18, t19, t20, t21, t22, t23)                                    \
    template<>                                                                                                        \
    struct load_Nchoose3_row_impl<uint16_t, Ch0, Ch1, Ch2, 8> {                                                       \
        static __forceinline__ __device__ void call(const uint8_t* rowBase, unsigned int dataOffset, float* ch0,      \
                                                    float* ch1, float* ch2) {                                         \
            const ushort4 a     = *(const ushort4*)(rowBase + 0);                                                     \
            const ushort4 b     = *(const ushort4*)(rowBase + 8);                                                     \
            const ushort4 c     = *(const ushort4*)(rowBase + 16);                                                    \
            const ushort4 d     = *(const ushort4*)(rowBase + 24);                                                    \
            const ushort4 e     = *(const ushort4*)(rowBase + 32);                                                    \
            const ushort4 f     = *(const ushort4*)(rowBase + 40);                                                    \
            ch0[dataOffset + 0] = float(t0);                                                                          \
            ch0[dataOffset + 1] = float(t1);                                                                          \
            ch0[dataOffset + 2] = float(t2);                                                                          \
            ch0[dataOffset + 3] = float(t3);                                                                          \
            ch0[dataOffset + 4] = float(t4);                                                                          \
            ch0[dataOffset + 5] = float(t5);                                                                          \
            ch0[dataOffset + 6] = float(t6);                                                                          \
            ch0[dataOffset + 7] = float(t7);                                                                          \
            ch1[dataOffset + 0] = float(t8);                                                                          \
            ch1[dataOffset + 1] = float(t9);                                                                          \
            ch1[dataOffset + 2] = float(t10);                                                                         \
            ch1[dataOffset + 3] = float(t11);                                                                         \
            ch1[dataOffset + 4] = float(t12);                                                                         \
            ch1[dataOffset + 5] = float(t13);                                                                         \
            ch1[dataOffset + 6] = float(t14);                                                                         \
            ch1[dataOffset + 7] = float(t15);                                                                         \
            ch2[dataOffset + 0] = float(t16);                                                                         \
            ch2[dataOffset + 1] = float(t17);                                                                         \
            ch2[dataOffset + 2] = float(t18);                                                                         \
            ch2[dataOffset + 3] = float(t19);                                                                         \
            ch2[dataOffset + 4] = float(t20);                                                                         \
            ch2[dataOffset + 5] = float(t21);                                                                         \
            ch2[dataOffset + 6] = float(t22);                                                                         \
            ch2[dataOffset + 7] = float(t23);                                                                         \
        }                                                                                                             \
    };
            NPPDX_NCHOOSE3_ROW_IMPL_U16_8(0, 1, 2, a.x, a.w, b.z, c.y, d.x, d.w, e.z, f.y, a.y, b.x, b.w, c.z, d.y, e.x,
                                          e.w, f.z, a.z, b.y, c.x, c.w, d.z, e.y, f.x, f.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_8(0, 2, 1, a.x, a.w, b.z, c.y, d.x, d.w, e.z, f.y, a.z, b.y, c.x, c.w, d.z, e.y,
                                          f.x, f.w, a.y, b.x, b.w, c.z, d.y, e.x, e.w, f.z)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_8(1, 0, 2, a.y, b.x, b.w, c.z, d.y, e.x, e.w, f.z, a.x, a.w, b.z, c.y, d.x, d.w,
                                          e.z, f.y, a.z, b.y, c.x, c.w, d.z, e.y, f.x, f.w)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_8(1, 2, 0, a.y, b.x, b.w, c.z, d.y, e.x, e.w, f.z, a.z, b.y, c.x, c.w, d.z, e.y,
                                          f.x, f.w, a.x, a.w, b.z, c.y, d.x, d.w, e.z, f.y)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_8(2, 0, 1, a.z, b.y, c.x, c.w, d.z, e.y, f.x, f.w, a.x, a.w, b.z, c.y, d.x, d.w,
                                          e.z, f.y, a.y, b.x, b.w, c.z, d.y, e.x, e.w, f.z)
            NPPDX_NCHOOSE3_ROW_IMPL_U16_8(2, 1, 0, a.z, b.y, c.x, c.w, d.z, e.y, f.x, f.w, a.y, b.x, b.w, c.z, d.y, e.x,
                                          e.w, f.z, a.x, a.w, b.z, c.y, d.x, d.w, e.z, f.y)
#undef NPPDX_NCHOOSE3_ROW_IMPL_U16_8

#if defined(__CUDACC__) && (!defined(NPPDX_ENABLE_HALF_INGEST) || (NPPDX_ENABLE_HALF_INGEST != 0))
            // __half (fp16) RGB: one-pixel-at-a-time, __half2float. No vectorized row path (no half4).
            template<int Ch0, int Ch1, int Ch2, unsigned int CellW>
            struct load_Nchoose3_row_impl<__half, Ch0, Ch1, Ch2, CellW> {
                static __forceinline__ __device__ void call(const uint8_t* rowBase, unsigned int dataOffset, float* ch0,
                                                            float* ch1, float* ch2) {
                    constexpr size_t pixelBytes = 3 * sizeof(__half);
#    pragma unroll
                    for (unsigned int col = 0; col < CellW; ++col) {
                        const __half* pix     = (const __half*)(rowBase + (size_t)col * pixelBytes);
                        ch0[dataOffset + col] = __half2float(pix[Ch0]);
                        ch1[dataOffset + col] = __half2float(pix[Ch1]);
                        ch2[dataOffset + col] = __half2float(pix[Ch2]);
                    }
                }
            };
#endif

            // Row-loop + load_Nchoose3_row_impl (template specializations per CellW). Aligned row → row impl; else scalar.
            template<typename T, int Ch0, int Ch1, int Ch2, int NChannels, typename CellDataType>
            __forceinline__ __device__ void load_Nchoose3(const uint8_t** src, const size_t* stride, CellDataType& cell,
                                                          const uint32_t nCellX, const uint32_t nCellY) {
                float* ch0 = cell.channel_ptr(0);
                float* ch1 = cell.channel_ptr(1);
                float* ch2 = cell.channel_ptr(2);

                constexpr unsigned int cell_width  = CellDataType::cell_width;
                constexpr unsigned int cell_height = CellDataType::cell_height;
                constexpr size_t       pixelBytes  = NChannels * sizeof(T);

                const uint8_t* srcPlane  = src[0];
                const size_t   srcStride = stride[0];
                const size_t   src0 =
                    (size_t)srcPlane + nCellY * cell_height * srcStride + nCellX * cell_width * pixelBytes;

#pragma unroll
                for (unsigned int row = 0; row < cell_height; ++row) {
                    const uint8_t*     rowBase    = (const uint8_t*)(src0 + row * srcStride);
                    const unsigned int dataOffset = row * cell_width;

                    if constexpr (NChannels == 3 && sizeof(T) == 1 &&
                                  (cell_width == 2 || cell_width == 4 || cell_width == 6 || cell_width == 8)) {
                        const bool rowAligned4 = (reinterpret_cast<uintptr_t>(rowBase) & 3u) == 0u;
                        if (rowAligned4) {
                            load_Nchoose3_row_impl<T, Ch0, Ch1, Ch2, cell_width>::call(rowBase, dataOffset, ch0, ch1,
                                                                                       ch2);
                        } else {
#pragma unroll
                            for (unsigned int col = 0; col < cell_width; ++col) {
                                const T* pix          = (const T*)(rowBase + (size_t)col * pixelBytes);
                                ch0[dataOffset + col] = float(pix[Ch0]);
                                ch1[dataOffset + col] = float(pix[Ch1]);
                                ch2[dataOffset + col] = float(pix[Ch2]);
                            }
                        }
                    } else if constexpr (NChannels == 3 && NPPDX_STD::is_same_v<T, uint16_t> &&
                                         (cell_width == 2 || cell_width == 4 || cell_width == 6 || cell_width == 8)) {
                        const bool rowAligned8 = (reinterpret_cast<uintptr_t>(rowBase) & 7u) == 0u;
                        if (rowAligned8) {
                            load_Nchoose3_row_impl<T, Ch0, Ch1, Ch2, cell_width>::call(rowBase, dataOffset, ch0, ch1,
                                                                                       ch2);
                        } else {
#pragma unroll
                            for (unsigned int col = 0; col < cell_width; ++col) {
                                const T* pix          = (const T*)(rowBase + (size_t)col * pixelBytes);
                                ch0[dataOffset + col] = float(pix[Ch0]);
                                ch1[dataOffset + col] = float(pix[Ch1]);
                                ch2[dataOffset + col] = float(pix[Ch2]);
                            }
                        }
#if defined(__CUDACC__) && (!defined(NPPDX_ENABLE_HALF_INGEST) || (NPPDX_ENABLE_HALF_INGEST != 0))
                    } else if constexpr (NChannels == 3 && std::is_same_v<T, __half>) {
                        load_Nchoose3_row_impl<__half, Ch0, Ch1, Ch2, cell_width>::call(rowBase, dataOffset, ch0, ch1,
                                                                                        ch2);
                    } else {
#else
                    } else {
#endif
#pragma unroll
                        for (unsigned int col = 0; col < cell_width; ++col) {
                            const size_t pixelOff = (size_t)col * pixelBytes;
                            if constexpr (sizeof(T) == 1 && NChannels <= 4 && (NChannels * sizeof(T) % 4 == 0)) {
                                const uchar4 p        = *(const uchar4*)(rowBase + pixelOff);
                                ch0[dataOffset + col] = float(channel_comp<Ch0>(p));
                                ch1[dataOffset + col] = float(channel_comp<Ch1>(p));
                                ch2[dataOffset + col] = float(channel_comp<Ch2>(p));
                            } else if constexpr (NPPDX_STD::is_same_v<T, uint16_t> && NChannels <= 4 &&
                                                 (NChannels * sizeof(T) % 8 == 0)) {
                                const ushort4 p       = *(const ushort4*)(rowBase + pixelOff);
                                ch0[dataOffset + col] = float(channel_comp<Ch0>(p));
                                ch1[dataOffset + col] = float(channel_comp<Ch1>(p));
                                ch2[dataOffset + col] = float(channel_comp<Ch2>(p));
                            } else {
                                const T* pix          = (const T*)(rowBase + pixelOff);
                                ch0[dataOffset + col] = float(pix[Ch0]);
                                ch1[dataOffset + col] = float(pix[Ch1]);
                                ch2[dataOffset + col] = float(pix[Ch2]);
                            }
                        }
                    }
                }
            }

            // Fast path: direct T* writes (no PixelBlock), register-friendly, no alignment requirement.
            template<typename T, bit_depth R, int Ch0, int Ch1, int Ch2, int AlphaChannel, int NChannels,
                     typename CellDataType>
            __forceinline__ __device__ void save_Nchoose3(uint8_t** dst, const size_t* stride, bool clip, float alpha,
                                                          const CellDataType& cell, const uint32_t nCellX,
                                                          const uint32_t nCellY) {
                const float* ch0 = cell.channel_ptr(0);
                const float* ch1 = cell.channel_ptr(1);
                const float* ch2 = cell.channel_ptr(2);

                constexpr unsigned int cell_width  = CellDataType::cell_width;
                constexpr unsigned int cell_height = CellDataType::cell_height;
                constexpr size_t       pixelBytes  = NChannels * sizeof(T);

                uint8_t*     dstPlane  = dst[0];
                const size_t dstStride = stride[0];
                const size_t dst0 =
                    (size_t)dstPlane + nCellY * cell_height * dstStride + nCellX * cell_width * pixelBytes;

                if (clip) {
                    if constexpr (AlphaChannel >= 0) {
                        const T a = (T)fclampf<R>(alpha);
#pragma unroll
                        for (unsigned int row = 0; row < cell_height; ++row) {
                            T*           rowBase    = (T*)(dst0 + row * dstStride);
                            const size_t dataOffset = row * cell_width;
#pragma unroll
                            for (unsigned int col = 0; col < cell_width; ++col) {
                                const size_t off            = (size_t)col * NChannels;
                                rowBase[off + Ch0]          = (T)fclampf<R>(ch0[dataOffset + col]);
                                rowBase[off + Ch1]          = (T)fclampf<R>(ch1[dataOffset + col]);
                                rowBase[off + Ch2]          = (T)fclampf<R>(ch2[dataOffset + col]);
                                rowBase[off + AlphaChannel] = a;
                            }
                        }
                    } else {
#pragma unroll
                        for (unsigned int row = 0; row < cell_height; ++row) {
                            T*           rowBase    = (T*)(dst0 + row * dstStride);
                            const size_t dataOffset = row * cell_width;
#pragma unroll
                            for (unsigned int col = 0; col < cell_width; ++col) {
                                const size_t off   = (size_t)col * NChannels;
                                rowBase[off + Ch0] = (T)fclampf<R>(ch0[dataOffset + col]);
                                rowBase[off + Ch1] = (T)fclampf<R>(ch1[dataOffset + col]);
                                rowBase[off + Ch2] = (T)fclampf<R>(ch2[dataOffset + col]);
                            }
                        }
                    }
                } else {
                    if constexpr (AlphaChannel >= 0) {
                        const T a = (T)alpha;
#pragma unroll
                        for (unsigned int row = 0; row < cell_height; ++row) {
                            T*           rowBase    = (T*)(dst0 + row * dstStride);
                            const size_t dataOffset = row * cell_width;
#pragma unroll
                            for (unsigned int col = 0; col < cell_width; ++col) {
                                const size_t off            = (size_t)col * NChannels;
                                rowBase[off + Ch0]          = (T)ch0[dataOffset + col];
                                rowBase[off + Ch1]          = (T)ch1[dataOffset + col];
                                rowBase[off + Ch2]          = (T)ch2[dataOffset + col];
                                rowBase[off + AlphaChannel] = a;
                            }
                        }
                    } else {
#pragma unroll
                        for (unsigned int row = 0; row < cell_height; ++row) {
                            T*           rowBase    = (T*)(dst0 + row * dstStride);
                            const size_t dataOffset = row * cell_width;
#pragma unroll
                            for (unsigned int col = 0; col < cell_width; ++col) {
                                const size_t off   = (size_t)col * NChannels;
                                rowBase[off + Ch0] = (T)ch0[dataOffset + col];
                                rowBase[off + Ch1] = (T)ch1[dataOffset + col];
                                rowBase[off + Ch2] = (T)ch2[dataOffset + col];
                            }
                        }
                    }
                }
            }

#if NPPDX_ST_GLOBAL_12_BYTES_PTX && defined(__CUDACC__) && defined(__NVCC__)
            // Three st.global.cs.u32 (12 bytes). dst must be 4-byte aligned; w* are pre-packed.
            __forceinline__ __device__ void nppdx_st_global_12_bytes_ptx(uint8_t* dst, uint32_t w0, uint32_t w1,
                                                                         uint32_t w2) {
                uint8_t* const p4 = dst + 4;
                uint8_t* const p8 = dst + 8;
                asm volatile("st.global.cs.u32 [%0], %1;\n\t"
                             "st.global.cs.u32 [%2], %3;\n\t"
                             "st.global.cs.u32 [%4], %5;"
                             :
                             : "l"(dst), "r"(w0), "l"(p4), "r"(w1), "l"(p8), "r"(w2)
                             : "memory");
            }
#endif

#if defined(__CUDACC__) || defined(NPPDX_CLANG_CUDA_COMPAT)
            __forceinline__ __device__ void store_GMEM_12_bytes_aligned(uint8_t* rowBase, const uchar4& a,
                                                                        const uchar4& b, const uchar4& c) {
#    if NPPDX_ST_GLOBAL_12_BYTES_PTX && defined(__NVCC__)
                // uchar4 is 4 bytes x,y,z,w at increasing addresses — same bit pattern as little-endian u32.
                const uint32_t w0 = *reinterpret_cast<const uint32_t*>(&a);
                const uint32_t w1 = *reinterpret_cast<const uint32_t*>(&b);
                const uint32_t w2 = *reinterpret_cast<const uint32_t*>(&c);
                nppdx_st_global_12_bytes_ptx(rowBase, w0, w1, w2);
#    else
                uchar4* const rowB4 = reinterpret_cast<uchar4*>(rowBase);
                rowB4[0]            = a;
                rowB4[1]            = b;
                rowB4[2]            = c;
#    endif
            }
#endif


            // <4,2,3> uint8_t specialization, Ch=(0,1,2), RGB24 packed.
            template<>
            __forceinline__ __device__ void
            save_Nchoose3<uint8_t, bit_depth::bpp_8u, 0, 1, 2, -1, 3, cell_4x2_3ch_float>(
                uint8_t** dst, const size_t* stride, bool clip, float alpha, const cell_4x2_3ch_float& cell,
                const uint32_t nCellX, const uint32_t nCellY) {
                (void)alpha;
                const float*           ch0         = cell.channel_ptr(0);
                const float*           ch1         = cell.channel_ptr(1);
                const float*           ch2         = cell.channel_ptr(2);
                constexpr unsigned int cell_width  = 4;
                constexpr unsigned int cell_height = 2;
                constexpr size_t       pixelBytes  = 3 * sizeof(uint8_t);

                uint8_t*     dstPlane  = dst[0];
                const size_t dstStride = stride[0];
                const size_t dst0 =
                    (size_t)dstPlane + nCellY * cell_height * dstStride + nCellX * cell_width * pixelBytes;

                if (clip) {
#pragma unroll
                    for (unsigned int row = 0; row < cell_height; ++row) {
                        uint8_t*           rowBase     = (uint8_t*)(dst0 + row * dstStride);
                        const unsigned int d           = row * cell_width;
                        const bool         rowUseBlock = (reinterpret_cast<uintptr_t>(rowBase) & 3u) == 0u;
                        if (rowUseBlock) {
                            uchar4 a, b, c;
                            a.x = (unsigned char)fclampf<bit_depth::bpp_8u>(ch0[d + 0]);
                            a.y = (unsigned char)fclampf<bit_depth::bpp_8u>(ch1[d + 0]);
                            a.z = (unsigned char)fclampf<bit_depth::bpp_8u>(ch2[d + 0]);
                            a.w = (unsigned char)fclampf<bit_depth::bpp_8u>(ch0[d + 1]);
                            b.x = (unsigned char)fclampf<bit_depth::bpp_8u>(ch1[d + 1]);
                            b.y = (unsigned char)fclampf<bit_depth::bpp_8u>(ch2[d + 1]);
                            b.z = (unsigned char)fclampf<bit_depth::bpp_8u>(ch0[d + 2]);
                            b.w = (unsigned char)fclampf<bit_depth::bpp_8u>(ch1[d + 2]);
                            c.x = (unsigned char)fclampf<bit_depth::bpp_8u>(ch2[d + 2]);
                            c.y = (unsigned char)fclampf<bit_depth::bpp_8u>(ch0[d + 3]);
                            c.z = (unsigned char)fclampf<bit_depth::bpp_8u>(ch1[d + 3]);
                            c.w = (unsigned char)fclampf<bit_depth::bpp_8u>(ch2[d + 3]);
                            store_GMEM_12_bytes_aligned(rowBase, a, b, c);
                        } else {
#pragma unroll
                            for (unsigned int c = 0; c < 4; ++c) {
                                const size_t off = (size_t)c * 3u;
                                const float  rr  = ch0[d + c];
                                const float  gg  = ch1[d + c];
                                const float  bb  = ch2[d + c];
                                rowBase[off + 0] = (uint8_t)fclampf<bit_depth::bpp_8u>(rr);
                                rowBase[off + 1] = (uint8_t)fclampf<bit_depth::bpp_8u>(gg);
                                rowBase[off + 2] = (uint8_t)fclampf<bit_depth::bpp_8u>(bb);
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (unsigned int row = 0; row < cell_height; ++row) {
                        uint8_t*           rowBase     = (uint8_t*)(dst0 + row * dstStride);
                        const unsigned int d           = row * cell_width;
                        const bool         rowUseBlock = (reinterpret_cast<uintptr_t>(rowBase) & 3u) == 0u;
                        if (rowUseBlock) {
                            uchar4 a, b, c;
                            a.x = (unsigned char)ch0[d + 0];
                            a.y = (unsigned char)ch1[d + 0];
                            a.z = (unsigned char)ch2[d + 0];
                            a.w = (unsigned char)ch0[d + 1];
                            b.x = (unsigned char)ch1[d + 1];
                            b.y = (unsigned char)ch2[d + 1];
                            b.z = (unsigned char)ch0[d + 2];
                            b.w = (unsigned char)ch1[d + 2];
                            c.x = (unsigned char)ch2[d + 2];
                            c.y = (unsigned char)ch0[d + 3];
                            c.z = (unsigned char)ch1[d + 3];
                            c.w = (unsigned char)ch2[d + 3];
                            store_GMEM_12_bytes_aligned(rowBase, a, b, c);
                        } else {
#pragma unroll
                            for (unsigned int c = 0; c < 4; ++c) {
                                const size_t off = (size_t)c * 3u;
                                rowBase[off + 0] = (unsigned char)ch0[d + c];
                                rowBase[off + 1] = (unsigned char)ch1[d + c];
                                rowBase[off + 2] = (unsigned char)ch2[d + c];
                            }
                        }
                    }
                }
            }

            template<>
            __forceinline__ __device__ void
            save_Nchoose3<uint8_t, bit_depth::bpp_8u, 0, 1, 2, -1, 3, cell_4x2_3ch_const_float>(
                uint8_t** dst, const size_t* stride, bool clip, float alpha, const cell_4x2_3ch_const_float& cell,
                const uint32_t nCellX, const uint32_t nCellY) {
                save_Nchoose3<uint8_t, bit_depth::bpp_8u, 0, 1, 2, -1, 3, cell_4x2_3ch_float>(
                    dst, stride, clip, alpha, reinterpret_cast<const cell_4x2_3ch_float&>(cell), nCellX, nCellY);
            }
            // Safe path: two full-width loops (both unrollable), fetch NChannels when in bounds then replicate halo.
            template<typename T, int Ch0, int Ch1, int Ch2, int NChannels, typename CellDataType>
            __forceinline__ __device__ void safe_load_Nchoose3(const uint8_t** src, const size_t* stride,
                                                               CellDataType& cell, const int32_t nCellX,
                                                               const int32_t nCellY, const int32_t imageWidth,
                                                               const int32_t imageHeight) {
                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;

#if USE_FAST_PATH
                if (basePixelX >= 0 && (basePixelX + cell_width - 1) < imageWidth && basePixelY >= 0 &&
                    (basePixelY + cell_height - 1) < imageHeight) {
                    load_Nchoose3<T, Ch0, Ch1, Ch2, NChannels>(src, stride, cell, (uint32_t)nCellX, (uint32_t)nCellY);
                    return;
                }
#endif

                float*         ch0_data  = cell.channel_ptr(0);
                float*         ch1_data  = cell.channel_ptr(1);
                float*         ch2_data  = cell.channel_ptr(2);
                const uint8_t* srcPlane  = src[0];
                const size_t   srcStride = stride[0];
                // Cell cannot straddle both left and right; fetch only needed edge side
                const bool needLeft = (basePixelX < 0);

#pragma unroll
                for (int32_t row = 0; row < cell_height; ++row) {
                    const int32_t pixelY     = basePixelY + row;
                    const int32_t clampY     = nppdx_max(0, nppdx_min(pixelY, (int32_t)imageHeight - 1));
                    const int32_t dataOffset = row * cell_width;

                    const T*      rowBase = (const T*)(srcPlane + clampY * srcStride);
                    const int32_t edgeIdx = needLeft ? 0 : (imageWidth - 1) * NChannels;
                    const float   e0      = float(rowBase[edgeIdx + Ch0]);
                    const float   e1      = float(rowBase[edgeIdx + Ch1]);
                    const float   e2      = float(rowBase[edgeIdx + Ch2]);

                    // Single loop (full cell width, unrollable): in bounds -> fetch NChannels, else -> replicate edge
#pragma unroll
                    for (int32_t i = 0; i < cell_width; ++i) {
                        const int32_t imgX = basePixelX + i;
                        if (imgX >= 0 && imgX < (int32_t)imageWidth) {
                            const size_t off         = (size_t)imgX * NChannels;
                            ch0_data[dataOffset + i] = float(rowBase[off + Ch0]);
                            ch1_data[dataOffset + i] = float(rowBase[off + Ch1]);
                            ch2_data[dataOffset + i] = float(rowBase[off + Ch2]);
                        } else {
                            ch0_data[dataOffset + i] = e0;
                            ch1_data[dataOffset + i] = e1;
                            ch2_data[dataOffset + i] = e2;
                        }
                    }
                }
            }


            // Safe path: valid range once, segment loop (write only [validStartX, validEndX)), branch-free interior.
            template<typename T, bit_depth R, int Ch0, int Ch1, int Ch2, int AlphaChannel, int NChannels,
                     typename CellDataType>
            __forceinline__ __device__ void safe_save_Nchoose3(uint8_t** dst, const size_t* stride, bool clip,
                                                               float alpha, const CellDataType& cell, int32_t nCellX,
                                                               int32_t nCellY, int32_t imageWidth,
                                                               int32_t imageHeight) {
                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;

#if USE_FAST_PATH
                if (nCellX >= 0 && nCellX * cell_width + cell_width - 1 < imageWidth && nCellY >= 0 &&
                    nCellY * cell_height + cell_height - 1 < imageHeight) {
                    save_Nchoose3<T, R, Ch0, Ch1, Ch2, AlphaChannel, NChannels>(dst, stride, clip, alpha, cell,
                                                                                (uint32_t)nCellX, (uint32_t)nCellY);
                    return;
                }
#endif

                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;

                const float* ch0_data = cell.channel_ptr(0);
                const float* ch1_data = cell.channel_ptr(1);
                const float* ch2_data = cell.channel_ptr(2);

                const int32_t validStartX = nppdx_max(0, -basePixelX);
                const int32_t validEndX   = nppdx_min(cell_width, imageWidth - basePixelX);
                const int32_t validStartY = nppdx_max(0, -basePixelY);
                const int32_t validEndY   = nppdx_min(cell_height, imageHeight - basePixelY);
                if (validStartX >= validEndX || validStartY >= validEndY)
                    return;

                constexpr size_t pixelBytes = NChannels * sizeof(T);
                uint8_t*         dstPlane   = dst[0];
                const size_t     dstStride  = stride[0];
                const size_t     dst0 =
                    (size_t)dstPlane + nCellY * cell_height * dstStride + nCellX * cell_width * pixelBytes;

                // Same pattern as safe_load_Nchoose3: T* rowBase per row, off = i*NChannels, direct channel indexing
                if constexpr (AlphaChannel >= 0) {
                    const T alphaValue = clip ? (T)fclampf<R>(alpha) : (T)alpha;
#pragma unroll
                    for (int32_t row = 0; row < cell_height; ++row) {
                        if (row >= validStartY && row < validEndY) {
                            T*            rowBase    = (T*)(dst0 + row * dstStride);
                            const int32_t dataOffset = row * cell_width;

                            if (clip) {
#pragma unroll
                                for (int32_t i = 0; i < cell_width; ++i) {
                                    if (i < validEndX) {
                                        const size_t off            = (size_t)i * NChannels;
                                        rowBase[off + Ch0]          = (T)fclampf<R>(ch0_data[dataOffset + i]);
                                        rowBase[off + Ch1]          = (T)fclampf<R>(ch1_data[dataOffset + i]);
                                        rowBase[off + Ch2]          = (T)fclampf<R>(ch2_data[dataOffset + i]);
                                        rowBase[off + AlphaChannel] = alphaValue;
                                    }
                                }
                            } else {
#pragma unroll
                                for (int32_t i = 0; i < cell_width; ++i) {
                                    if (i < validEndX) {
                                        const size_t off            = (size_t)i * NChannels;
                                        rowBase[off + Ch0]          = (T)ch0_data[dataOffset + i];
                                        rowBase[off + Ch1]          = (T)ch1_data[dataOffset + i];
                                        rowBase[off + Ch2]          = (T)ch2_data[dataOffset + i];
                                        rowBase[off + AlphaChannel] = alphaValue;
                                    }
                                }
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int32_t row = 0; row < cell_height; ++row) {
                        if (row >= validStartY && row < validEndY) {
                            T*            rowBase    = (T*)(dst0 + row * dstStride);
                            const int32_t dataOffset = row * cell_width;

                            if (clip) {
#pragma unroll
                                for (int32_t i = 0; i < cell_width; ++i) {
                                    if (i < validEndX) {
                                        const size_t off   = (size_t)i * NChannels;
                                        rowBase[off + Ch0] = (T)fclampf<R>(ch0_data[dataOffset + i]);
                                        rowBase[off + Ch1] = (T)fclampf<R>(ch1_data[dataOffset + i]);
                                        rowBase[off + Ch2] = (T)fclampf<R>(ch2_data[dataOffset + i]);
                                    }
                                }
                            } else {
#pragma unroll
                                for (int32_t i = 0; i < cell_width; ++i) {
                                    if (i < validEndX) {
                                        const size_t off   = (size_t)i * NChannels;
                                        rowBase[off + Ch0] = (T)ch0_data[dataOffset + i];
                                        rowBase[off + Ch1] = (T)ch1_data[dataOffset + i];
                                        rowBase[off + Ch2] = (T)ch2_data[dataOffset + i];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // NChoose3 formats

            template<>
            struct format_backend<packing_format::rgb24> {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, const int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_load_Nchoose3<uint8_t, 0, 1, 2, 3>(src, stride, cell, nCellX, nCellY, imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_save_Nchoose3<uint8_t, bit_depth::bpp_8u, 0, 1, 2, -1, 3>(
                        dst, stride, clip, alpha, cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };

            template<>
            struct format_backend<packing_format::rgb16> {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, const int32_t nCellX,
                                                                 const int32_t nCellY, const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_load_Nchoose3<uint16_t, 0, 1, 2, 3>(src, stride, cell, nCellX, nCellY, imageWidth,
                                                             imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell,
                                                                 const int32_t nCellX, const int32_t nCellY,
                                                                 const uint32_t imageWidth,
                                                                 const uint32_t imageHeight) {
                    safe_save_Nchoose3<uint16_t, bit_depth::bpp_16u, 0, 1, 2, -1, 3>(
                        dst, stride, clip, alpha, cell, nCellX, nCellY, imageWidth, imageHeight);
                }
            };


        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_INGEST_EXGEST_NCHOOSE3_FAMILY_HPP
