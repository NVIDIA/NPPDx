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

#ifndef NPPDX_DETAIL_BACKEND_INGEST_EXGEST_V210_FORMAT_HPP
#define NPPDX_DETAIL_BACKEND_INGEST_EXGEST_V210_FORMAT_HPP

#include <cstdint>
#include "nppdx/utils.hpp"
#include "nppdx/detail/backend/formats_impl/yuv_10u_packed_family.hpp"

#ifndef NPPDX_ST_GLOBAL_16_BYTES_PTX
#    define NPPDX_ST_GLOBAL_16_BYTES_PTX 1
#endif
#ifndef NPPDX_LD_GLOBAL_16_BYTES_PTX
#    define NPPDX_LD_GLOBAL_16_BYTES_PTX NPPDX_ST_GLOBAL_16_BYTES_PTX
#endif

namespace nppdx {
    namespace detail {
        namespace backend {

            using cell_6x1_3ch_float       = NPPCellData<CellConfig<UInt2D<6, 1>, 3>, float>;
            using cell_6x1_3ch_const_float = NPPCellData<CellConfig<UInt2D<6, 1>, 3>, const float>;
            using cell_6x2_3ch_float       = NPPCellData<CellConfig<UInt2D<6, 2>, 3>, float>;
            using cell_6x2_3ch_const_float = NPPCellData<CellConfig<UInt2D<6, 2>, 3>, const float>;

#if defined(__CUDACC__) && defined(__NVCC__)
#    if NPPDX_ST_GLOBAL_16_BYTES_PTX
            // Four st.global.cs.u32 (16 bytes). dst must be 4-byte aligned; w* are pre-packed.
            __forceinline__ __device__ void nppdx_st_global_16_bytes_ptx(uint8_t* dst, uint32_t w0, uint32_t w1,
                                                                         uint32_t w2, uint32_t w3) {
                uint8_t* const p4  = dst + 4;
                uint8_t* const p8  = dst + 8;
                uint8_t* const p12 = dst + 12;
                asm volatile("st.global.cs.u32 [%0], %1;\n\t"
                             "st.global.cs.u32 [%2], %3;\n\t"
                             "st.global.cs.u32 [%4], %5;\n\t"
                             "st.global.cs.u32 [%6], %7;"
                             :
                             : "l"(dst), "r"(w0), "l"(p4), "r"(w1), "l"(p8), "r"(w2), "l"(p12), "r"(w3)
                             : "memory");
            }
#    endif
#    if NPPDX_LD_GLOBAL_16_BYTES_PTX
            // Four ld.global.cs.u32 (16 bytes). src must be 4-byte aligned; w* are filled little-endian.
            __forceinline__ __device__ void nppdx_ld_global_16_bytes_ptx(const uint8_t* src, uint32_t& w0, uint32_t& w1,
                                                                         uint32_t& w2, uint32_t& w3) {
                const uint8_t* const p4  = src + 4;
                const uint8_t* const p8  = src + 8;
                const uint8_t* const p12 = src + 12;
                asm volatile("ld.global.cs.u32 %0, [%4];\n\t"
                             "ld.global.cs.u32 %1, [%5];\n\t"
                             "ld.global.cs.u32 %2, [%6];\n\t"
                             "ld.global.cs.u32 %3, [%7];"
                             : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
                             : "l"(src), "l"(p4), "l"(p8), "l"(p12)
                             : "memory");
            }
#    endif
#endif

#pragma pack(push, 1)
            // V210 macropixel: 16 bytes / 6 pixels. w[0] matches U01|Y0|V01; w[1..3] rotate Cb-Y-Cr slots.
            struct V210Macropixel {
                uint32_t w[4];
            };
#pragma pack(pop)

            // Slow-path fetch relative to a v210 block16 base (16 bytes / 6 pixels).
            // block16_offset = byte offset from row_ptr to that block16; pair-slot offset is inside the helper.
            __forceinline__ __device__ void v210_fetch_phase0_w0w1(const uint8_t* row_ptr, size_t block16_offset,
                                                                   uint32_t alignment_4, uint32_t& w0, uint32_t& w1) {
                const uint8_t* p = row_ptr + block16_offset;
                switch (alignment_4) {
                    case 0:
                        w0 = *reinterpret_cast<const uint32_t*>(p + 0);
                        w1 = *reinterpret_cast<const uint32_t*>(p + 4);
                        break;
                    case 2: {
                        const uint16_t lo_0 = *reinterpret_cast<const uint16_t*>(p + 0);
                        const uint16_t hi_0 = *reinterpret_cast<const uint16_t*>(p + 2);
                        const uint16_t lo_1 = *reinterpret_cast<const uint16_t*>(p + 4);
                        const uint16_t hi_1 = *reinterpret_cast<const uint16_t*>(p + 6);
                        w0                  = static_cast<uint32_t>(lo_0) | (static_cast<uint32_t>(hi_0) << 16);
                        w1                  = static_cast<uint32_t>(lo_1) | (static_cast<uint32_t>(hi_1) << 16);
                        break;
                    }
                    default: {
                        w0 = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                             (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
                        w1 = static_cast<uint32_t>(p[4]) | (static_cast<uint32_t>(p[5]) << 8) |
                             (static_cast<uint32_t>(p[6]) << 16) | (static_cast<uint32_t>(p[7]) << 24);
                        break;
                    }
                }
            }

            __forceinline__ __device__ void v210_fetch_phase2_w1w2(const uint8_t* row_ptr, size_t block16_offset,
                                                                   uint32_t alignment_4, uint32_t& w1, uint32_t& w2) {
                const uint8_t* p = row_ptr + block16_offset + 4u;
                switch (alignment_4) {
                    case 0:
                        w1 = *reinterpret_cast<const uint32_t*>(p + 0);
                        w2 = *reinterpret_cast<const uint32_t*>(p + 4);
                        break;
                    case 2: {
                        const uint16_t lo_0 = *reinterpret_cast<const uint16_t*>(p + 0);
                        const uint16_t hi_0 = *reinterpret_cast<const uint16_t*>(p + 2);
                        const uint16_t lo_1 = *reinterpret_cast<const uint16_t*>(p + 4);
                        const uint16_t hi_1 = *reinterpret_cast<const uint16_t*>(p + 6);
                        w1                  = static_cast<uint32_t>(lo_0) | (static_cast<uint32_t>(hi_0) << 16);
                        w2                  = static_cast<uint32_t>(lo_1) | (static_cast<uint32_t>(hi_1) << 16);
                        break;
                    }
                    default: {
                        w1 = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                             (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
                        w2 = static_cast<uint32_t>(p[4]) | (static_cast<uint32_t>(p[5]) << 8) |
                             (static_cast<uint32_t>(p[6]) << 16) | (static_cast<uint32_t>(p[7]) << 24);
                        break;
                    }
                }
            }

            __forceinline__ __device__ void v210_fetch_phase4_w2w3(const uint8_t* row_ptr, size_t block16_offset,
                                                                   uint32_t alignment_4, uint32_t& w2, uint32_t& w3) {
                const uint8_t* p = row_ptr + block16_offset + 8u;
                switch (alignment_4) {
                    case 0:
                        w2 = *reinterpret_cast<const uint32_t*>(p + 0);
                        w3 = *reinterpret_cast<const uint32_t*>(p + 4);
                        break;
                    case 2: {
                        const uint16_t lo_0 = *reinterpret_cast<const uint16_t*>(p + 0);
                        const uint16_t hi_0 = *reinterpret_cast<const uint16_t*>(p + 2);
                        const uint16_t lo_1 = *reinterpret_cast<const uint16_t*>(p + 4);
                        const uint16_t hi_1 = *reinterpret_cast<const uint16_t*>(p + 6);
                        w2                  = static_cast<uint32_t>(lo_0) | (static_cast<uint32_t>(hi_0) << 16);
                        w3                  = static_cast<uint32_t>(lo_1) | (static_cast<uint32_t>(hi_1) << 16);
                        break;
                    }
                    default: {
                        w2 = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                             (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
                        w3 = static_cast<uint32_t>(p[4]) | (static_cast<uint32_t>(p[5]) << 8) |
                             (static_cast<uint32_t>(p[6]) << 16) | (static_cast<uint32_t>(p[7]) << 24);
                        break;
                    }
                }
            }

            namespace detail {

                __forceinline__ __device__ void rmw32(uint32_t* addr, uint32_t mask, uint32_t value) {
                    uint32_t old = *addr;
                    for (;;) {
                        const uint32_t expected = old;
                        const uint32_t desired  = (expected & ~mask) | (value & mask);
                        const uint32_t actual   = atomicCAS(addr, expected, desired);
                        if (actual == expected) {
                            return;
                        }
                        old = actual;
                    }
                }

                __forceinline__ __device__ void rmw16(uint16_t* addr, uint16_t mask, uint16_t value) {
                    uint16_t old = *addr;
                    for (;;) {
                        const uint16_t expected = old;
                        const uint16_t desired  = (expected & ~mask) | (value & mask);
                        const uint16_t actual   = atomicCAS(addr, expected, desired);
                        if (actual == expected) {
                            return;
                        }
                        old = actual;
                    }
                }

                // No u8 atomicCAS — one byte via halfword CAS on the 2-aligned container.
                __forceinline__ __device__ void rmw8(uint8_t* addr, uint8_t mask, uint8_t value) {
                    const uintptr_t byte_addr = reinterpret_cast<uintptr_t>(addr);
                    uint16_t* const half_ptr  = reinterpret_cast<uint16_t*>(byte_addr & ~uintptr_t(1));
                    const unsigned  byte_off  = static_cast<unsigned>(byte_addr & 1u);
                    const uint16_t  mask16    = static_cast<uint16_t>(mask) << (8u * byte_off);
                    const uint16_t  value16   = static_cast<uint16_t>(value & mask) << (8u * byte_off);
                    rmw16(half_ptr, mask16, value16);
                }

            } // namespace detail

            // Masked RMW of two adjacent uint32_t at p+0 and p+4 (p must be 4-byte aligned).
            // Snapshot both words, issue both atomicCAS, return if both succeed; otherwise
            // detail::rmw32 only on word(s) that lost the race. Not a single 64-bit atomic —
            // masks may differ per word.
            __forceinline__ __device__ void rmw_uint2(uint8_t* p, uint32_t w0, uint32_t m0, uint32_t w1, uint32_t m1) {
                uint32_t* const addr0 = reinterpret_cast<uint32_t*>(p + 0);
                uint32_t* const addr1 = reinterpret_cast<uint32_t*>(p + 4);
                const uint32_t  old0  = *addr0;
                const uint32_t  old1  = *addr1;
                const uint32_t  desired0 = old0 ^ ((old0 ^ w0) & m0);
                const uint32_t  desired1 = old1 ^ ((old1 ^ w1) & m1);
                const uint32_t  actual0  = atomicCAS(addr0, old0, desired0);
                const uint32_t  actual1  = atomicCAS(addr1, old1, desired1);
                if (actual0 == old0 && actual1 == old1) {
                    return;
                }
                if (actual0 != old0) {
                    detail::rmw32(addr0, m0, w0);
                }
                if (actual1 != old1) {
                    detail::rmw32(addr1, m1, w1);
                }
            }

            // block16_offset = byte offset from row_ptr to that block16; pair-slot offset is inside the helper.
            __forceinline__ __device__ void v210_store_phase0_macropixel(uint8_t* row_ptr, size_t block16_offset,
                                                                         uint32_t alignment_4, float y0, float u01,
                                                                         float v01, float y1) {
                uint8_t*       p  = row_ptr + block16_offset;
                const uint32_t w0 = (uint32_t(u01) & 0x3FF) | ((uint32_t(y0) & 0x3FF) << 10) |
                                    ((uint32_t(v01) & 0x3FF) << 20); //0x3FFFFFFF
                const uint32_t w1 = ((uint32_t(y1) & 0x3FF));        //0x000003FF
                switch (alignment_4) {
                    case 0: {
                        rmw_uint2(p, w0, 0x3FFFFFFFu, w1, 0x000003FFu);
                    } break;

                    case 2: {
                        const uint16_t s0 = uint16_t(w0 & 0xFFFF); //lo
                        const uint16_t s1 = uint16_t(w0 >> 16);    //hi
                        const uint16_t s2 = uint16_t(w1 & 0x3FF);  //0x000003FF
                        detail::rmw16(reinterpret_cast<uint16_t*>(p + 0), 0xFFFF, s0);
                        detail::rmw16(reinterpret_cast<uint16_t*>(p + 2), 0x3FFF, s1);
                        detail::rmw16(reinterpret_cast<uint16_t*>(p + 4), 0x03FF, s2);
                    } break;

                    default: {
                        const uint8_t b0 = uint8_t(w0 & 0xFF);
                        const uint8_t b1 = uint8_t((w0 >> 8) & 0xFF);
                        const uint8_t b2 = uint8_t((w0 >> 16) & 0xFF);
                        const uint8_t b3 = uint8_t((w0 >> 24) & 0xFF);
                        const uint8_t b4 = uint8_t(w1 & 0xFF);
                        const uint8_t b5 = uint8_t((w1 >> 8) & 0xFF);
                        detail::rmw8(p + 0, 0xFF, b0);
                        detail::rmw8(p + 1, 0xFF, b1);
                        detail::rmw8(p + 2, 0xFF, b2);
                        detail::rmw8(p + 3, 0x3F, b3);
                        detail::rmw8(p + 4, 0xFF, b4);
                        detail::rmw8(p + 5, 0x03, b5);
                    } break;
                } //switch (alignment_4)
            } //v210_store_phase0_macropixel

            __forceinline__ __device__ void v210_store_phase2_macropixel(uint8_t* row_ptr, size_t block16_offset,
                                                                         uint32_t alignment_4, float y2, float u23,
                                                                         float v23, float y3) {
                uint8_t*       p  = row_ptr + block16_offset;
                const uint32_t w1 = ((uint32_t(u23) & 0x3FF) << 10) | ((uint32_t(y2) & 0x3FF) << 20); //0x3FFFFC00
                const uint32_t w2 = ((uint32_t(y3) & 0x3FF) << 10) | ((uint32_t(v23) & 0x3FF));       //0x000FFFFF
                switch (alignment_4) {
                    case 0: {
                        rmw_uint2(p + 4, w1, 0x3FFFFC00u, w2, 0x000FFFFFu);
                    } break;

                    case 2: {
                        const uint16_t s0 = uint16_t(w1 & 0xFC00);         //lo
                        const uint16_t s1 = uint16_t((w1 >> 16) & 0x3FFF); //hi
                        const uint16_t s2 = uint16_t(w2 & 0xFFFF);         //lo
                        const uint16_t s3 = uint16_t((w2 >> 16) & 0x000F); //hi
                        detail::rmw16(reinterpret_cast<uint16_t*>(p + 4), 0xFC00, s0);
                        detail::rmw16(reinterpret_cast<uint16_t*>(p + 6), 0x3FFF, s1);
                        detail::rmw16(reinterpret_cast<uint16_t*>(p + 8), 0xFFFF, s2);
                        detail::rmw16(reinterpret_cast<uint16_t*>(p + 10), 0x000F, s3);
                    } break;

                    default: {
                        const uint8_t b1 = uint8_t((w1 >> 8) & 0xFF);
                        const uint8_t b2 = uint8_t((w1 >> 16) & 0xFF);
                        const uint8_t b3 = uint8_t((w1 >> 24) & 0x3F);
                        const uint8_t b4 = uint8_t(w2 & 0xFF);
                        const uint8_t b5 = uint8_t((w2 >> 8) & 0xFF);
                        const uint8_t b6 = uint8_t((w2 >> 16) & 0x0F);
                        detail::rmw8(p + 5, 0xFC, b1);
                        detail::rmw8(p + 6, 0xFF, b2);
                        detail::rmw8(p + 7, 0x3F, b3);
                        detail::rmw8(p + 8, 0xFF, b4);
                        detail::rmw8(p + 9, 0xFF, b5);
                        detail::rmw8(p + 10, 0x0F, b6);
                    } break;
                } //switch (alignment_4)
            } //v210_store_phase2_macropixel

            __forceinline__ __device__ void v210_store_phase4_macropixel(uint8_t* row_ptr, size_t block16_offset,
                                                                         uint32_t alignment_4, float y4, float u45,
                                                                         float v45, float y5) {
                uint8_t*       p  = row_ptr + block16_offset;
                const uint32_t w2 = ((uint32_t(u45) & 0x3FF) << 20); //0x3FF00000
                const uint32_t w3 = (uint32_t(y4) & 0x3FF) | ((uint32_t(v45) & 0x3FF) << 10) |
                                    ((uint32_t(y5) & 0x3FF) << 20); //0x3FFFFFFF
                switch (alignment_4) {
                    case 0: {
                        rmw_uint2(p + 8, w2, 0x3FF00000u, w3, 0x3FFFFFFFu);
                    } break;

                    case 2: {
                        const uint16_t s1 = uint16_t((w2 >> 16) & 0x3FF0); //hi
                        const uint16_t s2 = uint16_t(w3 & 0xFFFF);         //lo
                        const uint16_t s3 = uint16_t((w3 >> 16) & 0x3FFF); //hi
                        detail::rmw16(reinterpret_cast<uint16_t*>(p + 10), 0x3FF0, s1);
                        detail::rmw16(reinterpret_cast<uint16_t*>(p + 12), 0xFFFF, s2);
                        detail::rmw16(reinterpret_cast<uint16_t*>(p + 14), 0x3FFF, s3);
                    } break;

                    default: {
                        const uint8_t b2 = uint8_t((w2 >> 16) & 0xF0);
                        const uint8_t b3 = uint8_t((w2 >> 24) & 0x3F);
                        const uint8_t b4 = uint8_t(w3 & 0xFF);
                        const uint8_t b5 = uint8_t((w3 >> 8) & 0xFF);
                        const uint8_t b6 = uint8_t((w3 >> 16) & 0xFF);
                        const uint8_t b7 = uint8_t((w3 >> 24) & 0x3F);
                        detail::rmw8(p + 10, 0xF0, b2);
                        detail::rmw8(p + 11, 0xFF, b3);
                        detail::rmw8(p + 12, 0xFF, b4);
                        detail::rmw8(p + 13, 0xFF, b5);
                        detail::rmw8(p + 14, 0xFF, b6);
                        detail::rmw8(p + 15, 0x3F, b7);
                    } break;
                } //switch (alignment_4)
            } //v210_store_phase4_macropixel

            // One macropixel row (6 px): 16-byte burst load/store, shared by cell<6,1> and cell<6,2>.
            __forceinline__ __device__ void fast_load_v210_row6(float* ch0, float* ch1, float* ch2,
                                                                const uint8_t* row_ptr, int32_t n_cell_x) {
                const uint8_t* const mp = row_ptr + static_cast<size_t>(n_cell_x) * 16u;
                uint32_t             w0, w1, w2, w3;
#if NPPDX_LD_GLOBAL_16_BYTES_PTX && defined(__CUDACC__) && defined(__NVCC__)
                if ((reinterpret_cast<uintptr_t>(mp) & 3u) == 0u) {
                    nppdx_ld_global_16_bytes_ptx(mp, w0, w1, w2, w3);
                } else
#endif
                {
                    const uint64_t d0 = load_uint64_unaligned(mp);
                    const uint64_t d1 = load_uint64_unaligned(mp + 8);
                    w0                = static_cast<uint32_t>(d0);
                    w1                = static_cast<uint32_t>(d0 >> 32);
                    w2                = static_cast<uint32_t>(d1);
                    w3                = static_cast<uint32_t>(d1 >> 32);
                }
                const uint32_t u01 = w0 & 0x3FFu;
                const uint32_t v01 = (w0 >> 20) & 0x3FFu;
                const uint32_t u23 = (w1 >> 10) & 0x3FFu;
                const uint32_t v23 = w2 & 0x3FFu;
                const uint32_t u45 = (w2 >> 20) & 0x3FFu;
                const uint32_t v45 = (w3 >> 10) & 0x3FFu;
                ch0[0]             = float((w0 >> 10) & 0x3FFu);
                ch0[1]             = float(w1 & 0x3FFu);
                ch0[2]             = float((w1 >> 20) & 0x3FFu);
                ch0[3]             = float((w2 >> 10) & 0x3FFu);
                ch0[4]             = float(w3 & 0x3FFu);
                ch0[5]             = float((w3 >> 20) & 0x3FFu);
                ch1[0] = ch1[1] = float(u01);
                ch1[2] = ch1[3] = float(u23);
                ch1[4] = ch1[5] = float(u45);
                ch2[0] = ch2[1] = float(v01);
                ch2[2] = ch2[3] = float(v23);
                ch2[4] = ch2[5] = float(v45);
            }

            __forceinline__ __device__ void fast_save_v210_row6(const float* ch0, const float* ch1, const float* ch2,
                                                                uint8_t* row_ptr, int32_t n_cell_x, bool clip) {
                V210Macropixel mp {};
                if (clip) {
                    const uint32_t u01 = uint32_t(fclampf<bit_depth::bpp_10u>((ch1[0] + ch1[1]) * 0.5f));
                    const uint32_t v01 = uint32_t(fclampf<bit_depth::bpp_10u>((ch2[0] + ch2[1]) * 0.5f));
                    const uint32_t y0  = uint32_t(fclampf<bit_depth::bpp_10u>(ch0[0]));
                    const uint32_t y1  = uint32_t(fclampf<bit_depth::bpp_10u>(ch0[1]));
                    const uint32_t u23 = uint32_t(fclampf<bit_depth::bpp_10u>((ch1[2] + ch1[3]) * 0.5f));
                    const uint32_t v23 = uint32_t(fclampf<bit_depth::bpp_10u>((ch2[2] + ch2[3]) * 0.5f));
                    const uint32_t y2  = uint32_t(fclampf<bit_depth::bpp_10u>(ch0[2]));
                    const uint32_t y3  = uint32_t(fclampf<bit_depth::bpp_10u>(ch0[3]));
                    const uint32_t u45 = uint32_t(fclampf<bit_depth::bpp_10u>((ch1[4] + ch1[5]) * 0.5f));
                    const uint32_t v45 = uint32_t(fclampf<bit_depth::bpp_10u>((ch2[4] + ch2[5]) * 0.5f));
                    const uint32_t y4  = uint32_t(fclampf<bit_depth::bpp_10u>(ch0[4]));
                    const uint32_t y5  = uint32_t(fclampf<bit_depth::bpp_10u>(ch0[5]));
                    mp.w[0]            = u01 | (y0 << 10) | (v01 << 20);
                    mp.w[1]            = y1 | (u23 << 10) | (y2 << 20);
                    mp.w[2]            = v23 | (y3 << 10) | (u45 << 20);
                    mp.w[3]            = y4 | (v45 << 10) | (y5 << 20);
                } else {
                    const uint32_t u01 = uint32_t((ch1[0] + ch1[1]) * 0.5f) & 0x3FFu;
                    const uint32_t v01 = uint32_t((ch2[0] + ch2[1]) * 0.5f) & 0x3FFu;
                    const uint32_t y0  = uint32_t(ch0[0]) & 0x3FFu;
                    const uint32_t y1  = uint32_t(ch0[1]) & 0x3FFu;
                    const uint32_t u23 = uint32_t((ch1[2] + ch1[3]) * 0.5f) & 0x3FFu;
                    const uint32_t v23 = uint32_t((ch2[2] + ch2[3]) * 0.5f) & 0x3FFu;
                    const uint32_t y2  = uint32_t(ch0[2]) & 0x3FFu;
                    const uint32_t y3  = uint32_t(ch0[3]) & 0x3FFu;
                    const uint32_t u45 = uint32_t((ch1[4] + ch1[5]) * 0.5f) & 0x3FFu;
                    const uint32_t v45 = uint32_t((ch2[4] + ch2[5]) * 0.5f) & 0x3FFu;
                    const uint32_t y4  = uint32_t(ch0[4]) & 0x3FFu;
                    const uint32_t y5  = uint32_t(ch0[5]) & 0x3FFu;
                    mp.w[0]            = u01 | (y0 << 10) | (v01 << 20);
                    mp.w[1]            = y1 | (u23 << 10) | (y2 << 20);
                    mp.w[2]            = v23 | (y3 << 10) | (u45 << 20);
                    mp.w[3]            = y4 | (v45 << 10) | (y5 << 20);
                }
                uint8_t* const dst = row_ptr + static_cast<size_t>(n_cell_x) * 16u;
#if NPPDX_ST_GLOBAL_16_BYTES_PTX && defined(__CUDACC__) && defined(__NVCC__)
                if ((reinterpret_cast<uintptr_t>(dst) & 3u) == 0u) {
                    nppdx_st_global_16_bytes_ptx(dst, mp.w[0], mp.w[1], mp.w[2], mp.w[3]);
                } else
#endif
                    if ((reinterpret_cast<uintptr_t>(dst) & 15u) == 0u) {
                    *reinterpret_cast<V210Macropixel*>(dst) = mp;
                } else {
                    uint32_t* wp = reinterpret_cast<uint32_t*>(dst);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        wp[i] = mp.w[i];
                    }
                }
            }

            // Interior ingest: align-4 src/stride, two uint32 loads per 4:2:2 pair (phase 0/2/4 → +0/+4/+8).
            template<typename CellDataType>
            __forceinline__ __device__ void fast_load_v210(const uint8_t** src, const size_t* stride,
                                                           CellDataType& cell, int32_t n_cell_x, int32_t n_cell_y) {
                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;

                const int32_t  base_pixel_x = n_cell_x * cell_width;
                const int32_t  base_pixel_y = n_cell_y * cell_height;
                float*         ch0          = cell.channel_ptr(0);
                float*         ch1          = cell.channel_ptr(1);
                float*         ch2          = cell.channel_ptr(2);
                const uint8_t* plane        = src[0];
                const size_t   row_stride   = stride[0];
                const size_t   block16_off0 = static_cast<size_t>(base_pixel_x / 6) * 16u;
                const int      start_phase  = base_pixel_x % 6;

#pragma unroll
                for (int row = 0; row < cell_height; ++row) {
                    const uint8_t* row_ptr     = plane + static_cast<size_t>(base_pixel_y + row) * row_stride;
                    const int32_t  data_off    = row * cell_width;
                    int            phase       = start_phase;
                    size_t         block16_off = block16_off0;

#pragma unroll
                    for (int col = 0; col < cell_width; col += 2) {
                        const uint8_t* p = row_ptr + block16_off;

                        const uint32_t w_lo = *reinterpret_cast<const uint32_t*>(p + phase * 2);
                        const uint32_t w_hi = *reinterpret_cast<const uint32_t*>(p + phase * 2 + 4);

                        uint32_t slots[6]; //skipping the float conversion and the 0x3FF mask
                        slots[0] = w_lo;
                        slots[1] = w_lo >> 10;
                        slots[2] = w_lo >> 20;
                        slots[3] = w_hi;
                        slots[4] = w_hi >> 10;
                        slots[5] = w_hi >> 20;

                        const uint32_t pi = phase >> 1;
                        const float    u  = float(slots[0 + pi] & 0x3FF);
                        const float    y0 = float(slots[1 + pi] & 0x3FF);
                        const float    v  = float(slots[2 + pi] & 0x3FF);
                        const float    y1 = float(slots[3 + pi] & 0x3FF);

                        ch0[data_off + col]     = y0;
                        ch1[data_off + col]     = u;
                        ch2[data_off + col]     = v;
                        ch0[data_off + col + 1] = y1;
                        ch1[data_off + col + 1] = u;
                        ch2[data_off + col + 1] = v;

                        phase = (phase == 4) ? 0 : (phase + 2);
                        block16_off += 16u * (phase == 0);
                    }
                }
            }

            // Interior exgest: align-4 dst/stride, two uint32 RMW per 4:2:2 pair (phase → offset + masks).
            template<typename CellDataType>
            __forceinline__ __device__ void fast_save_v210(uint8_t** dst, const size_t* stride, bool clip,
                                                           const CellDataType& cell, int32_t n_cell_x,
                                                           int32_t n_cell_y) {
                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;

                const int32_t base_pixel_x = n_cell_x * cell_width;
                const int32_t base_pixel_y = n_cell_y * cell_height;
                const float*  ch0          = cell.channel_ptr(0);
                const float*  ch1          = cell.channel_ptr(1);
                const float*  ch2          = cell.channel_ptr(2);
                uint8_t*      plane        = dst[0];
                const size_t  row_stride   = stride[0];
                const size_t  block16_off0 = static_cast<size_t>(base_pixel_x / 6) * 16u;
                const int     start_phase  = base_pixel_x % 6;

#pragma unroll
                for (int row = 0; row < cell_height; ++row) {
                    uint8_t*      row_ptr     = plane + static_cast<size_t>(base_pixel_y + row) * row_stride;
                    const int32_t data_off    = row * cell_width;
                    uint32_t      m2          = 0u - (start_phase == 2);
                    uint32_t      m4          = 0u - (start_phase == 4);
                    size_t        block16_off = block16_off0;

#pragma unroll
                    for (int col = 0; col < cell_width; col += 2) {
                        uint32_t u_pair, v_pair, y0, y1;
                        if (clip) {
                            // fclampf<bpp_10u> ∈ [0,1023]; uint32_t cast is exact 10-bit (no & 0x3FF).
                            u_pair = uint32_t(
                                fclampf<bit_depth::bpp_10u>((ch1[data_off + col] + ch1[data_off + col + 1]) * 0.5f));
                            v_pair = uint32_t(
                                fclampf<bit_depth::bpp_10u>((ch2[data_off + col] + ch2[data_off + col + 1]) * 0.5f));
                            y0 = uint32_t(fclampf<bit_depth::bpp_10u>(ch0[data_off + col]));
                            y1 = uint32_t(fclampf<bit_depth::bpp_10u>(ch0[data_off + col + 1]));
                        } else {
                            u_pair = uint32_t((ch1[data_off + col] + ch1[data_off + col + 1]) * 0.5f) & 0x3FFu;
                            v_pair = uint32_t((ch2[data_off + col] + ch2[data_off + col + 1]) * 0.5f) & 0x3FFu;
                            y0     = uint32_t(ch0[data_off + col]) & 0x3FFu;
                            y1     = uint32_t(ch0[data_off + col + 1]) & 0x3FFu;
                        }

                        const uint32_t pi = (0u - m2) + 2u * (0u - m4);
                        uint32_t       slots[6];
                        slots[4u - ((m2 | m4) & 4u)]      = 0u;
                        slots[5u - ((0u - m4) & 1u) * 4u] = 0u;
                        slots[pi + 0]                     = u_pair;
                        slots[pi + 1]                     = y0;
                        slots[pi + 2]                     = v_pair;
                        slots[pi + 3]                     = y1;
                        const uint32_t val_a = slots[0] | (slots[1] << 10) | (slots[2] << 20);
                        const uint32_t val_b = slots[3] | (slots[4] << 10) | (slots[5] << 20);
                        const uint32_t mask_a =
                            0x3FFFFFFFu ^ (m2 & (0x3FFFFFFFu ^ 0x3FFFFC00u)) ^ (m4 & (0x3FFFFFFFu ^ 0x3FF00000u));
                        const uint32_t mask_b =
                            0x000003FFu ^ (m2 & (0x000003FFu ^ 0x000FFFFFu)) ^ (m4 & (0x000003FFu ^ 0x3FFFFFFFu));
                        rmw_uint2(row_ptr + block16_off + (pi << 2), val_a, mask_a, val_b, mask_b);
                        block16_off += 16u & m4;
                        const uint32_t t = ~(m2 | m4);
                        m4               = m2;
                        m2               = t;
                    }
                }
            }

            template<>
            __forceinline__ __device__ void fast_load_v210<cell_6x1_3ch_float>(const uint8_t**     src,
                                                                               const size_t*       stride,
                                                                               cell_6x1_3ch_float& cell,
                                                                               int32_t n_cell_x, int32_t n_cell_y) {
                fast_load_v210_row6(cell.channel_ptr(0), cell.channel_ptr(1), cell.channel_ptr(2),
                                    src[0] + static_cast<size_t>(n_cell_y) * stride[0], n_cell_x);
            }

            template<>
            __forceinline__ __device__ void fast_save_v210<cell_6x1_3ch_float>(uint8_t** dst, const size_t* stride,
                                                                               bool                      clip,
                                                                               const cell_6x1_3ch_float& cell,
                                                                               int32_t n_cell_x, int32_t n_cell_y) {
                fast_save_v210_row6(cell.channel_ptr(0), cell.channel_ptr(1), cell.channel_ptr(2),
                                    dst[0] + static_cast<size_t>(n_cell_y) * stride[0], n_cell_x, clip);
            }

            template<>
            __forceinline__ __device__ void fast_save_v210<cell_6x1_3ch_const_float>(
                uint8_t** dst, const size_t* stride, bool clip, const cell_6x1_3ch_const_float& cell, int32_t n_cell_x,
                int32_t n_cell_y) {
                fast_save_v210<cell_6x1_3ch_float>(dst, stride, clip, reinterpret_cast<const cell_6x1_3ch_float&>(cell),
                                                   n_cell_x, n_cell_y);
            }

            template<>
            __forceinline__ __device__ void fast_load_v210<cell_6x2_3ch_float>(const uint8_t**     src,
                                                                               const size_t*       stride,
                                                                               cell_6x2_3ch_float& cell,
                                                                               int32_t n_cell_x, int32_t n_cell_y) {
                fast_load_v210_row6(cell.channel_ptr(0), cell.channel_ptr(1), cell.channel_ptr(2),
                                    src[0] + static_cast<size_t>(n_cell_y * 2) * stride[0], n_cell_x);
                fast_load_v210_row6(cell.channel_ptr(0) + 6, cell.channel_ptr(1) + 6, cell.channel_ptr(2) + 6,
                                    src[0] + static_cast<size_t>(n_cell_y * 2 + 1) * stride[0], n_cell_x);
            }

            template<>
            __forceinline__ __device__ void fast_save_v210<cell_6x2_3ch_float>(uint8_t** dst, const size_t* stride,
                                                                               bool                      clip,
                                                                               const cell_6x2_3ch_float& cell,
                                                                               int32_t n_cell_x, int32_t n_cell_y) {
                fast_save_v210_row6(cell.channel_ptr(0), cell.channel_ptr(1), cell.channel_ptr(2),
                                    dst[0] + static_cast<size_t>(n_cell_y * 2) * stride[0], n_cell_x, clip);
                fast_save_v210_row6(cell.channel_ptr(0) + 6, cell.channel_ptr(1) + 6, cell.channel_ptr(2) + 6,
                                    dst[0] + static_cast<size_t>(n_cell_y * 2 + 1) * stride[0], n_cell_x, clip);
            }

            template<>
            __forceinline__ __device__ void fast_save_v210<cell_6x2_3ch_const_float>(
                uint8_t** dst, const size_t* stride, bool clip, const cell_6x2_3ch_const_float& cell, int32_t n_cell_x,
                int32_t n_cell_y) {
                fast_save_v210<cell_6x2_3ch_float>(dst, stride, clip, reinterpret_cast<const cell_6x2_3ch_float&>(cell),
                                                   n_cell_x, n_cell_y);
            }

            //clean-sheet v210 ingest.  v210 is packed 16 bytes per 6 pixels.
            template<typename CellDataType>
            __forceinline__ __device__ void safe_load_v210(const uint8_t** src, const size_t* stride,
                                                           CellDataType& cell, int32_t n_cell_x, int32_t n_cell_y,
                                                           int32_t image_width, int32_t image_height) {
                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;
                static_assert(cell_width % 2 == 0, "v210 ingest requires even cell_width");

                const int32_t base_pixel_x = n_cell_x * cell_width;
                const int32_t base_pixel_y = n_cell_y * cell_height;

                float*         ch0        = cell.channel_ptr(0);
                float*         ch1        = cell.channel_ptr(1);
                float*         ch2        = cell.channel_ptr(2);
                const uint8_t* plane      = src[0];
                const size_t   row_stride = stride[0];

                if (image_width <= 0 || image_height <= 0) {
#pragma unroll
                    for (int32_t row = 0; row < cell_height; ++row) {
                        const int32_t data_off = row * cell_width;
#pragma unroll
                        for (int32_t i = 0; i < cell_width; ++i) {
                            ch0[data_off + i] = 0.f;
                            ch1[data_off + i] = 0.f;
                            ch2[data_off + i] = 0.f;
                        }
                    }
                    return;
                }

                if ((base_pixel_x < 0) || (image_width == 1)) {
                    for (int row = 0; row < cell_height; ++row) {
                        const int32_t y_clip =
                            static_cast<int32_t>(nppdx_min(nppdx_max(base_pixel_y + row, 0), image_height - 1));
                        const uint8_t* row_ptr     = plane + static_cast<size_t>(y_clip) * row_stride;
                        const uint32_t alignment_4 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(row_ptr) % 4u);
                        uint32_t       w0 = 0, w1 = 0;
                        v210_fetch_phase0_w0w1(row_ptr, 0u, alignment_4, w0, w1);
                        const uint32_t fetch = w0;

                        const float   y_edge   = float((fetch >> 10) & 0x3FF);
                        const float   u_edge   = float((fetch >> 00) & 0x3FF);
                        const float   v_edge   = float((fetch >> 20) & 0x3FF);
                        const int32_t data_off = row * cell_width;
                        for (int x = 0; x < cell_width; ++x) {
                            ch0[data_off + x] = y_edge;
                            ch1[data_off + x] = u_edge;
                            ch2[data_off + x] = v_edge;
                        }
                    }
                    return;
                }

#if USE_FAST_PATH
                if constexpr (cell_width == 6) {
                    if (base_pixel_x >= 0 && base_pixel_x + cell_width <= image_width && base_pixel_y >= 0 &&
                        base_pixel_y + cell_height <= image_height && (reinterpret_cast<uintptr_t>(plane) % 4u) == 0u &&
                        (row_stride % 4u) == 0u) {
                        fast_load_v210<CellDataType>(src, stride, cell, n_cell_x, n_cell_y);
                        return;
                    }
                } else if (base_pixel_x >= 0 && base_pixel_x + cell_width <= image_width && base_pixel_y >= 0 &&
                           base_pixel_y + cell_height <= image_height &&
                           (reinterpret_cast<uintptr_t>(plane) % 4u) == 0u && (row_stride % 4u) == 0u) {
                    fast_load_v210<CellDataType>(src, stride, cell, n_cell_x, n_cell_y);
                    return;
                }
#endif

                {
                    const bool needs_right_edge          = (base_pixel_x + cell_width > image_width);
                    size_t     right_edge_block16_offset = 0u;
                    if (needs_right_edge) {
                        right_edge_block16_offset = static_cast<size_t>((image_width - 1) / 6) * 16u;
                    }

                    for (int row = 0; row < cell_height; ++row) {
                        const int32_t y_clip =
                            static_cast<int32_t>(nppdx_min(nppdx_max(base_pixel_y + row, 0), image_height - 1));
                        const uint8_t* row_ptr     = plane + static_cast<size_t>(y_clip) * row_stride;
                        const uint32_t alignment_4 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(row_ptr) % 4u);
                        const int32_t  data_off    = row * cell_width;

                        uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
                        float    y_edge = 0.f, u_edge = 0.f, v_edge = 0.f;

                        if (needs_right_edge) {
                            const int32_t width_mod_6 = image_width % 6;
                            switch (width_mod_6) {
                                case 1:
                                case 2: {
                                    v210_fetch_phase0_w0w1(row_ptr, right_edge_block16_offset, alignment_4, w0, w1);

                                    if (width_mod_6 == 2) {
                                        u_edge = float((w0 >> 00) & 0x3FF);
                                        v_edge = float((w0 >> 20) & 0x3FF);
                                        y_edge = float((w1 >> 00) & 0x3FF);
                                    } else {
                                        u_edge = float((w0 >> 00) & 0x3FF);
                                        v_edge = float((w0 >> 20) & 0x3FF);
                                        y_edge = float((w0 >> 10) & 0x3FF);
                                    }
                                } break;

                                case 3:
                                case 4: {
                                    if (width_mod_6 == 4) {
                                        v210_fetch_phase2_w1w2(row_ptr, right_edge_block16_offset, alignment_4, w1, w2);
                                        u_edge = float((w1 >> 10) & 0x3FF);
                                        v_edge = float((w2 >> 00) & 0x3FF);
                                        y_edge = float((w2 >> 10) & 0x3FF);
                                    } else {
                                        v210_fetch_phase2_w1w2(row_ptr, right_edge_block16_offset, alignment_4, w1, w2);
                                        u_edge = float((w1 >> 10) & 0x3FF);
                                        v_edge = float((w2 >> 00) & 0x3FF);
                                        y_edge = float((w1 >> 20) & 0x3FF);
                                    }
                                } break;

                                case 5:
                                case 0: {
                                    v210_fetch_phase4_w2w3(row_ptr, right_edge_block16_offset, alignment_4, w2, w3);

                                    if (width_mod_6 == 0) {
                                        u_edge = float((w2 >> 20) & 0x3FF);
                                        v_edge = float((w3 >> 10) & 0x3FF);
                                        y_edge = float((w3 >> 20) & 0x3FF);
                                    } else {
                                        u_edge = float((w2 >> 20) & 0x3FF);
                                        v_edge = float((w3 >> 10) & 0x3FF);
                                        y_edge = float((w3 >> 00) & 0x3FF);
                                    }
                                } break;
                            }
                        }

                        const int32_t beyond_image_width = (image_width & ~1) - 1;
                        int           phase              = base_pixel_x % 6;
                        size_t        cur_block16_offset = static_cast<size_t>((base_pixel_x / 6) * 16);
#pragma unroll
                        for (int col = 0; col < cell_width; col += 2) {
                            if (base_pixel_x + col >= beyond_image_width) {
                                ch0[data_off + col] = ch0[data_off + col + 1] = y_edge;
                                ch1[data_off + col] = ch1[data_off + col + 1] = u_edge;
                                ch2[data_off + col] = ch2[data_off + col + 1] = v_edge;
                            } else {
                                switch (phase) {
                                    case 0:
                                        v210_fetch_phase0_w0w1(row_ptr, cur_block16_offset, alignment_4, w0, w1);
                                        ch0[data_off + col] = float((w0 >> 10) & 0x3FF);
                                        ch1[data_off + col] = ch1[data_off + col + 1] = float((w0 >> 00) & 0x3FF);
                                        ch2[data_off + col] = ch2[data_off + col + 1] = float((w0 >> 20) & 0x3FF);
                                        ch0[data_off + col + 1]                       = float((w1 >> 00) & 0x3FF);
                                        phase                                         = 2;
                                        break;

                                    case 2:
                                        v210_fetch_phase2_w1w2(row_ptr, cur_block16_offset, alignment_4, w1, w2);
                                        ch0[data_off + col]     = float((w1 >> 20) & 0x3FF);
                                        ch0[data_off + col + 1] = float((w2 >> 10) & 0x3FF);
                                        ch1[data_off + col] = ch1[data_off + col + 1] = float((w1 >> 10) & 0x3FF);
                                        ch2[data_off + col] = ch2[data_off + col + 1] = float((w2 >> 00) & 0x3FF);
                                        phase                                         = 4;
                                        break;

                                    case 4:
                                        v210_fetch_phase4_w2w3(row_ptr, cur_block16_offset, alignment_4, w2, w3);
                                        ch0[data_off + col]     = float((w3 >> 00) & 0x3FF);
                                        ch0[data_off + col + 1] = float((w3 >> 20) & 0x3FF);
                                        ch1[data_off + col] = ch1[data_off + col + 1] = float((w2 >> 20) & 0x3FF);
                                        ch2[data_off + col] = ch2[data_off + col + 1] = float((w3 >> 10) & 0x3FF);
                                        phase                                         = 0;
                                        cur_block16_offset += 16;
                                        break;
                                }
                            }
                        }
                    }
                }
            } //safe_load_v210

            //clean sheet rewrite v210 exgest
            template<typename CellDataType>
            __forceinline__ __device__ void safe_save_v210(uint8_t** dst, const size_t* stride, bool clip, float alpha,
                                                           const CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                           int32_t imageWidth, int32_t imageHeight) {
                (void)alpha;
                constexpr int32_t cell_width  = CellDataType::cell_width;
                constexpr int32_t cell_height = CellDataType::cell_height;
                static_assert(cell_width % 2 == 0, "v210 exgest / 4:2:2 requires even cell_width");

                if (imageWidth <= 0 || imageHeight <= 0)
                    return;

                const float*  ch0        = cell.channel_ptr(0);
                const float*  ch1        = cell.channel_ptr(1);
                const float*  ch2        = cell.channel_ptr(2);
                const int32_t basePixelX = nCellX * cell_width;
                const int32_t basePixelY = nCellY * cell_height;

                uint8_t*     plane      = dst[0];
                const size_t row_stride = stride[0];

                //reject if cell is outside image
                if ((nCellX < 0) || (nCellY < 0) || (basePixelX >= imageWidth) || (basePixelY >= imageHeight))
                    return;

#if USE_FAST_PATH
                if constexpr (cell_width == 6) {
                    // 6-wide cell is one macropixel; imageWidthEven is for 4:2:2 pair columns only.
                    if (basePixelX + cell_width <= imageWidth && basePixelY + cell_height <= imageHeight &&
                        (reinterpret_cast<uintptr_t>(plane) % 4u) == 0u && (row_stride % 4u) == 0u) {
                        fast_save_v210<CellDataType>(dst, stride, clip, cell, nCellX, nCellY);
                        return;
                    }
                } else {
                    const int32_t imageWidthEven = imageWidth & ~1;
                    if (basePixelX + cell_width <= imageWidthEven && basePixelY + cell_height <= imageHeight &&
                        (reinterpret_cast<uintptr_t>(plane) % 4u) == 0u && (row_stride % 4u) == 0u) {
                        fast_save_v210<CellDataType>(dst, stride, clip, cell, nCellX, nCellY);
                        return;
                    }
                }
#endif

                //clip case is applied on macropixel exgest - atomic CAS and divergence limits perf
                //write macropixels that are inside image
                //handle odd pixel at end of cell
                for (int32_t row = 0; row < cell_height; ++row) {
                    if (basePixelY + row >= imageHeight)
                        return;

                    uint8_t*       row_ptr     = plane + static_cast<size_t>(basePixelY + row) * row_stride;
                    const uint32_t alignment_4 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(row_ptr) % 4u);

                    const int32_t dataOff = row * cell_width;

                    const int32_t imageWidthEven     = imageWidth & ~1;
                    int           phase              = basePixelX % 6;
                    size_t        cur_block16_offset = static_cast<size_t>((basePixelX / 6) * 16);

                    for (int col = 0; col < cell_width; col += 2) {
                        if (basePixelX + col < imageWidthEven) { //write out all of the macropixels
                            const float u_pair = (ch1[dataOff + col] + ch1[dataOff + col + 1]) * 0.5f;
                            const float v_pair = (ch2[dataOff + col] + ch2[dataOff + col + 1]) * 0.5f;
                            if (clip) {
                                switch (phase) {
                                    case 0:
                                        v210_store_phase0_macropixel(
                                            row_ptr, cur_block16_offset, alignment_4,
                                            fclampf<bit_depth::bpp_10u>(ch0[dataOff + col]),
                                            fclampf<bit_depth::bpp_10u>(u_pair), fclampf<bit_depth::bpp_10u>(v_pair),
                                            fclampf<bit_depth::bpp_10u>(ch0[dataOff + col + 1]));
                                        phase = 2;
                                        break;
                                    case 2:
                                        v210_store_phase2_macropixel(
                                            row_ptr, cur_block16_offset, alignment_4,
                                            fclampf<bit_depth::bpp_10u>(ch0[dataOff + col]),
                                            fclampf<bit_depth::bpp_10u>(u_pair), fclampf<bit_depth::bpp_10u>(v_pair),
                                            fclampf<bit_depth::bpp_10u>(ch0[dataOff + col + 1]));
                                        phase = 4;
                                        break;
                                    case 4:
                                        v210_store_phase4_macropixel(
                                            row_ptr, cur_block16_offset, alignment_4,
                                            fclampf<bit_depth::bpp_10u>(ch0[dataOff + col]),
                                            fclampf<bit_depth::bpp_10u>(u_pair), fclampf<bit_depth::bpp_10u>(v_pair),
                                            fclampf<bit_depth::bpp_10u>(ch0[dataOff + col + 1]));
                                        phase = 0;
                                        cur_block16_offset += 16;
                                        break;
                                }
                            } else {
                                switch (phase) {
                                    case 0:
                                        v210_store_phase0_macropixel(row_ptr, cur_block16_offset, alignment_4,
                                                                     ch0[dataOff + col], u_pair, v_pair,
                                                                     ch0[dataOff + col + 1]);
                                        phase = 2;
                                        break;
                                    case 2:
                                        v210_store_phase2_macropixel(row_ptr, cur_block16_offset, alignment_4,
                                                                     ch0[dataOff + col], u_pair, v_pair,
                                                                     ch0[dataOff + col + 1]);
                                        phase = 4;
                                        break;
                                    case 4:
                                        v210_store_phase4_macropixel(row_ptr, cur_block16_offset, alignment_4,
                                                                     ch0[dataOff + col], u_pair, v_pair,
                                                                     ch0[dataOff + col + 1]);
                                        phase = 0;
                                        cur_block16_offset += 16;
                                        break;
                                }
                            }
                        } else if ((imageWidthEven != imageWidth) &&
                                   (basePixelX + col ==
                                    imageWidth - 1)) { //solo odd-width pixel only (OOB cols in this branch are ignored)
                            // Y+U only (!second_in_pair); y0/y2/y4 by phase — CPU v210_merge_encode_odd_tail
                            uint32_t y_s;
                            uint32_t u_s;
                            if (clip) {
                                y_s = (uint32_t)fclampf<bit_depth::bpp_10u>(ch0[dataOff + col]);
                                u_s = (uint32_t)fclampf<bit_depth::bpp_10u>(ch1[dataOff + col]);
                            } else {
                                y_s = (uint32_t)ch0[dataOff + col];
                                u_s = (uint32_t)ch1[dataOff + col];
                            }
                            y_s &= 0x3FFu;
                            u_s &= 0x3FFu;

                            uint8_t* p = row_ptr + cur_block16_offset;
                            // Odd width: no perf target — rmw8 works for alignment 0/1/2/3 (no align×phase matrix).
                            switch (phase) {
                                case 0: {
                                    const uint32_t w0 = u_s | (y_s << 10);
                                    detail::rmw8(p + 0, 0xFFu, static_cast<uint8_t>(w0 & 0xFFu));
                                    detail::rmw8(p + 1, 0xFFu, static_cast<uint8_t>((w0 >> 8) & 0xFFu));
                                    detail::rmw8(p + 2, 0x0Fu, static_cast<uint8_t>((w0 >> 16) & 0x0Fu));
                                } break;
                                case 2: {
                                    const uint32_t w1 = (u_s << 10) | (y_s << 20);
                                    detail::rmw8(p + 5, 0xFCu, static_cast<uint8_t>((w1 >> 8) & 0xFCu));
                                    detail::rmw8(p + 6, 0xFFu, static_cast<uint8_t>((w1 >> 16) & 0xFFu));
                                    detail::rmw8(p + 7, 0x3Fu, static_cast<uint8_t>((w1 >> 24) & 0x3Fu));
                                } break;
                                default: {
                                    detail::rmw8(p + 10, 0xF0u, static_cast<uint8_t>(((u_s << 20) >> 16) & 0xF0u));
                                    detail::rmw8(p + 11, 0xFFu, static_cast<uint8_t>(((u_s << 20) >> 24) & 0xFFu));
                                    detail::rmw8(p + 12, 0xFFu, static_cast<uint8_t>(y_s & 0xFFu));
                                    detail::rmw8(p + 13, 0x03u, static_cast<uint8_t>((y_s >> 8) & 0x03u));
                                } break;
                            }
                        } //else odd tail pixel
                    } //for col
                } //for row


            } //safe_save_v210

            template<>
            struct format_backend<packing_format::v210> {
                static constexpr bool is_implemented = true;

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_load(const uint8_t** src, const size_t* stride,
                                                                 CellDataType& cell, int32_t nCellX, int32_t nCellY,
                                                                 int32_t imageWidth, int32_t imageHeight) {
                    safe_load_v210<CellDataType>(src, stride, cell, nCellX, nCellY, imageWidth, imageHeight);
                }

                template<typename CellDataType>
                __forceinline__ __device__ static void safe_save(uint8_t** dst, const size_t* stride, bool clip,
                                                                 float alpha, const CellDataType& cell, int32_t nCellX,
                                                                 int32_t nCellY, uint32_t imageWidth,
                                                                 uint32_t imageHeight) {
                    safe_save_v210<CellDataType>(dst, stride, clip, alpha, cell, nCellX, nCellY,
                                                 static_cast<int32_t>(imageWidth), static_cast<int32_t>(imageHeight));
                }
            };

        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_INGEST_EXGEST_V210_FORMAT_HPP
