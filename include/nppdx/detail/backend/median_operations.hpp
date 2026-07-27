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

#ifndef NPPDX_DETAIL_BACKEND_MEDIAN_OPERATIONS_HPP
#define NPPDX_DETAIL_BACKEND_MEDIAN_OPERATIONS_HPP


#include "nppdx/operators/function.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {
            namespace median {

                // Optimal 9-input median selection network (19 CEs, 7 layers). See
                // https://bertdobbelaere.github.io/median_networks.html#N9L19D7
                //
                // CE cost summary (half-CE = ce_min_only/ce_max_only; subtract ~0.5 per for effective cost):
                //   Per-pixel N9L19D7:     19 CEs −8 half-CEs ≈ 15 effective.
                //   IET sliding-window:     ~15 CE/pixel after 6 CE one-time, −8 half-CEs ≈ 12.5 effective.
                //
                // IET alternative (3x3 and 5x5): sort one direction, then the other, then diagonally.
                // Raw IET is 21 CEs; with extinction (final sort3 only needs median in 4) we use 20.
                // Cell version can reuse partial 1x3 sorts across pixels to justify IET.
                // Reference: IET Image Processing, 5x5 median filter,
                // https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2016.0737
                // DOI: 10.1049/iet-ipr.2016.0737
                //
                // Design wall (cell <4,2,3>): Doppelaere + 5 shared CEs → ~12 CE-equiv/pixel; IET column-share → ~12.5.
                // 6-input-share reduces to 4-input-share median-of-7: sort6 then discard the two outliers → middle 4
                // + 3 new inputs. Batcher-style first layers only partially sort; getting that effective sort4 (middle 4
                // from 6) needs ~10 CE − 2 half-CEs, and the merges to find the median of 7 cost too many CEs to win.
                // Other dead ends: float2 doesn't vectorize (coalescing covers loads); insert/remove has predicate and
                // array cost vs register networks.
                namespace {
                    template<typename T>
                    __device__ __forceinline__ void ce_full(T& a, T& b) {
                        T lo = a < b ? a : b;
                        T hi = a > b ? a : b;
                        a    = lo;
                        b    = hi;
                    }
                    template<typename T>
                    __device__ __forceinline__ void ce_min_only(T& a, T& b) {
                        a = a < b ? a : b;
                    }
                    template<typename T>
                    __device__ __forceinline__ void ce_max_only(T& a, T& b) {
                        a = a > b ? a : b;
                    }

                    __device__ __forceinline__ void ce_full(float& a, float& b) {
                        float lo = fminf(a, b);
                        float hi = fmaxf(a, b);
                        a        = lo;
                        b        = hi;
                    }
                    __device__ __forceinline__ void ce_min_only(float& a, float& b) {
                        a = fminf(a, b);
                    }
                    __device__ __forceinline__ void ce_max_only(float& a, float& b) {
                        a = fmaxf(a, b);
                    }

                    __device__ __forceinline__ void sort3(float& a, float& b, float& c) {
                        ce_full(a, b);
                        ce_full(b, c);
                        ce_full(a, b);
                    }

                    // Median of 5: 3 full CEs + 4 half (extinction). Result in v[2]. Effective 7 − 4/2 = 5 full-CE equivalent.
                    // [(0,1),(2,3)] [(0,2),(1,3)] [(2,4)] [(1,2)] [(2,4)] - (0,2),(1,3),(1,2),(2,4) last are half (only one side needed).
                    __device__ __forceinline__ void n5_median_in_place(float v[5]) {
                        ce_full(v[0], v[1]);
                        ce_full(v[2], v[3]);
                        ce_max_only(v[2], v[0]); // v[2] = max; v[0] not read again
                        ce_min_only(v[1], v[3]); // v[1] = min; v[3] not read again
                        ce_full(v[2], v[4]);
                        ce_max_only(v[2], v[1]); // v[2] = max; v[1] not read again
                        ce_min_only(v[2], v[4]); // v[2] = median (min with v[4]); v[4] not read again
                    }

                    // N5 median with (0,1) already applied (for horizontal pair sharing - shared full CE). Caller sets v[0]<=v[1]; result in v[2].
                    __device__ __forceinline__ void n5_median_in_place_after_0_1(float v[5]) {
                        ce_full(v[2], v[3]);
                        ce_max_only(v[2], v[0]);
                        ce_min_only(v[1], v[3]);
                        ce_full(v[2], v[4]);
                        ce_max_only(v[2], v[1]);
                        ce_min_only(v[2], v[4]);
                    }

                    // CE4: 4 ce_full ops on (a,b,c,d) - first two layers of 4-wire block. (a,b),(c,d) then (a,c),(b,d).
                    __device__ __forceinline__ void ce4(float& a, float& b, float& c, float& d) {
                        ce_full(a, b);
                        ce_full(c, d);
                        ce_full(a, c);
                        ce_full(b, d);
                    }

                    // N9 median after shared CEs (4,5) and (7,8) already applied (Bert Doppelaere rest). Result in v[4].
                    __device__ __forceinline__ void n9_median_after_shared_2ce(float v[9]) {
                        ce_full(v[0], v[7]); // (0,7)
                        ce_full(v[4], v[8]); // (4,8)
                        ce_full(v[0], v[2]); // (0,2)
                        ce_full(v[1], v[5]); // (1,5)
                        ce_full(v[3], v[8]); // (3,8)
                        ce_full(v[4], v[7]); // (4,7)
                        ce_max_only(v[3], v[0]);
                        ce_max_only(v[4], v[1]);
                        ce_min_only(v[2], v[8]);
                        ce_min_only(v[5], v[7]);
                        ce_full(v[3], v[4]);
                        ce_full(v[5], v[6]);
                        ce_full(v[2], v[5]);
                        ce_min_only(v[4], v[6]);
                        ce_max_only(v[3], v[2]);
                        ce_min_only(v[4], v[5]);
                        ce_max_only(v[4], v[3]);
                    }

                    // remaining14_median_after_shared_5ce - already done 5 CEs [1,2],[3,5] (center) and [0,7],[1,5],[0,2] (left/right). Do remaining 14 (same network as compute_3x3_median_per_pixel, 8 half-CEs). Result in v[4].
                    // Verified against NPP reference in IET for 3x3 (incl. 2x2 block path below).
                    __device__ __forceinline__ void remaining14_median_after_shared_5ce(float v[9]) {
                        ce_full(v[4], v[8]); // (4,8)
                        ce_full(v[3], v[8]); // (3,8)
                        ce_full(v[4], v[7]); // (4,7)
                        ce_max_only(v[3], v[0]);
                        ce_max_only(v[4], v[1]);
                        ce_min_only(v[2], v[8]);
                        ce_min_only(v[5], v[7]);
                        ce_full(v[3], v[4]);
                        ce_full(v[5], v[6]);
                        ce_full(v[2], v[5]);
                        ce_min_only(v[4], v[6]);
                        ce_max_only(v[3], v[2]);
                        ce_min_only(v[4], v[5]);
                        ce_max_only(v[4], v[3]);
                    }

                    // Complete N25 median in-place: L1/L2 on wires 20..23, then L3..L16. v[0..19] already CE4'd; v[20..24]=row0. Result in v[12].
                    __device__ __forceinline__ void n25_complete_median_in_place(float v[25]) {
                        ce_full(v[20], v[21]);
                        ce_full(v[22], v[23]);
                        ce_full(v[20], v[22]);
                        ce_full(v[21], v[23]);
                        ce_full(v[0], v[4]);
                        ce_full(v[1], v[5]);
                        ce_full(v[2], v[6]);
                        ce_full(v[3], v[7]);
                        ce_full(v[8], v[12]);
                        ce_full(v[9], v[13]);
                        ce_full(v[10], v[14]);
                        ce_full(v[11], v[15]);
                        ce_full(v[16], v[20]);
                        ce_full(v[17], v[21]);
                        ce_full(v[18], v[22]);
                        ce_full(v[19], v[23]);
                        ce_max_only(v[8], v[0]);
                        ce_full(v[1], v[9]);
                        ce_full(v[2], v[10]);
                        ce_full(v[3], v[11]);
                        ce_full(v[4], v[12]);
                        ce_full(v[5], v[13]);
                        ce_full(v[6], v[14]);
                        ce_min_only(v[7], v[15]);
                        ce_full(v[3], v[19]);
                        ce_full(v[5], v[21]);
                        ce_full(v[6], v[22]);
                        ce_min_only(v[7], v[23]);
                        ce_max_only(v[16], v[8]);
                        ce_full(v[9], v[17]);
                        ce_full(v[10], v[18]);
                        ce_full(v[12], v[20]);
                        ce_max_only(v[12], v[1]);
                        ce_max_only(v[9], v[2]);
                        ce_full(v[3], v[5]);
                        ce_max_only(v[10], v[4]);
                        ce_min_only(v[11], v[21]);
                        ce_min_only(v[13], v[22]);
                        ce_min_only(v[14], v[19]);
                        ce_full(v[18], v[20]);
                        ce_full(v[9], v[10]);
                        ce_full(v[11], v[13]);
                        ce_min_only(v[7], v[13]);
                        ce_full(v[10], v[12]);
                        ce_full(v[11], v[14]);
                        ce_full(v[5], v[11]);
                        ce_max_only(v[16], v[10]);
                        ce_full(v[12], v[18]);
                        ce_max_only(v[12], v[3]);
                        ce_max_only(v[16], v[9]);
                        ce_min_only(v[11], v[20]);
                        ce_min_only(v[14], v[18]);
                        ce_max_only(v[16], v[5]);
                        ce_full(v[6], v[12]);
                        ce_min_only(v[7], v[14]);
                        ce_full(v[11], v[17]);
                        ce_full(v[7], v[16]);
                        ce_full(v[11], v[24]);
                        ce_min_only(v[12], v[17]);
                        ce_max_only(v[11], v[7]);
                        ce_min_only(v[16], v[24]);
                        ce_max_only(v[11], v[6]);
                        ce_full(v[12], v[16]);
                        ce_max_only(v[12], v[11]);
                        ce_min_only(v[12], v[16]);
                    }

                    // Median of 21: Dobbelaere N21 66 CEs, 15 layers. Result in v[10].
                    __device__ __forceinline__ void n21_median_in_place(float v[21]) {
                        // L1
                        ce_full(v[0], v[1]);
                        ce_full(v[2], v[3]);
                        ce_full(v[4], v[5]);
                        ce_full(v[6], v[7]);
                        ce_full(v[8], v[9]);
                        ce_full(v[10], v[11]);
                        ce_full(v[12], v[13]);
                        ce_full(v[14], v[15]);
                        ce_full(v[16], v[17]);
                        ce_full(v[18], v[19]);
                        // L2
                        ce_full(v[0], v[2]);
                        ce_full(v[1], v[3]);
                        ce_full(v[4], v[6]);
                        ce_full(v[5], v[7]);
                        ce_full(v[8], v[10]);
                        ce_full(v[9], v[11]);
                        ce_full(v[12], v[14]);
                        ce_full(v[13], v[15]);
                        ce_full(v[16], v[18]);
                        ce_full(v[17], v[19]);
                        // L3
                        ce_full(v[1], v[5]);
                        ce_full(v[2], v[6]);
                        ce_full(v[3], v[15]);
                        ce_full(v[4], v[16]);
                        ce_full(v[13], v[17]);
                        ce_full(v[14], v[18]);
                        // L4
                        ce_full(v[1], v[14]);
                        ce_full(v[2], v[13]);
                        ce_full(v[3], v[7]);
                        ce_full(v[5], v[18]);
                        ce_full(v[6], v[17]);
                        ce_full(v[12], v[16]);
                        // L5
                        ce_full(v[0], v[16]);
                        ce_full(v[1], v[2]);
                        ce_full(v[3], v[19]);
                        ce_full(v[5], v[13]);
                        ce_full(v[6], v[14]);
                        ce_full(v[17], v[18]);
                        // L6
                        ce_full(v[0], v[4]);
                        ce_full(v[5], v[14]);
                        ce_full(v[6], v[10]);
                        ce_full(v[9], v[13]);
                        ce_full(v[15], v[19]);
                        // L7
                        ce_full(v[5], v[8]);
                        ce_full(v[6], v[12]);
                        ce_full(v[7], v[13]);
                        ce_full(v[11], v[14]);
                        // L8
                        ce_full(v[2], v[12]);
                        ce_full(v[7], v[17]);
                        ce_full(v[8], v[9]);
                        ce_full(v[10], v[11]);
                        // L9
                        ce_full(v[3], v[9]);
                        ce_full(v[7], v[11]);
                        ce_full(v[8], v[12]);
                        ce_full(v[10], v[16]);
                        // L10
                        ce_full(v[3], v[10]);
                        ce_full(v[4], v[12]);
                        ce_full(v[7], v[15]);
                        ce_full(v[9], v[16]);
                        // L11
                        ce_full(v[7], v[10]);
                        ce_full(v[9], v[12]);
                        // L12
                        ce_full(v[7], v[9]);
                        ce_full(v[10], v[12]);
                        // L13
                        ce_full(v[9], v[10]);
                        // L14
                        ce_full(v[10], v[20]);
                        // L15
                        ce_full(v[9], v[10]);
                    }

                } // namespace

                // Plus median by loading the full 3x3 (same addressing as compute_3x3_median_per_pixel) then N5 on cross r01,r10,r11,r12,r21.
                template<typename TileStorageType, unsigned int NumChannels, int HaloLeft, int HaloTop,
                         typename ProcessingType = float>
                __device__ void compute_5_median_plus_from_3x3_per_pixel(const TileStorageType& channels, int px,
                                                                         int py, ProcessingType* result) {
                    const int smem_x = px + HaloLeft;
                    const int smem_y = py + HaloTop;
#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        float r01  = (float)channels.channel(c)(smem_y - 1, smem_x);
                        float r10  = (float)channels.channel(c)(smem_y, smem_x - 1);
                        float r11  = (float)channels.channel(c)(smem_y, smem_x);
                        float r12  = (float)channels.channel(c)(smem_y, smem_x + 1);
                        float r21  = (float)channels.channel(c)(smem_y + 1, smem_x);
                        float v[5] = {r11, r10, r12, r01, r21};
                        n5_median_in_place(v);
                        result[c] = (ProcessingType)v[2];
                    }
                }

                // Forward declarations for dispatch (implementations below).
                template<unsigned int CellW, unsigned int CellH, typename TileStorageType, unsigned int NumChannels,
                         typename CellType, typename ProcessingType>
                __device__ void compute_5_median_plus_cell_impl(const TileStorageType&, int, int, CellType&);
                template<unsigned int CellW, unsigned int CellH, typename TileStorageType, unsigned int NumChannels,
                         typename CellType, typename ProcessingType>
                __device__ void compute_5_median_plus_cell_even_width(const TileStorageType&, int, int, CellType&);

                // Plus median cell: single entry point; daisy-chains to even-width optimized path when CellW is even.
                template<unsigned int CellW, unsigned int CellH, typename TileStorageType, unsigned int NumChannels,
                         typename CellType, typename ProcessingType = float>
                __device__ void compute_5_median_plus_cell(const TileStorageType& input_tile, int cell_smem_x,
                                                           int cell_smem_y, CellType& output_cell) {
                    if constexpr (CellW % 2 == 0) {
                        compute_5_median_plus_cell_even_width<CellW, CellH, TileStorageType, NumChannels, CellType,
                                                              ProcessingType>(input_tile, cell_smem_x, cell_smem_y,
                                                                              output_cell);
                    } else {
                        compute_5_median_plus_cell_impl<CellW, CellH, TileStorageType, NumChannels, CellType,
                                                        ProcessingType>(input_tile, cell_smem_x, cell_smem_y,
                                                                        output_cell);
                    }
                }

                // Plus median cell (per-pixel, any CellW): 3-high unroll - each iteration of cy processes 3 output rows
                // using row_0/row_1/row_2 as a cyclical buffer. Same addressing as slow path (0,0).
                template<unsigned int CellW, unsigned int CellH, typename TileStorageType, unsigned int NumChannels,
                         typename CellType, typename ProcessingType = float>
                __device__ void compute_5_median_plus_cell_impl(const TileStorageType& input_tile, int cell_smem_x,
                                                                int cell_smem_y, CellType& output_cell) {
                    constexpr int row_len = CellW + 2;
#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        float row_0[row_len], row_1[row_len], row_2[row_len];
                        // Prolog (outside cy loop): load first 3 tile rows for strip cy=0 (above=row_0, center=row_1, below=row_2)
                        const int tile_y0 = cell_smem_y - 1;
                        const int tile_y1 = cell_smem_y;
                        const int tile_y2 = cell_smem_y + 1;
#pragma unroll
                        for (int x = 0; x < row_len; ++x) {
                            const int tx = cell_smem_x - 1 + x;
                            row_0[x]     = (float)input_tile.channel(c)(tile_y0, tx);
                            row_1[x]     = (float)input_tile.channel(c)(tile_y1, tx);
                            row_2[x]     = (float)input_tile.channel(c)(tile_y2, tx);
                        }
                        // Process in batches of 3 rows; each iteration uses current row_0/row_1/row_2, then fetches one new row per output row
                        for (unsigned int cy = 0; cy < CellH; cy += 3) {
                            // Row cy: center=row_1, above=row_0, below=row_2. Plus taps (center, left, right, top, bottom) -> [cx+1], [cx], [cx+2], [cx+1], [cx+1]
#pragma unroll
                            for (unsigned int cx = 0; cx < CellW; ++cx) {
                                float v[5] = {row_1[cx + 1], row_1[cx], row_1[cx + 2], row_0[cx + 1], row_2[cx + 1]};
                                n5_median_in_place(v);
                                output_cell(c, cy * CellW + cx) = (ProcessingType)v[2];
                            }
                            if (cy + 1 >= CellH)
                                break;
                                // Fetch next row into cyclical buffer (tile row cy+2)
#pragma unroll
                            for (int x = 0; x < row_len; ++x) {
                                const int tx = cell_smem_x - 1 + x;
                                row_0[x]     = (float)input_tile.channel(c)(cell_smem_y + static_cast<int>(cy) + 2, tx);
                            }
                            // Row cy+1: center=row_2, above=row_1, below=row_0
#pragma unroll
                            for (unsigned int cx = 0; cx < CellW; ++cx) {
                                float v[5] = {row_2[cx + 1], row_2[cx], row_2[cx + 2], row_1[cx + 1], row_0[cx + 1]};
                                n5_median_in_place(v);
                                output_cell(c, (cy + 1) * CellW + cx) = (ProcessingType)v[2];
                            }
                            if (cy + 2 >= CellH)
                                break;
                                // Fetch next row (tile row cy+3)
#pragma unroll
                            for (int x = 0; x < row_len; ++x) {
                                const int tx = cell_smem_x - 1 + x;
                                row_1[x]     = (float)input_tile.channel(c)(cell_smem_y + static_cast<int>(cy) + 3, tx);
                            }
                            // Row cy+2: center=row_0, above=row_2, below=row_1
#pragma unroll
                            for (unsigned int cx = 0; cx < CellW; ++cx) {
                                float v[5] = {row_0[cx + 1], row_0[cx], row_0[cx + 2], row_2[cx + 1], row_1[cx + 1]};
                                n5_median_in_place(v);
                                output_cell(c, (cy + 2) * CellW + cx) = (ProcessingType)v[2];
                            }
                            if (cy + 3 >= CellH)
                                break;
                                // Fetch next row (tile row cy+4) for next strip
#pragma unroll
                            for (int x = 0; x < row_len; ++x) {
                                const int tx = cell_smem_x - 1 + x;
                                row_2[x]     = (float)input_tile.channel(c)(cell_smem_y + static_cast<int>(cy) + 4, tx);
                            }
                        }
                    }
                }

                // Plus median cell, even CellW only: same 3-high unroll but cx+=2 with one shared full CE per pair.
                // Layout puts overlapping pair at (0,1): v0=[center, right, left, top, bottom], v1=[left, center, right, top, bottom].
                template<unsigned int CellW, unsigned int CellH, typename TileStorageType, unsigned int NumChannels,
                         typename CellType, typename ProcessingType = float>
                __device__ void compute_5_median_plus_cell_even_width(const TileStorageType& input_tile,
                                                                      int cell_smem_x, int cell_smem_y,
                                                                      CellType& output_cell) {
                    static_assert(CellW % 2 == 0, "even-width path requires CellW even");
                    constexpr int row_len = CellW + 2;
#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        float     row_0[row_len], row_1[row_len], row_2[row_len];
                        const int tile_y0 = cell_smem_y - 1;
                        const int tile_y1 = cell_smem_y;
                        const int tile_y2 = cell_smem_y + 1;
#pragma unroll
                        for (int x = 0; x < row_len; ++x) {
                            const int tx = cell_smem_x - 1 + x;
                            row_0[x]     = (float)input_tile.channel(c)(tile_y0, tx);
                            row_1[x]     = (float)input_tile.channel(c)(tile_y1, tx);
                            row_2[x]     = (float)input_tile.channel(c)(tile_y2, tx);
                        }
                        for (unsigned int cy = 0; cy < CellH; cy += 3) {
                            // Row cy: center=row_1. Shared pair at v0[0],v0[1] and v1[0],v1[1].
#pragma unroll
                            for (unsigned int cx = 0; cx < CellW; cx += 2) {
                                float v0[5] = {row_1[cx + 1], row_1[cx + 2], row_1[cx], row_0[cx + 1],
                                               row_2[cx + 1]}; // center, right, left, top, bottom
                                float v1[5] = {row_1[cx + 1], row_1[cx + 2], row_1[cx + 3], row_0[cx + 2],
                                               row_2[cx + 2]}; // left, center, right, top, bottom
                                ce_full(v0[0], v0[1]);
                                v1[0] = v0[0];
                                v1[1] = v0[1];
                                n5_median_in_place_after_0_1(v0);
                                n5_median_in_place_after_0_1(v1);
                                output_cell(c, cy * CellW + cx)     = (ProcessingType)v0[2];
                                output_cell(c, cy * CellW + cx + 1) = (ProcessingType)v1[2];
                            }
                            if (cy + 1 >= CellH)
                                break;
#pragma unroll
                            for (int x = 0; x < row_len; ++x) {
                                const int tx = cell_smem_x - 1 + x;
                                row_0[x]     = (float)input_tile.channel(c)(cell_smem_y + static_cast<int>(cy) + 2, tx);
                            }
                            // Row cy+1: center=row_2
#pragma unroll
                            for (unsigned int cx = 0; cx < CellW; cx += 2) {
                                float v0[5] = {row_2[cx + 1], row_2[cx + 2], row_2[cx], row_1[cx + 1], row_0[cx + 1]};
                                float v1[5] = {row_2[cx + 1], row_2[cx + 2], row_2[cx + 3], row_1[cx + 2],
                                               row_0[cx + 2]};
                                ce_full(v0[0], v0[1]);
                                v1[0] = v0[0];
                                v1[1] = v0[1];
                                n5_median_in_place_after_0_1(v0);
                                n5_median_in_place_after_0_1(v1);
                                output_cell(c, (cy + 1) * CellW + cx)     = (ProcessingType)v0[2];
                                output_cell(c, (cy + 1) * CellW + cx + 1) = (ProcessingType)v1[2];
                            }
                            if (cy + 2 >= CellH)
                                break;
#pragma unroll
                            for (int x = 0; x < row_len; ++x) {
                                const int tx = cell_smem_x - 1 + x;
                                row_1[x]     = (float)input_tile.channel(c)(cell_smem_y + static_cast<int>(cy) + 3, tx);
                            }
                            // Row cy+2: center=row_0
#pragma unroll
                            for (unsigned int cx = 0; cx < CellW; cx += 2) {
                                float v0[5] = {row_0[cx + 1], row_0[cx + 2], row_0[cx], row_2[cx + 1], row_1[cx + 1]};
                                float v1[5] = {row_0[cx + 1], row_0[cx + 2], row_0[cx + 3], row_2[cx + 2],
                                               row_1[cx + 2]};
                                ce_full(v0[0], v0[1]);
                                v1[0] = v0[0];
                                v1[1] = v0[1];
                                n5_median_in_place_after_0_1(v0);
                                n5_median_in_place_after_0_1(v1);
                                output_cell(c, (cy + 2) * CellW + cx)     = (ProcessingType)v0[2];
                                output_cell(c, (cy + 2) * CellW + cx + 1) = (ProcessingType)v1[2];
                            }
                            if (cy + 3 >= CellH)
                                break;
#pragma unroll
                            for (int x = 0; x < row_len; ++x) {
                                const int tx = cell_smem_x - 1 + x;
                                row_2[x]     = (float)input_tile.channel(c)(cell_smem_y + static_cast<int>(cy) + 4, tx);
                            }
                        }
                    }
                }

                // One row of plus median: cache 3 rows (above, center, below) in registers from coalesced read, then N5 per pixel. No sharing.
                template<unsigned int CellW, typename TileStorageType, int HaloLeft, int HaloTop,
                         typename ProcessingType = float>
                __device__ void compute_5_median_plus_row(const TileStorageType& channels, unsigned int c,
                                                          int cell_smem_x, int row_smem_y, ProcessingType* result_row) {
                    constexpr int row_len       = CellW + 2; // center row + left/right halo
                    const int     tile_x_base   = cell_smem_x + HaloLeft;
                    const int     tile_y_center = row_smem_y + HaloTop;

                    float row_above[row_len], row_center[row_len], row_below[row_len];
#pragma unroll
                    for (int x = 0; x < row_len; ++x) {
                        const int tx  = tile_x_base - 1 + x;
                        row_above[x]  = (float)channels.channel(c)(tile_y_center - 1, tx);
                        row_center[x] = (float)channels.channel(c)(tile_y_center, tx);
                        row_below[x]  = (float)channels.channel(c)(tile_y_center + 1, tx);
                    }
#pragma unroll
                    for (unsigned int px = 0; px < CellW; ++px) {
                        float v[5];
                        v[0] = row_center[px + 1];
                        v[1] = row_center[px];
                        v[2] = row_center[px + 2];
                        v[3] = row_above[px + 1];
                        v[4] = row_below[px + 1];
                        n5_median_in_place(v);
                        result_row[px] = (ProcessingType)v[2];
                    }
                }

                // Per-pixel 3x3 median (radius_1_0): v[0..8] = r00..r22 row-major; median in r11.
                template<typename TileStorageType, unsigned int NumChannels, int HaloLeft, int HaloTop,
                         typename ProcessingType = float>
                __device__ void compute_3x3_median_per_pixel(const TileStorageType& channels, int px, int py,
                                                             ProcessingType* result) {
                    const int smem_x = px + HaloLeft;
                    const int smem_y = py + HaloTop;

#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        float r00 = (float)channels.channel(c)(smem_y - 1, smem_x - 1);
                        float r01 = (float)channels.channel(c)(smem_y - 1, smem_x);
                        float r02 = (float)channels.channel(c)(smem_y - 1, smem_x + 1);
                        float r10 = (float)channels.channel(c)(smem_y, smem_x - 1);
                        float r11 = (float)channels.channel(c)(smem_y, smem_x);
                        float r12 = (float)channels.channel(c)(smem_y, smem_x + 1);
                        float r20 = (float)channels.channel(c)(smem_y + 1, smem_x - 1);
                        float r21 = (float)channels.channel(c)(smem_y + 1, smem_x);
                        float r22 = (float)channels.channel(c)(smem_y + 1, smem_x + 1);

                        // Bert Doppelaere: [(0,7),(1,2),(3,5),(4,8)] [(0,2),(1,5),(3,8),(4,7)] [(0,3),(1,4),(2,8),(5,7)]
                        //                  [(3,4),(5,6)] [(2,5),(4,6)] [(2,3),(4,5)] [(3,4)]  -> median at index 4 (r11)
                        ce_full(r00, r21);
                        ce_full(r01, r02);
                        ce_full(r10, r12);
                        ce_full(r11, r22);
                        ce_full(r00, r02);
                        ce_full(r01, r12);
                        ce_full(r10, r22);
                        ce_full(r11, r21);
                        ce_max_only(r10, r00);
                        ce_max_only(r11, r01);
                        ce_min_only(r02, r22);
                        ce_min_only(r12, r21);
                        ce_full(r10, r11);
                        ce_full(r12, r20);
                        ce_full(r02, r12);
                        ce_min_only(r11, r20);
                        ce_max_only(r10, r02);
                        ce_min_only(r11, r12);
                        ce_max_only(r11, r10);
                        result[c] = (ProcessingType)r11;
                    }
                }

                // Per-pixel 5x5 median (radius r2_0): v[0..24] = r00..r44 row-major; median at v[12].
                // Dobbelaere N25 85 CEs, 16 layers. See https://bertdobbelaere.github.io/median_networks.html#N25L85D16
                template<typename TileStorageType, unsigned int NumChannels, int HaloLeft, int HaloTop,
                         typename ProcessingType = float>
                __device__ void compute_5x5_median_per_pixel(const TileStorageType& channels, int px, int py,
                                                             ProcessingType* result) {
                    const int smem_x = px + HaloLeft;
                    const int smem_y = py + HaloTop;

#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        float v[25];
                        for (int dy = -2; dy <= 2; ++dy) {
                            for (int dx = -2; dx <= 2; ++dx) {
                                v[(dy + 2) * 5 + (dx + 2)] = (float)channels.channel(c)(smem_y + dy, smem_x + dx);
                            }
                        }
                        // N25 85-CE median selection network (Dobbelaere, 16 layers). Median in v[12].
                        // Extinction: 24 CEs use ce_min_only/ce_max_only (wire never read again) -> effective 73 CEs.
                        // L1:  [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(20,21),(22,23)]
                        ce_full(v[0], v[1]);
                        ce_full(v[2], v[3]);
                        ce_full(v[4], v[5]);
                        ce_full(v[6], v[7]);
                        ce_full(v[8], v[9]);
                        ce_full(v[10], v[11]);
                        ce_full(v[12], v[13]);
                        ce_full(v[14], v[15]);
                        ce_full(v[16], v[17]);
                        ce_full(v[18], v[19]);
                        ce_full(v[20], v[21]);
                        ce_full(v[22], v[23]);
                        // L2:  [(0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,14),(13,15),(16,18),(17,19),(20,22),(21,23)]
                        ce_full(v[0], v[2]);
                        ce_full(v[1], v[3]);
                        ce_full(v[4], v[6]);
                        ce_full(v[5], v[7]);
                        ce_full(v[8], v[10]);
                        ce_full(v[9], v[11]);
                        ce_full(v[12], v[14]);
                        ce_full(v[13], v[15]);
                        ce_full(v[16], v[18]);
                        ce_full(v[17], v[19]);
                        ce_full(v[20], v[22]);
                        ce_full(v[21], v[23]);
                        // L3:  [(0,4),(1,5),(2,6),(3,7),(8,12),(9,13),(10,14),(11,15),(16,20),(17,21),(18,22),(19,23)]
                        ce_full(v[0], v[4]);
                        ce_full(v[1], v[5]);
                        ce_full(v[2], v[6]);
                        ce_full(v[3], v[7]);
                        ce_full(v[8], v[12]);
                        ce_full(v[9], v[13]);
                        ce_full(v[10], v[14]);
                        ce_full(v[11], v[15]);
                        ce_full(v[16], v[20]);
                        ce_full(v[17], v[21]);
                        ce_full(v[18], v[22]);
                        ce_full(v[19], v[23]);
                        // L4:  [(0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15)]
                        ce_max_only(v[8], v[0]); // 0 extinct
                        ce_full(v[1], v[9]);
                        ce_full(v[2], v[10]);
                        ce_full(v[3], v[11]);
                        ce_full(v[4], v[12]);
                        ce_full(v[5], v[13]);
                        ce_full(v[6], v[14]);
                        ce_min_only(v[7], v[15]); // 15 extinct
                        // L5:  [(3,19),(5,21),(6,22),(7,23),(8,16),(9,17),(10,18),(12,20)]
                        ce_full(v[3], v[19]);
                        ce_full(v[5], v[21]);
                        ce_full(v[6], v[22]);
                        ce_min_only(v[7], v[23]); // 23 extinct
                        ce_max_only(v[16], v[8]); // 8 extinct
                        ce_full(v[9], v[17]);
                        ce_full(v[10], v[18]);
                        ce_full(v[12], v[20]);
                        // L6:  [(1,12),(2,9),(3,5),(4,10),(11,21),(13,22),(14,19),(18,20)]
                        ce_max_only(v[12], v[1]); // 1 extinct
                        ce_max_only(v[9], v[2]);  // 2 extinct
                        ce_full(v[3], v[5]);
                        ce_max_only(v[10], v[4]);  // 4 extinct
                        ce_min_only(v[11], v[21]); // 21 extinct
                        ce_min_only(v[13], v[22]); // 22 extinct
                        ce_min_only(v[14], v[19]); // 19 extinct
                        ce_full(v[18], v[20]);
                        // L7:  [(9,10),(11,13)]
                        ce_full(v[9], v[10]);
                        ce_full(v[11], v[13]);
                        // L8:  [(7,13),(10,12),(11,14)]
                        ce_min_only(v[7], v[13]); // 13 extinct
                        ce_full(v[10], v[12]);
                        ce_full(v[11], v[14]);
                        // L9:  [(5,11),(10,16),(12,18)]
                        ce_full(v[5], v[11]);
                        ce_max_only(v[16], v[10]); // 10 extinct
                        ce_full(v[12], v[18]);
                        // L10: [(3,12),(9,16),(11,20),(14,18)]
                        ce_max_only(v[12], v[3]);  // 3 extinct
                        ce_max_only(v[16], v[9]);  // 9 extinct
                        ce_min_only(v[11], v[20]); // 20 extinct
                        ce_min_only(v[14], v[18]); // 18 extinct
                        // L11: [(5,16),(6,12),(7,14),(11,17)]
                        ce_max_only(v[16], v[5]); // 5 extinct
                        ce_full(v[6], v[12]);
                        ce_min_only(v[7], v[14]); // 14 extinct
                        ce_full(v[11], v[17]);
                        // L12: [(7,16),(11,24),(12,17)]
                        ce_full(v[7], v[16]);
                        ce_full(v[11], v[24]);
                        ce_min_only(v[12], v[17]); // 17 extinct
                        // L13: [(7,11),(16,24)]
                        ce_max_only(v[11], v[7]);  // 7 extinct
                        ce_min_only(v[16], v[24]); // 24 extinct
                        // L14: [(6,11),(12,16)]
                        ce_max_only(v[11], v[6]); // 6 extinct
                        ce_full(v[12], v[16]);
                        // L15: [(11,12)]
                        ce_max_only(v[12], v[11]); // 11 extinct
                        // L16: [(12,16)]
                        ce_min_only(v[12], v[16]); // 16 extinct; median in v[12]
                        result[c] = (ProcessingType)v[12];
                    }
                }

                // Per-pixel 21-element round median (radius r1_5): 5x5 minus corners; Dobbelaere N21 66 CEs, 15 layers; median at v[10].
                // 21 positions (dy,dx): (-2,-1),(-2,0),(-2,1), (-1,-2)..(-1,2), (0,-2)..(0,2), (1,-2)..(1,2), (2,-1),(2,0),(2,1).
                template<typename TileStorageType, unsigned int NumChannels, int HaloLeft, int HaloTop,
                         typename ProcessingType = float>
                __device__ void compute_21_round_median_per_pixel(const TileStorageType& channels, int px, int py,
                                                                  ProcessingType* result) {
                    const int smem_x = px + HaloLeft;
                    const int smem_y = py + HaloTop;

#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        float v[21];
                        // Row -2: cols -1,0,1
                        v[0] = (float)channels.channel(c)(smem_y - 2, smem_x - 1);
                        v[1] = (float)channels.channel(c)(smem_y - 2, smem_x);
                        v[2] = (float)channels.channel(c)(smem_y - 2, smem_x + 1);
                        // Row -1: cols -2..2
                        v[3] = (float)channels.channel(c)(smem_y - 1, smem_x - 2);
                        v[4] = (float)channels.channel(c)(smem_y - 1, smem_x - 1);
                        v[5] = (float)channels.channel(c)(smem_y - 1, smem_x);
                        v[6] = (float)channels.channel(c)(smem_y - 1, smem_x + 1);
                        v[7] = (float)channels.channel(c)(smem_y - 1, smem_x + 2);
                        // Row 0
                        v[8]  = (float)channels.channel(c)(smem_y, smem_x - 2);
                        v[9]  = (float)channels.channel(c)(smem_y, smem_x - 1);
                        v[10] = (float)channels.channel(c)(smem_y, smem_x);
                        v[11] = (float)channels.channel(c)(smem_y, smem_x + 1);
                        v[12] = (float)channels.channel(c)(smem_y, smem_x + 2);
                        // Row 1
                        v[13] = (float)channels.channel(c)(smem_y + 1, smem_x - 2);
                        v[14] = (float)channels.channel(c)(smem_y + 1, smem_x - 1);
                        v[15] = (float)channels.channel(c)(smem_y + 1, smem_x);
                        v[16] = (float)channels.channel(c)(smem_y + 1, smem_x + 1);
                        v[17] = (float)channels.channel(c)(smem_y + 1, smem_x + 2);
                        // Row 2: cols -1,0,1
                        v[18] = (float)channels.channel(c)(smem_y + 2, smem_x - 1);
                        v[19] = (float)channels.channel(c)(smem_y + 2, smem_x);
                        v[20] = (float)channels.channel(c)(smem_y + 2, smem_x + 1);

                        n21_median_in_place(v);
                        result[c] = (ProcessingType)v[10];
                    }
                }

                // Cell-based 21-round median: same pattern as compute_3x3_median_cell - per row, call per-pixel helper with (smem_x, smem_y).
                template<unsigned int CellW, unsigned int CellH, typename TileStorageType, unsigned int NumChannels,
                         int HaloLeft, int HaloTop, typename CellType, typename ProcessingType = float>
                __device__ void compute_21_round_median_cell(const TileStorageType& input_tile, int cell_smem_x,
                                                             int cell_smem_y, CellType& output_cell) {
#pragma unroll
                    for (unsigned int cy = 0; cy < CellH; ++cy) {
                        const int smem_y = cell_smem_y + static_cast<int>(cy);

                        ProcessingType result[CellW][NumChannels];
#pragma unroll
                        for (unsigned int cx = 0; cx < CellW; ++cx) {
                            const int smem_x = cell_smem_x + static_cast<int>(cx);
                            compute_21_round_median_per_pixel<TileStorageType, NumChannels, HaloLeft, HaloTop,
                                                              ProcessingType>(input_tile, smem_x, smem_y, result[cx]);
                        }

#pragma unroll
                        for (unsigned int c = 0; c < NumChannels; ++c) {
#pragma unroll
                            for (unsigned int cx = 0; cx < CellW; ++cx) {
                                output_cell(c, cy * CellW + cx) = result[cx][c];
                            }
                        }
                    }
                }

                // One row of 5x5 median using Dobbelaere columnar prolog: load 25, 5x CE4 (columnar).
                // Cyclic column buffer: window col j at slot (cx+j)%5; load new column into slot cx%5 (explicit ifs, no divergence when unrolled).
                template<unsigned int CellW, typename TileStorageType, int HaloLeft, int HaloTop,
                         typename ProcessingType = float>
                __device__ void compute_5x5_median_row_columnar(const TileStorageType& channels, unsigned int c,
                                                                int cell_smem_x, int py, ProcessingType* result_row) {
                    constexpr int ColSlots    = 5;
                    const int     tile_y_base = py + HaloTop;
                    const int     tile_x_base = cell_smem_x + HaloLeft;

                    float row0[ColSlots], row1[ColSlots], row2[ColSlots], row3[ColSlots], row4[ColSlots];
                    // Prolog: fetch first 5x5 (coalesced by row)
#pragma unroll
                    for (int x = 0; x < ColSlots; ++x) {
                        const int tx = tile_x_base - 2 + x;
                        row0[x]      = (float)channels.channel(c)(tile_y_base - 2, tx);
                        row1[x]      = (float)channels.channel(c)(tile_y_base - 1, tx);
                        row2[x]      = (float)channels.channel(c)(tile_y_base, tx);
                        row3[x]      = (float)channels.channel(c)(tile_y_base + 1, tx);
                        row4[x]      = (float)channels.channel(c)(tile_y_base + 2, tx);
                    }
                    // Prolog: 5x CE4 - columnar (row1..4 per column) only
#pragma unroll
                    for (int x = 0; x < ColSlots; ++x)
                        ce4(row1[x], row2[x], row3[x], row4[x]);

#pragma unroll
                    for (unsigned int cx = 0; cx < CellW; ++cx) {
                        float v[25];
                        // Window col j from slot (cx+j)%ColSlots
#pragma unroll
                        for (int j = 0; j < ColSlots; ++j) {
                            const int s  = (cx + j) % ColSlots;
                            v[4 * j + 0] = row1[s];
                            v[4 * j + 1] = row2[s];
                            v[4 * j + 2] = row3[s];
                            v[4 * j + 3] = row4[s];
                            v[20 + j]    = row0[s];
                        }
                        n25_complete_median_in_place(v);
                        result_row[cx] = (ProcessingType)v[12];

                        if (cx < CellW - 1) {
                            const int new_col = tile_x_base + static_cast<int>(cx) + 3;
                            // Load into slot cx%5 (next window's rightmost). Explicit branches - no divergence when loop unrolled.
                            if (cx == 0 || cx == 5 || cx == 10) {
                                row0[0] = (float)channels.channel(c)(tile_y_base - 2, new_col);
                                row1[0] = (float)channels.channel(c)(tile_y_base - 1, new_col);
                                row2[0] = (float)channels.channel(c)(tile_y_base, new_col);
                                row3[0] = (float)channels.channel(c)(tile_y_base + 1, new_col);
                                row4[0] = (float)channels.channel(c)(tile_y_base + 2, new_col);
                                ce4(row1[0], row2[0], row3[0], row4[0]);
                            } else if (cx == 1 || cx == 6 || cx == 11) {
                                row0[1] = (float)channels.channel(c)(tile_y_base - 2, new_col);
                                row1[1] = (float)channels.channel(c)(tile_y_base - 1, new_col);
                                row2[1] = (float)channels.channel(c)(tile_y_base, new_col);
                                row3[1] = (float)channels.channel(c)(tile_y_base + 1, new_col);
                                row4[1] = (float)channels.channel(c)(tile_y_base + 2, new_col);
                                ce4(row1[1], row2[1], row3[1], row4[1]);
                            } else if (cx == 2 || cx == 7 || cx == 12) {
                                row0[2] = (float)channels.channel(c)(tile_y_base - 2, new_col);
                                row1[2] = (float)channels.channel(c)(tile_y_base - 1, new_col);
                                row2[2] = (float)channels.channel(c)(tile_y_base, new_col);
                                row3[2] = (float)channels.channel(c)(tile_y_base + 1, new_col);
                                row4[2] = (float)channels.channel(c)(tile_y_base + 2, new_col);
                                ce4(row1[2], row2[2], row3[2], row4[2]);
                            } else if (cx == 3 || cx == 8 || cx == 13) {
                                row0[3] = (float)channels.channel(c)(tile_y_base - 2, new_col);
                                row1[3] = (float)channels.channel(c)(tile_y_base - 1, new_col);
                                row2[3] = (float)channels.channel(c)(tile_y_base, new_col);
                                row3[3] = (float)channels.channel(c)(tile_y_base + 1, new_col);
                                row4[3] = (float)channels.channel(c)(tile_y_base + 2, new_col);
                                ce4(row1[3], row2[3], row3[3], row4[3]);
                            } else {
                                row0[4] = (float)channels.channel(c)(tile_y_base - 2, new_col);
                                row1[4] = (float)channels.channel(c)(tile_y_base - 1, new_col);
                                row2[4] = (float)channels.channel(c)(tile_y_base, new_col);
                                row3[4] = (float)channels.channel(c)(tile_y_base + 1, new_col);
                                row4[4] = (float)channels.channel(c)(tile_y_base + 2, new_col);
                                ce4(row1[4], row2[4], row3[4], row4[4]);
                            }
                        }
                    }
                }

                // 2x2 block path: load 4 full rows (CellW+2), buffer 4 shared middle pixels, complete N9 per corner.
                // Requires CellW and CellH even. Wire order + shared CEs verified vs NPP in IET (not only code inspection).
                template<unsigned int CellW, unsigned int CellH, typename TileStorageType, unsigned int NumChannels,
                         int HaloLeft, int HaloTop, typename CellType, typename ProcessingType = float>
                __device__ void compute_3x3_median_cell_2x2(const TileStorageType& input_tile, int cell_smem_x,
                                                            int cell_smem_y, CellType& output_cell) {
                    static_assert(CellW % 2 == 0 && CellH % 2 == 0, "2x2 path requires even cell width and height");
                    constexpr unsigned int row_len     = CellW + 2;
                    const int              base_tile_x = cell_smem_x + HaloLeft;
                    const int              base_tile_y = cell_smem_y + HaloTop;
#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
#pragma unroll
                        for (unsigned int by = 0; by < CellH; by += 2) {
                            // Load 4 full rows (one row per loop for coalesced reads)
                            float     row0[row_len], row1[row_len], row2[row_len], row3[row_len];
                            const int ty0 = base_tile_y + static_cast<int>(by) - 1;
#pragma unroll
                            for (unsigned int col = 0; col < row_len; ++col) {
                                const int tx = base_tile_x - 1 + static_cast<int>(col);
                                row0[col]    = (float)input_tile.channel(c)(ty0 + 0, tx);
                            }
#pragma unroll
                            for (unsigned int col = 0; col < row_len; ++col) {
                                const int tx = base_tile_x - 1 + static_cast<int>(col);
                                row1[col]    = (float)input_tile.channel(c)(ty0 + 1, tx);
                            }
#pragma unroll
                            for (unsigned int col = 0; col < row_len; ++col) {
                                const int tx = base_tile_x - 1 + static_cast<int>(col);
                                row2[col]    = (float)input_tile.channel(c)(ty0 + 2, tx);
                            }
#pragma unroll
                            for (unsigned int col = 0; col < row_len; ++col) {
                                const int tx = base_tile_x - 1 + static_cast<int>(col);
                                row3[col]    = (float)input_tile.channel(c)(ty0 + 3, tx);
                            }
#pragma unroll
                            for (unsigned int bx = 0; bx < CellW; bx += 2) {
                                // Buffer the 4 shared middle pixels (r11,r12,r21,r22); CE in place on buffer only
                                float m11 = row1[bx + 1], m12 = row1[bx + 2], m21 = row2[bx + 1], m22 = row2[bx + 2];
                                ce_full(m11, m12);
                                ce_full(m21, m22);

                                // Upper half: 3 more shared CEs, then remaining14 for upper-left and upper-right
                                float sh[9];
                                sh[0] = row0[bx + 1];
                                sh[7] = row0[bx + 2];
                                sh[1] = m11;
                                sh[2] = m12;
                                sh[3] = m21;
                                sh[5] = m22;
                                ce_full(sh[0], sh[7]);
                                ce_full(sh[0], sh[2]);
                                ce_full(sh[1], sh[5]);

                                float          v[9];
                                ProcessingType out_ul, out_ur, out_ll, out_lr;
                                // Upper-left: full v[] then remaining14 (v[] is overwritten in place)
                                v[0] = sh[0];
                                v[1] = sh[1];
                                v[2] = sh[2];
                                v[3] = sh[3];
                                v[4] = row0[bx + 0];
                                v[5] = sh[5];
                                v[6] = row1[bx + 0];
                                v[7] = sh[7];
                                v[8] = row2[bx + 0];
                                remaining14_median_after_shared_5ce(v);
                                out_ul = (ProcessingType)v[4];
                                // Upper-right: reinit full v[] (remaining14 overwrote it); only right column differs
                                v[0] = sh[0];
                                v[1] = sh[1];
                                v[2] = sh[2];
                                v[3] = sh[3];
                                v[4] = row0[bx + 3];
                                v[5] = sh[5];
                                v[6] = row1[bx + 3];
                                v[7] = sh[7];
                                v[8] = row2[bx + 3];
                                remaining14_median_after_shared_5ce(v);
                                out_ur = (ProcessingType)v[4];

                                // Lower half: same 4 middle pixels, 3 CEs on row3, then remaining14 for lower-left and lower-right
                                sh[0] = row3[bx + 1];
                                sh[7] = row3[bx + 2];
                                sh[1] = m11;
                                sh[2] = m12;
                                sh[3] = m21;
                                sh[5] = m22;
                                ce_full(sh[0], sh[7]);
                                ce_full(sh[0], sh[2]);
                                ce_full(sh[1], sh[5]);
                                // Lower-left: full v[]
                                v[0] = sh[0];
                                v[1] = sh[1];
                                v[2] = sh[2];
                                v[3] = sh[3];
                                v[4] = row1[bx + 0];
                                v[5] = sh[5];
                                v[6] = row2[bx + 0];
                                v[7] = sh[7];
                                v[8] = row3[bx + 0];
                                remaining14_median_after_shared_5ce(v);
                                out_ll = (ProcessingType)v[4];
                                // Lower-right: reinit full v[]
                                v[0] = sh[0];
                                v[1] = sh[1];
                                v[2] = sh[2];
                                v[3] = sh[3];
                                v[4] = row1[bx + 3];
                                v[5] = sh[5];
                                v[6] = row2[bx + 3];
                                v[7] = sh[7];
                                v[8] = row3[bx + 3];
                                remaining14_median_after_shared_5ce(v);
                                out_lr = (ProcessingType)v[4];

                                output_cell(c, (by + 0) * CellW + (bx + 0)) = out_ul;
                                output_cell(c, (by + 0) * CellW + (bx + 1)) = out_ur;
                                output_cell(c, (by + 1) * CellW + (bx + 0)) = out_ll;
                                output_cell(c, (by + 1) * CellW + (bx + 1)) = out_lr;
                            }
                        }
                    }
                }

                // 3x3 median cell: even CellW x CellH use compute_3x3_median_cell_2x2; otherwise sliding-window path
                // (three tile rows, column sort3 per column, then per-output-pixel row work - same math as N9L19D7 per pixel).
                template<unsigned int CellW, unsigned int CellH, typename TileStorageType, unsigned int NumChannels,
                         int HaloLeft, int HaloTop, typename CellType, typename ProcessingType = float>
                __device__ void compute_3x3_median_cell(const TileStorageType& input_tile, int cell_smem_x,
                                                        int cell_smem_y, CellType& output_cell) {
                    if constexpr (CellW % 2 == 0 && CellH % 2 == 0) {
                        compute_3x3_median_cell_2x2<CellW, CellH, TileStorageType, NumChannels, HaloLeft, HaloTop,
                                                    CellType, ProcessingType>(input_tile, cell_smem_x, cell_smem_y,
                                                                              output_cell);
                        return;
                    } else {
#pragma unroll
                        for (unsigned int c = 0; c < NumChannels; ++c) {
#pragma unroll
                            for (unsigned int cy = 0; cy < CellH; ++cy) {
                                const int smem_y = cell_smem_y + static_cast<int>(cy);

                                // Load 3 rows (CellW+2 wide) then column-sort in place. Match compute_3x3_median_per_pixel:
                                // it uses smem_x = px + HaloLeft, smem_y = py + HaloTop and reads at (smem_y±1, smem_x±1).
                                // So tile row = py + HaloTop ± 1, tile col = px + HaloLeft ± 1. Here px = cell_smem_x + cx (we load all cols for the row).
                                float row_0[CellW + 2], row_1[CellW + 2], row_2[CellW + 2];
#pragma unroll
                                for (unsigned int i = 0; i < CellW + 2; ++i) {
                                    const int tile_x = (cell_smem_x - 1 + static_cast<int>(i)) + HaloLeft;
                                    const int tile_y = (smem_y - 1) + HaloTop;
                                    row_0[i]         = (float)input_tile.channel(c)(tile_y, tile_x);
                                }
#pragma unroll
                                for (unsigned int i = 0; i < CellW + 2; ++i) {
                                    const int tile_x = (cell_smem_x - 1 + static_cast<int>(i)) + HaloLeft;
                                    const int tile_y = smem_y + HaloTop;
                                    row_1[i]         = (float)input_tile.channel(c)(tile_y, tile_x);
                                }
#pragma unroll
                                for (unsigned int i = 0; i < CellW + 2; ++i) {
                                    const int tile_x = (cell_smem_x - 1 + static_cast<int>(i)) + HaloLeft;
                                    const int tile_y = (smem_y + 1) + HaloTop;
                                    row_2[i]         = (float)input_tile.channel(c)(tile_y, tile_x);
                                }
#pragma unroll
                                // Vertical sorts, shared work
                                for (unsigned int i = 0; i < CellW + 2; ++i) {
                                    sort3(row_0[i], row_1[i],
                                          row_2[i]); // in place: row_0=min, row_1=med, row_2=max per column
                                }

                                // For each pixel: 3 sorted columns -> row sorts + final -> median at v[4].
                                // Trailing edge (row_*[cx]) is never reused for next cx, so use row_*[cx] directly and avoid v[0],v[3],v[6].
#pragma unroll
                                for (unsigned int cx = 0; cx < CellW; ++cx) {
                                    float v[9];
                                    v[1] = row_0[cx + 1];
                                    v[2] = row_0[cx + 2];
                                    v[4] = row_1[cx + 1];
                                    v[5] = row_1[cx + 2];
                                    v[7] = row_2[cx + 1];
                                    v[8] = row_2[cx + 2];

                                    // partial sorts with extinction (row_0[cx], row_1[cx], row_2[cx] used in place)
                                    // row 0 - max of 3
                                    v[2] = fmaxf(row_0[cx], fmaxf(v[1], v[2]));
                                    // row 1 - median of 3
                                    ce_full(row_1[cx], v[4]);
                                    v[4] = fminf(v[4], v[5]);
                                    v[4] = fmaxf(v[4], row_1[cx]);
                                    // row 2 - min of 3 (stomp row_2[cx])
                                    row_2[cx] = fminf(row_2[cx], fminf(v[7], v[8]));

                                    // diagonal median of 3
                                    ce_full(v[2], v[4]);
                                    v[4] = fminf(v[4], row_2[cx]);
                                    v[4] = fmaxf(v[4], v[2]);

                                    output_cell(c, cy * CellW + cx) = (ProcessingType)v[4];
                                }
                            }
                        }
                    }
                }

                // Dispatch: fast cell path by median_radius (if constexpr). Halo 1 for r0_5/r1_0, 2 for r1_5/r2_0.
                template<median_radius R, unsigned int CellW, unsigned int CellH, typename TileStorageType,
                         unsigned int NumChannels, int HaloLeft, int HaloTop, typename CellType,
                         typename ProcessingType>
                __device__ void median_cell(const TileStorageType& input_tile, int cell_smem_x, int cell_smem_y,
                                            CellType& output_cell) {
                    if constexpr (R == median_radius::r1_0) {
                        compute_3x3_median_cell<CellW, CellH, TileStorageType, NumChannels, HaloLeft, HaloTop, CellType,
                                                ProcessingType>(input_tile, cell_smem_x, cell_smem_y, output_cell);
                    } else if constexpr (R == median_radius::r0_5) {
                        compute_5_median_plus_cell<CellW, CellH, TileStorageType, NumChannels, CellType,
                                                   ProcessingType>(input_tile, cell_smem_x, cell_smem_y, output_cell);
                    } else if constexpr (R == median_radius::r1_5) {
                        compute_21_round_median_cell<CellW, CellH, TileStorageType, NumChannels, HaloLeft, HaloTop,
                                                     CellType, ProcessingType>(input_tile, cell_smem_x, cell_smem_y,
                                                                               output_cell);
                    } else if constexpr (R == median_radius::r2_0) {
                        ProcessingType row_buf[CellW];
                        for (unsigned int py = 0; py < CellH; ++py) {
                            const int row_smem_y = cell_smem_y + static_cast<int>(py);
                            for (unsigned int c = 0; c < NumChannels; ++c) {
                                compute_5x5_median_row_columnar<CellW, TileStorageType, HaloLeft, HaloTop,
                                                                ProcessingType>(input_tile, c, cell_smem_x, row_smem_y,
                                                                                row_buf);
                                for (unsigned int px = 0; px < CellW; ++px)
                                    output_cell(c, py * CellW + px) = row_buf[px];
                            }
                        }
                    }
                }

                // Dispatch: per-pixel slow path (boundary / partial cells).
                template<median_radius R, typename TileStorageType, unsigned int NumChannels, int HaloLeft, int HaloTop,
                         typename ProcessingType>
                __device__ void median_per_pixel(const TileStorageType& input_tile, int smem_x, int smem_y,
                                                 ProcessingType* result) {
                    if constexpr (R == median_radius::r1_0) {
                        compute_3x3_median_per_pixel<TileStorageType, NumChannels, HaloLeft, HaloTop, ProcessingType>(
                            input_tile, smem_x, smem_y, result);
                    } else if constexpr (R == median_radius::r0_5) {
                        compute_5_median_plus_from_3x3_per_pixel<TileStorageType, NumChannels, HaloLeft, HaloTop,
                                                                 ProcessingType>(input_tile, smem_x, smem_y, result);
                    } else if constexpr (R == median_radius::r2_0) {
                        compute_5x5_median_per_pixel<TileStorageType, NumChannels, HaloLeft, HaloTop, ProcessingType>(
                            input_tile, smem_x, smem_y, result);
                    } else if constexpr (R == median_radius::r1_5) {
                        compute_21_round_median_per_pixel<TileStorageType, NumChannels, HaloLeft, HaloTop,
                                                          ProcessingType>(input_tile, smem_x, smem_y, result);
                    }
                }

            } // namespace median
        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_MEDIAN_OPERATIONS_HPP
