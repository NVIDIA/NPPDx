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

#ifndef NPPDX_DETAIL_BACKEND_SHARPEN_OPERATIONS_HPP
#define NPPDX_DETAIL_BACKEND_SHARPEN_OPERATIONS_HPP

#include "nppdx/operators/function.hpp"

namespace nppdx {
    namespace detail {
        namespace backend {
            namespace sharpen {

                // Weights type: must expose corner_w, side_w, center_w (floating-point).
                // Weighted additive Laplacian: result = corner_w*(p00+p02+p20+p22) + side_w*(p01+p10+p12+p21) + center_w*p11.
                // For additive sharpening, the weights sum to 1.  for a simple generalized kernel, the positive weights should sum to 1.
                // Thus it can also be used for a blur
                //  a b a
                //  b c b
                //  a b a, where a = corner_w, b = side_w, c = center_w
                // no clipping is performed during this operation, but rather is saved for exgest
                template<typename Weights, typename TileStorageType, unsigned int NumChannels,
                         typename ProcessingType = float>
                __device__ void compute_3x3_sharpen(const TileStorageType& channels, int center_smem_x, int center_smem_y,
                                                   ProcessingType* result) {
                    const int cx = center_smem_x - 1;
                    const int cy = center_smem_y - 1; //-1 because we're using the center pixel as the center of the 3x3 kernel
                    constexpr ProcessingType cw = static_cast<ProcessingType>(Weights::corner_w);
                    constexpr ProcessingType sw = static_cast<ProcessingType>(Weights::side_w);
                    constexpr ProcessingType mw = static_cast<ProcessingType>(Weights::center_w);
#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        const ProcessingType p00 = channels.channel(c)(cy, cx);
                        const ProcessingType p01 = channels.channel(c)(cy, cx + 1);
                        const ProcessingType p02 = channels.channel(c)(cy, cx + 2);
                        const ProcessingType p10 = channels.channel(c)(cy + 1, cx);
                        const ProcessingType p11 = channels.channel(c)(cy + 1, cx + 1);
                        const ProcessingType p12 = channels.channel(c)(cy + 1, cx + 2);
                        const ProcessingType p20 = channels.channel(c)(cy + 2, cx);
                        const ProcessingType p21 = channels.channel(c)(cy + 2, cx + 1);
                        const ProcessingType p22 = channels.channel(c)(cy + 2, cx + 2);

                        //result[c] = cw * (p00 + p02 + p20 + p22) + sw * (p01 + p10 + p12 + p21) + mw * p11;
                        //result[c] = cw * p00 + sw * p01 + cw * p02 + sw * p10 + mw * p11 + sw * p12 +  cw * p20 + sw * p21 + cw * p22;
                        result[c] = cw * (p00 + p20) + cw * (p02 + p22) + 
                                    sw * (p01 + p21) + sw * (p10 + p12) + 
                                    mw * p11;
                    }
                }

                // Cell-based 3x3 sharpen: fill a CellW x CellH output cell from the tile.
                // completely unrolls the per-pixel version
                template<typename Weights, unsigned int CellW, unsigned int CellH, typename TileStorageType,
                         unsigned int NumChannels, typename CellType, typename ProcessingType = float>
                __device__ void compute_3x3_sharpen_cell(const TileStorageType& input_tile, int cell_smem_x,
                                                        int cell_smem_y, CellType& output_cell) {
#pragma unroll
                    //for each row
                    for (unsigned int cy = 0; cy < CellH; ++cy) {
                        const int smem_y = cell_smem_y + static_cast<int>(cy);

                        ProcessingType result[CellW][NumChannels];  //cache output row to allow coalesced writes
#pragma unroll
                        //process a row
                        for (unsigned int cx = 0; cx < CellW; ++cx) {
                            const int          smem_x    = cell_smem_x + static_cast<int>(cx);
                            compute_3x3_sharpen<Weights, TileStorageType, NumChannels, ProcessingType>(
                                input_tile, smem_x, smem_y, result[cx]);
                            }

                        // write the row results (planar: all ch0, then all ch1, then all ch2)
#pragma unroll
                        for (unsigned int c = 0; c < NumChannels; ++c) {
#pragma unroll
                            for (unsigned int cx = 0; cx < CellW; ++cx) {
                                output_cell(c, cy * CellW + cx) = result[cx][c];
                            }
                        }
                    }
                }

                // NPP adaptation since it requires an integer kernel - multiply everything by 144 (Divisor)
                template<typename Weights, int Divisor = 144>
                struct npp_kernel_from_weights {
                    static constexpr float corner_w  = Weights::corner_w;
                    static constexpr float side_w    = Weights::side_w;
                    static constexpr float center_w   = Weights::center_w;

                    static constexpr int round_f(float x) {
                        return static_cast<int>(x + (x >= 0.f ? 0.5f : -0.5f));
                    }
                    static constexpr int corner  = round_f(corner_w * static_cast<float>(Divisor));
                    static constexpr int side    = round_f(side_w * static_cast<float>(Divisor));
                    static constexpr int center  = round_f(center_w * static_cast<float>(Divisor));
                    static constexpr int divisor = Divisor;
                };
            } // namespace sharpen
        }     // namespace backend
    }         // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_SHARPEN_OPERATIONS_HPP
