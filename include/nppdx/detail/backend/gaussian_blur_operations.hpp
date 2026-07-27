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

#ifndef NPPDX_DETAIL_BACKEND_GAUSSIAN_BLUR_OPERATIONS_HPP
#define NPPDX_DETAIL_BACKEND_GAUSSIAN_BLUR_OPERATIONS_HPP

#include "nppdx/detail/config.hpp"
#include "nppdx/detail/backend/gaussian_blur_coeff_tables.hpp"
#include "nppdx/operators/function.hpp"
#include "nppdx/detail/database/region.hpp"
#include <nppdx/shared_memory.hpp>

namespace nppdx {
    namespace detail {
        namespace backend {
            namespace gaussian_blur {

                // Discrete Gaussian FIR: gaussian_fir_execution_spec<RT,Tail> bundles n_taps (odd tap count 3..21 by
                // band; wide uses 2*taps_std-1), RadiusTenths (table row), and padded constexpr weights (same sigma=R).

                // Single-output FIR (full 2D product of 1D taps). Template NTaps must match RadiusTenths kernel width.
                template<unsigned int NTaps, int RadiusTenths, gaussian_tail_width TailWidth, typename TileStorageType,
                         unsigned int NumChannels, int HaloLeft, int HaloTop, typename ProcessingType = float>
                __device__ void compute_gaussian_blur_fir_2d(const TileStorageType& channels, int px, int py,
                                                             ProcessingType* result) {
                    static_assert(NTaps == gaussian_fir_execution_spec<RadiusTenths, TailWidth>::n_taps,
                                  "NTaps must match gaussian_blur_params::kernel_w for this RadiusTenths / tail");
                    constexpr int            halo = static_cast<int>(NTaps / 2u);
                    constexpr padded_fir_row w1d =
                        gaussian_fir_execution_spec<RadiusTenths, TailWidth>::weights_1d_padded();

                    const int smem_x = px + HaloLeft;
                    const int smem_y = py + HaloTop;
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        ProcessingType sum = ProcessingType(0);
#pragma unroll 5
                        for (int dy = -halo; dy <= halo; ++dy) {
#pragma unroll 5
                            for (int dx = -halo; dx <= halo; ++dx) {
                                sum += channels.channel(c)(smem_y + dy, smem_x + dx) *
                                       w1d[static_cast<unsigned int>(dx + halo)] *
                                       w1d[static_cast<unsigned int>(dy + halo)];
                            }
                        }
                        result[c] = sum;
                    }
                }

                // Cell-based separable Gaussian blur. Row fetch len = CellW + NTaps - 1 (e.g. 16 floats for 4-wide + 13 tap).
                template<unsigned int NTaps, int RadiusTenths, gaussian_tail_width TailWidth, unsigned int CellW,
                         unsigned int CellH, typename TileStorageType, unsigned int NumChannels, typename CellType,
                         typename ProcessingType = float>
                __device__ void compute_gaussian_blur_cell_separable(const TileStorageType& input_channels,
                                                                     int cell_smem_x, int cell_smem_y,
                                                                     CellType& output_cell) {
                    static_assert(NTaps == gaussian_fir_execution_spec<RadiusTenths, TailWidth>::n_taps,
                                  "NTaps must match gaussian_blur_params::kernel_w for this RadiusTenths / tail");
                    static_assert(NTaps == 3u || NTaps == 5u || NTaps == 7u || NTaps == 9u || NTaps == 11u ||
                                      NTaps == 13u || NTaps == 15u || NTaps == 17u || NTaps == 19u || NTaps == 21u,
                                  "Gaussian FIR cellular path supports odd tap counts 3..21 used by the RT bands");
                    constexpr padded_fir_row w1d =
                        gaussian_fir_execution_spec<RadiusTenths, TailWidth>::weights_1d_padded();

                    constexpr int          halo     = static_cast<int>(NTaps / 2);
                    constexpr unsigned int buf_rows = CellH + NTaps - 1;
                    constexpr unsigned int buf_cols = CellW;
                    constexpr unsigned int row_len = CellW + NTaps - 1; // input cols needed per row for horizontal pass

#pragma unroll
                    for (unsigned int ch = 0; ch < NumChannels; ++ch) {
                        ProcessingType horz[buf_rows * buf_cols];
#pragma unroll
                        for (unsigned int ry = 0; ry < buf_rows; ++ry) {
                            const int smem_y = cell_smem_y - halo + static_cast<int>(ry);
                            // Fetch entire row once (one read per input column).
                            ProcessingType row_buf[row_len];
#pragma unroll
                            for (unsigned int i = 0; i < row_len; ++i) {
                                row_buf[i] =
                                    input_channels.channel(ch)(smem_y, cell_smem_x - halo + static_cast<int>(i));
                            }
                            // Horizontal dot products into horz[ry][cx].
#pragma unroll
                            for (unsigned int cx = 0; cx < buf_cols; ++cx) {
                                ProcessingType s = ProcessingType(0);
#pragma unroll
                                for (int dx = 0; dx < static_cast<int>(NTaps); ++dx) {
                                    s += row_buf[cx + static_cast<unsigned int>(dx)] *
                                         w1d[static_cast<unsigned int>(dx)];
                                }
                                horz[ry * buf_cols + cx] = s;
                            }
                        }
                        // Vertical pass: for each output (cy, cx), sum kernel[dy] * horz[cy+halo+dy][cx]
#pragma unroll
                        for (unsigned int cy = 0; cy < CellH; ++cy) {
#pragma unroll
                            for (unsigned int cx = 0; cx < CellW; ++cx) {
                                ProcessingType s = ProcessingType(0);
#pragma unroll
                                for (int dy = 0; dy < static_cast<int>(NTaps); ++dy) {
                                    s += horz[(cy + static_cast<unsigned int>(dy)) * buf_cols + cx] *
                                         w1d[static_cast<unsigned int>(dy)];
                                }
                                const unsigned int pixel_idx = cy * CellW + cx;
                                output_cell(ch, pixel_idx)   = s;
                            }
                        }
                    }
                }

                // Top-level Gaussian for one cell: separable FIR in registers when the cell is fully inside the ROI;
                // otherwise per-pixel full 2D FIR at tile edges.
                template<int RadiusTenths, gaussian_tail_width TailWidth, typename Region, unsigned int CellW,
                         unsigned int CellH, typename TileStorageType, unsigned int NumChannels, typename CellType,
                         typename ProcessingType>
                __device__ void hybrid_gaussian_cell(const TileStorageType& input_tile, int cell_smem_x,
                                                     int cell_smem_y, bool fully_valid, CellType& cell) {
                    using fir_spec = gaussian_fir_execution_spec<RadiusTenths, TailWidth>;

                    if (fully_valid) {
                        compute_gaussian_blur_cell_separable<fir_spec::n_taps, RadiusTenths, TailWidth, CellW, CellH,
                                                             TileStorageType, NumChannels, CellType, ProcessingType>(
                            input_tile, cell_smem_x, cell_smem_y, cell);
                    } else {
#pragma unroll
                        for (unsigned int py = 0; py < CellH; ++py) {
                            int smem_y = cell_smem_y + static_cast<int>(py);
#pragma unroll
                            for (unsigned int px = 0; px < CellW; ++px) {
                                int          smem_x    = cell_smem_x + static_cast<int>(px);
                                unsigned int pixel_idx = py * CellW + px;
                                if (copy(Region::value).contains(OuterPos(int2(smem_x, smem_y)))) {
                                    ProcessingType blur_result[NumChannels];
                                    compute_gaussian_blur_fir_2d<fir_spec::n_taps, RadiusTenths, TailWidth,
                                                                 TileStorageType, NumChannels, 0, 0, ProcessingType>(
                                        input_tile, smem_x, smem_y, blur_result);
#pragma unroll
                                    for (unsigned int c = 0; c < NumChannels; ++c) {
                                        cell(c, pixel_idx) = blur_result[c];
                                    }
                                }
                            }
                        }
                    }
                }

            } // namespace gaussian_blur
        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_GAUSSIAN_BLUR_OPERATIONS_HPP
