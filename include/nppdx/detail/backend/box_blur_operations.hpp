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

#ifndef NPPDX_DETAIL_BACKEND_BOX_BLUR_OPERATIONS_HPP
#define NPPDX_DETAIL_BACKEND_BOX_BLUR_OPERATIONS_HPP

namespace nppdx {
    namespace detail {
        namespace backend {
            namespace box_blur {


                template<unsigned int KernelW, unsigned int KernelH, typename TileStorageType, unsigned int NumChannels,
                         int HaloLeft, int HaloTop, typename ProcessingType = float>
                __device__ void compute_NxM_blur(const TileStorageType& channels, int px, int py,
                                                 ProcessingType* result) {
                    static_assert(KernelW >= 1 && (KernelW % 2) == 1, "Kernel width must be odd");
                    static_assert(KernelH >= 1 && (KernelH % 2) == 1, "Kernel height must be odd");

                    const int smem_x = px + HaloLeft;
                    const int smem_y = py + HaloTop;

                    constexpr int            halo_x  = KernelW / 2;
                    constexpr int            halo_y  = KernelH / 2;
                    constexpr ProcessingType inv_sum = ProcessingType(1) / ProcessingType(KernelW * KernelH);

#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        ProcessingType sum = ProcessingType(0);

#pragma unroll
                        for (int dy = -halo_y; dy <= halo_y; ++dy) {
#pragma unroll
                            for (int dx = -halo_x; dx <= halo_x; ++dx) {
                                sum += channels.channel(c)(smem_y + dy, smem_x + dx);
                            }
                        }
                        result[c] = sum * inv_sum;
                    }
                }

                // Cell-based box blur (separable implementation; details internal to this function).
                // Uses 1D array (CellW + KernelW - 1 elements) for register efficiency.
                template<unsigned int KernelW, unsigned int KernelH, unsigned int CellW, unsigned int CellH,
                         typename TileStorageType, unsigned int NumChannels, typename CellType, typename ProcessingType = float>
                __device__ void compute_NxM_blur_cell(const TileStorageType& input_channels, int cell_smem_x, int cell_smem_y,
                                                      CellType& output_cell) {
                    static_assert(KernelW >= 1 && (KernelW % 2) == 1, "Kernel width must be odd");
                    static_assert(KernelH >= 1 && (KernelH % 2) == 1, "Kernel height must be odd");

                    constexpr int            halo_x  = KernelW / 2;
                    constexpr int            halo_y  = KernelH / 2;
                    constexpr ProcessingType inv_sum = ProcessingType(1) / ProcessingType(KernelW * KernelH);

                    //-------------------- Optimized: horizontal row sums, slide vertically per column
#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        // Pre-build horizontal sums for first column position (leftmost KernelW pixels)
                        // One sum for each vertical position covering CellH + KernelH - 1 rows
                        ProcessingType row_sums[CellH + KernelH - 1];
#pragma unroll
                        for (unsigned int ry = 0; ry < CellH + KernelH - 1; ++ry) {
                            const int smem_y = cell_smem_y - halo_y + ry;
                            ProcessingType sum = ProcessingType(0);
#pragma unroll
                            for (int dx = 0; dx < KernelW; ++dx) {
                                const int smem_x = cell_smem_x - halo_x + dx;
                                sum += input_channels.channel(c)(smem_y, smem_x);
                            }
                            row_sums[ry] = sum;
                        }

                        // Process each output column
#pragma unroll
                        for (unsigned int cx = 0; cx < CellW; ++cx) {
                            // Initialize vertical sum with first KernelH-1 row sums
                            ProcessingType vert_sum = ProcessingType(0);
#pragma unroll
                            for (int ry = 0; ry < KernelH - 1; ++ry) {
                                vert_sum += row_sums[ry];
                            }

                            // Slide vertically through rows
#pragma unroll
                            for (unsigned int cy = 0; cy < CellH; ++cy) {
                                // Add last row sum to complete the kernel
                                vert_sum += row_sums[cy + KernelH - 1];

                                // Produce output
                                const unsigned int pixel_idx = cy * CellW + cx;
                                output_cell(c, pixel_idx) = vert_sum * inv_sum;

                                // Slide the window down (remove top row sum)
                                vert_sum -= row_sums[cy];
                            }

                            // Update row sums by sliding horizontally (remove left pixel, add right pixel)
                            if (cx < CellW - 1) {
#pragma unroll
                                for (unsigned int ry = 0; ry < CellH + KernelH - 1; ++ry) {
                                    const int smem_y = cell_smem_y - halo_y + ry;
                                    const int smem_x_right = cell_smem_x - halo_x + cx + KernelW;
                                    const int smem_x_left = cell_smem_x - halo_x + cx;
                                    row_sums[ry] += input_channels.channel(c)(smem_y, smem_x_right) -
                                                   input_channels.channel(c)(smem_y, smem_x_left);
                                }
                            }
                        }
                    }
                    //------------------
                }
            } // namespace box_blur
        }     // namespace backend
    }         // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_BOX_BLUR_OPERATIONS_HPP
