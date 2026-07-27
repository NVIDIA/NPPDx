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

#ifndef NPPDX_DETAIL_DATABASE_FUNCTION_OPERATION_IMPLEMENTATION_HPP
#define NPPDX_DETAIL_DATABASE_FUNCTION_OPERATION_IMPLEMENTATION_HPP

#include "nppdx/detail/database/region.hpp"
#include "nppdx/detail/database/cell_executor.hpp"
#include "nppdx/detail/database/input_output/npp_adapter.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/detail/backend/median_operations.hpp"
#include "nppdx/detail/backend/color_conversion_operations.hpp"
#include "nppdx/detail/backend/gamma_transform_operations.hpp"
#include "nppdx/detail/backend/affine_channel_map_operations.hpp"
#include "nppdx/detail/backend/box_blur_operations.hpp"
#include "nppdx/detail/backend/gaussian_blur_operations.hpp"
#include "nppdx/detail/backend/sharpen_operations.hpp"
#include "nppdx/detail/backend/resize_operations.hpp"
#include "nppdx/operators/function.hpp"
#include "nppdx/detail/database/operation_common.hpp"
#include "nppdx/detail/database/region.hpp"
#include "nppdx/operators/halo.hpp"

namespace nppdx {
    namespace detail {
        template<typename Operation, typename Layout, typename CellCfg>
        struct FuncOpImplHelper: public OpImplHelper<Operation, Layout, CellCfg> {
            NPPDX_DECL_SC Halo4 local_halo =
                operation_operator_traits<typename Operation::tag_type>::get_local_halo(Operation::value);
            NPPDX_DECL_SC region output_region = make_output_region(Layout::value, local_halo); // Within smem buffer.
            NPPDX_DECL_SC auto   cell_grid     = aligned_cell_grid {output_region.size, CellCfg::value.size};

            using executor_type = aligned_cell_executor<NPPDX_MAKE_UINT2D(cell_grid.cell_count()),
                                                        NPPDX_MAKE_INT2D(output_region.offset), CellCfg>;

            NPPDX_DECL_NSCFHD OuterPos smem_pos(InnerCellIdx cell_idx) {
                return get_smem_pos(copy(cell_grid), cell_idx, output_region.offset);
            }
        };




        // =============================================================================
        // Color Convert
        // =============================================================================
        template<color_space InputColorSpace, color_space OutputColorSpace, bit_depth InputBitDepth,
                 bit_depth OutputBitDepth, typename Layout, typename CellCfg, int SM, int BlockThreads>
        class op_impl<ColorConvert<InputColorSpace, OutputColorSpace, InputBitDepth, OutputBitDepth>, Layout, CellCfg,
                      SM, BlockThreads>
        {
        private:
            using Operation = ColorConvert<InputColorSpace, OutputColorSpace, InputBitDepth, OutputBitDepth>;
            using OpHelper  = FuncOpImplHelper<Operation, Layout, CellCfg>;

            template<typename T, typename CellType>
            NPPDX_DECL_SD void execute_inner(CellType cell) {
                backend::color_conversion::convert<InputBitDepth, OutputBitDepth, InputColorSpace, OutputColorSpace>(
                    cell);
            }

        public:
            // Register-based execute
            template<typename T>
            NPPDX_DECL_SD void execute(T* data, int, int) {
                using cell_type = NPPCellData<CellCfg, T>;
                using executor  = typename OpHelper::executor_type;
                executor::template for_each_linear_idx</*Strided=*/false>(
                    [data](uint) { execute_inner<T>(cell_type(data)); });
            }

            // Shared memory execute
            NPPDX_DECL_SD void execute(typename OpHelper::TileStorageType& tile, int, int) {
                using internal_processing_type = typename OpHelper::TileStorageType::ChannelSliceType::value_type;
                using executor                 = typename OpHelper::executor_type;
                executor::template execute_inplace</*Strided=*/false>(tile, Layout::value.memory_size(), [](auto cell) {
                    execute_inner<internal_processing_type>(cell);
                });
                __syncthreads();
            }
        };

        // =============================================================================
        // Gamma Transform
        // =============================================================================
        template<bit_depth BitDepth, gamma_dir TransformDirection, gamma_transfer_function TransferFunction,
                 typename Layout, typename CellCfg, int SM, int BlockThreads>
        struct op_impl<GammaTransform<BitDepth, TransformDirection, TransferFunction>, Layout, CellCfg, SM,
                       BlockThreads> {
            using Operation = GammaTransform<BitDepth, TransformDirection, TransferFunction>;
            using OpHelper  = FuncOpImplHelper<Operation, Layout, CellCfg>;

        private:
            template<typename T>
            NPPDX_DECL_SD void apply_gamma(T (&v)[3]) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    v[i] = backend::scale_to_zerone<BitDepth>(v[i]);
                    v[i] = backend::gamma_transform_operations::interpolate<TransformDirection, TransferFunction>(v[i]);
                    v[i] = backend::scale_from_zerone<BitDepth>(v[i]);
                }
            }

            template<typename T, typename CellType>
            NPPDX_DECL_SD void execute_inner(CellType cell) {
#pragma unroll
                for (unsigned int i = 0; i < OpHelper::cell_cfg.cell_pixels(); ++i) {
                    T v[3] = {cell(0, i), cell(1, i), cell(2, i)};
                    apply_gamma(v);
                    cell(0, i) = v[0], cell(1, i) = v[1], cell(2, i) = v[2];
                }
            }

        public:
            // Register-based execute
            template<typename T>
            NPPDX_DECL_SD void execute(T* data, int, int) {
                using cell_type = NPPCellData<CellCfg, T>;
                using executor  = typename OpHelper::executor_type;
                executor::template for_each_linear_idx</*Strided=*/false>(
                    [data](uint) { execute_inner<T>(cell_type(data)); });
            }

            // Shared memory execute
            NPPDX_DECL_SD void execute(typename OpHelper::TileStorageType& tile, int, int) {
                using internal_processing_type = typename OpHelper::TileStorageType::ChannelSliceType::value_type;
                using executor                 = typename OpHelper::executor_type;
                executor::template execute_inplace</*Strided=*/false>(tile, Layout::value.memory_size(), [](auto cell) {
                    execute_inner<internal_processing_type>(cell);
                });
                __syncthreads();
            }
        };

        // =============================================================================
        // Affine channel map
        // =============================================================================
        template<int32_t PreOffset, int32_t Numerator, int32_t Denominator, int32_t PostOffset, int32_t ClipMin,
                 int32_t ClipMax, uint32_t ChannelMask, typename Layout, typename CellCfg, int SM, int BlockThreads>
        struct op_impl<AffineChannelMap<PreOffset, Numerator, Denominator, PostOffset, ClipMin, ClipMax, ChannelMask>,
                       Layout, CellCfg, SM, BlockThreads> {
            using Operation =
                AffineChannelMap<PreOffset, Numerator, Denominator, PostOffset, ClipMin, ClipMax, ChannelMask>;
            using OpHelper = FuncOpImplHelper<Operation, Layout, CellCfg>;

        private:
            template<typename T>
            NPPDX_DECL_SD void apply_affine(T (&v)[3]) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    if ((ChannelMask & (uint32_t {1} << i)) != 0) {
                        v[i] = backend::affine_channel_map_operations::apply<PreOffset, Numerator, Denominator,
                                                                             PostOffset, ClipMin, ClipMax>(v[i]);
                    }
                }
            }

            template<typename T, typename CellType>
            NPPDX_DECL_SD void execute_inner(CellType cell) {
#pragma unroll
                for (unsigned int i = 0; i < OpHelper::cell_cfg.cell_pixels(); ++i) {
                    T v[3] = {cell(0, i), cell(1, i), cell(2, i)};
                    apply_affine(v);
                    cell(0, i) = v[0], cell(1, i) = v[1], cell(2, i) = v[2];
                }
            }

        public:
            template<typename T>
            NPPDX_DECL_SD void execute(T* data, int, int) {
                using cell_type = NPPCellData<CellCfg, T>;
                using executor  = typename OpHelper::executor_type;
                executor::template for_each_linear_idx</*Strided=*/false>(
                    [data](uint) { execute_inner<T>(cell_type(data)); });
            }

            NPPDX_DECL_SD void execute(typename OpHelper::TileStorageType& tile, int, int) {
                using internal_processing_type = typename OpHelper::TileStorageType::ChannelSliceType::value_type;
                using executor                 = typename OpHelper::executor_type;
                executor::template execute_inplace</*Strided=*/false>(tile, Layout::value.memory_size(), [](auto cell) {
                    execute_inner<internal_processing_type>(cell);
                });
                __syncthreads();
            }
        };

        // =============================================================================
        // Box Blur
        // =============================================================================
        template<unsigned int KernelW, unsigned int KernelH, typename Layout, typename CellCfg, int SM,
                 int BlockThreads>
        struct op_impl<BoxBlur<KernelW, KernelH>, Layout, CellCfg, SM, BlockThreads> {
            using Operation = BoxBlur<KernelW, KernelH>;
            using OpHelper  = FuncOpImplHelper<Operation, Layout, CellCfg>;

        public:
            // Register-based execute (not supported, requires shared memory)
            template<typename T>
            NPPDX_DECL_SD void execute(T*, int, int) {
                static_assert(!NPPDX_STD::is_same_v<T, T>, "Box Blur is not supported for register-based execution");
            }

            // Double-buffered execute (input -> output)
            NPPDX_DECL_SD void execute(const typename OpHelper::TileStorageType& input_tile,
                                       typename OpHelper::TileStorageType&       output_tile, int, int) {
                using internal_processing_type = typename OpHelper::TileStorageType::ChannelSliceType::value_type;
                using executor                 = typename OpHelper::executor_type;

                execute_naive<executor>(input_tile, output_tile);
                __syncthreads();
            }

        private:

            // Naive implementation for small kernels
            template<typename executor>
            NPPDX_DECL_SD void execute_naive(const typename OpHelper::TileStorageType& input_tile,
                                             typename OpHelper::TileStorageType&       output_tile) {
                using T = typename OpHelper::TileStorageType::ChannelSliceType::value_type;
                T                       regs[CellCfg::value.elements_per_cell()];
                NPPCellData<CellCfg, T> cell(regs);

                executor::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    const OuterPos smem_cell_start = OpHelper::smem_pos(cell_idx);
                    // Compute valid pixel range for this cell (bounds check optimization)
                    const int2 smem_cell_end = smem_cell_start + int2(copy(CellCfg::CellSize::value));

                    constexpr int2 clip_size = OpHelper::output_region.outer_right_bottom_pos();

                    // Check if entire cell is within valid region (common case).
                    // We do not need to check on the lower edge.
                    const bool fully_valid = all(smem_cell_end <= clip_size);

                    if (fully_valid) {
                        execute_cell</*Fast=*/true>(input_tile, cell, smem_cell_start);
                    } else {
                        execute_cell</*Fast=*/false>(input_tile, cell, smem_cell_start);
                    }
                    store_cell_to_tile_clipped(cell, output_tile, smem_cell_start, uint2(clip_size));
                });
            }

            // Fast path: separable cell blur. Slow path: per-pixel naive blur + clip checks.
            template<bool Fast, typename CellType>
            NPPDX_DECL_SD void execute_cell(const typename OpHelper::TileStorageType& input_tile, CellType& cell,
                                            int2 smem_pos) {
                if constexpr (Fast) {
                    constexpr unsigned int cell_w = CellCfg::CellSize::X;
                    constexpr unsigned int cell_h = CellCfg::CellSize::Y;
                    backend::box_blur::compute_NxM_blur_cell<KernelW, KernelH, cell_w, cell_h,
                                                             typename OpHelper::TileStorageType, OpHelper::num_channels,
                                                             CellType>(input_tile, smem_pos.x, smem_pos.y, cell);
                } else {
                    constexpr int2 cell_size = int2(CellCfg::value.size);
                    // Conditionally unroll the outer two loop levels.
                    // Aim toward the number of registers involved, with a bias towards horizontal unroll
                    constexpr uint inner_iters = OpHelper::num_channels * KernelW * KernelH;
                    constexpr int  y_unroll    = inner_iters < 50u ? cell_size.y : 1;
                    constexpr int  x_unroll    = inner_iters < 300u ? cell_size.x : 1;

#pragma unroll y_unroll
                    for (unsigned int py = 0; py < cell_size.y; ++py) {
                        int smem_y = smem_pos.y + static_cast<int>(py);
#pragma unroll x_unroll
                        for (unsigned int px = 0; px < cell_size.x; ++px) {
                            int          smem_x    = smem_pos.x + static_cast<int>(px);
                            unsigned int pixel_idx = py * cell_size.x + px;

                            if (copy(OpHelper::output_region).contains(OuterPos(int2 {smem_x, smem_y}))) {
                                float blur_result[OpHelper::num_channels];
                                backend::box_blur::compute_NxM_blur<
                                    KernelW, KernelH, typename OpHelper::TileStorageType, OpHelper::num_channels, 0, 0>(
                                    input_tile, smem_x, smem_y, blur_result);
#pragma unroll
                                for (unsigned int c = 0; c < OpHelper::num_channels; ++c) {
                                    cell(c, pixel_idx) = blur_result[c];
                                }
                            }
                        }
                    }
                }
            }
        };

        // =============================================================================
        // Gaussian Blur - radius 0.1-10.0 (RadiusTenths 1-100). Path is strictly radius-derived (FIR vs IIR+5x5);
        // user only specifies RadiusTenths. All operation logic lives in gaussian_blur_operations.hpp.
        // =============================================================================
        template<int RadiusTenths, gaussian_tail_width TailWidth, typename Layout, typename CellCfg, int SM,
                 int BlockThreads>
        struct op_impl<GaussianBlur<RadiusTenths, TailWidth>, Layout, CellCfg, SM, BlockThreads> {
            using Operation = GaussianBlur<RadiusTenths, TailWidth>;
            using OpHelper  = FuncOpImplHelper<Operation, Layout, CellCfg>;

        public:
            // Register-based execute (not supported, requires shared memory)
            template<typename T>
            NPPDX_DECL_SD void execute(T*, int, int) {
                static_assert(!NPPDX_STD::is_same_v<T, T>,
                              "Gaussian Blur is not supported for register-based execution");
            }

            NPPDX_DECL_SD void execute(const typename OpHelper::TileStorageType& input_tile,
                                       typename OpHelper::TileStorageType&       output_tile, int, int) {
                using T = typename OpHelper::TileStorageType::ChannelSliceType::value_type;
                T                       regs[CellCfg::value.elements_per_cell()];
                NPPCellData<CellCfg, T> cell(regs);

                OpHelper::executor_type::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    const OuterPos smem_cell_start = OpHelper::smem_pos(cell_idx);
                    // Compute valid pixel range for this cell (bounds check optimization)
                    const int2 smem_cell_end = smem_cell_start + int2(copy(CellCfg::CellSize::value));

                    constexpr int2 clip_size = OpHelper::output_region.outer_right_bottom_pos();

                    // Check if entire cell is within valid region (common case).
                    // We do not need to check on the lower edge.
                    const bool fully_valid = all(smem_cell_end <= clip_size);

                    backend::gaussian_blur::hybrid_gaussian_cell<
                        RadiusTenths, TailWidth, NPPDX_MAKE_REGION(OpHelper::output_region), CellCfg::CellSize::X,
                        CellCfg::CellSize::Y, typename OpHelper::TileStorageType, OpHelper::num_channels,
                        NPPCellData<CellCfg, T>, T>(input_tile, smem_cell_start->x, smem_cell_start->y, fully_valid,
                                                    cell);

                    store_cell_to_tile_clipped(cell, output_tile, smem_cell_start, uint2(clip_size));
                });
                __syncthreads();
            }
        };

        // =============================================================================
        // Median - radius enum; backend::median::median_cell / median_per_pixel dispatch (if constexpr).
        // =============================================================================

        // All supported radii: one implementation; backend::median::median_cell / median_per_pixel dispatch by Radius.
        template<median_radius Radius, typename Layout, typename CellCfg, int SM, int BlockThreads>
        struct op_impl<Median<Radius>, Layout, CellCfg, SM, BlockThreads> {
            using Operation = Median<Radius>;
            using OpHelper  = FuncOpImplHelper<Operation, Layout, CellCfg>;

            // Register-based execute (not supported, requires shared memory)
            template<typename T>
            NPPDX_DECL_SD void execute(T*, int, int) {
                static_assert(!NPPDX_STD::is_same_v<T, T>, "Median is not supported for register-based execution");
            }

            __device__ static void execute(const typename OpHelper::TileStorageType& input_tile,
                                           typename OpHelper::TileStorageType&       output_tile, int, int) {
                using T = typename OpHelper::TileStorageType::ChannelSliceType::value_type;
                T                       regs[CellCfg::value.elements_per_cell()];
                NPPCellData<CellCfg, T> cell(regs);

                OpHelper::executor_type::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    const OuterPos smem_cell_start = OpHelper::smem_pos(cell_idx);
                    const int2     smem_cell_end   = smem_cell_start + int2(copy(CellCfg::CellSize::value));
                    constexpr int2 clip_size       = OpHelper::output_region.outer_right_bottom_pos();

                    // Check if entire cell is within valid region (common case).
                    // We do not need to check on the lower edge.
                    const bool fully_valid = all(smem_cell_end <= clip_size);

                    if (fully_valid) {
                        execute_cell_fast(input_tile, cell, smem_cell_start);
                    } else {
                        execute_cell_slow(input_tile, cell, smem_cell_start);
                    }
                    store_cell_to_tile_clipped(cell, output_tile, smem_cell_start, uint2(clip_size));
                });
                __syncthreads();
            }

        private:
            template<typename CellType, typename PosType>
            __device__ static void execute_cell_fast(const typename OpHelper::TileStorageType& input_tile,
                                                     CellType& cell, const PosType& pos) {
                using internal_processing_type = typename OpHelper::TileStorageType::ChannelSliceType::value_type;
                constexpr uint2 cell_size      = copy(CellCfg::CellSize::value);
                backend::median::median_cell<Radius, cell_size.x, cell_size.y, typename OpHelper::TileStorageType,
                                             OpHelper::num_channels, 0, 0, CellType, internal_processing_type>(
                    input_tile, pos->x, pos->y, cell);
            }

            template<typename CellType, typename PosType>
            __device__ static void execute_cell_slow(const typename OpHelper::TileStorageType& input_tile,
                                                     CellType& cell, const PosType& pos) {
                using internal_processing_type = typename OpHelper::TileStorageType::ChannelSliceType::value_type;
                constexpr uint2 cell_size      = copy(CellCfg::CellSize::value);
                constexpr uint2 unroll_size =
                    (Radius == median_radius::r1_5) ? copy(CellCfg::CellSize::value) : uint2(1, 1);
#pragma unroll unroll_size.y
                for (unsigned int py = 0; py < cell_size.y; ++py) {
                    const int smem_y = pos->y + static_cast<int>(py);
#pragma unroll unroll_size.x
                    for (unsigned int px = 0; px < cell_size.x; ++px) {
                        const int          smem_x    = pos->x + static_cast<int>(px);
                        const unsigned int pixel_idx = py * cell_size.x + px;
                        if (copy(OpHelper::output_region).contains(OuterPos(int2 {smem_x, smem_y}))) {
                            internal_processing_type median_result[OpHelper::num_channels];
                            backend::median::median_per_pixel<Radius, typename OpHelper::TileStorageType,
                                                              OpHelper::num_channels, 0, 0, internal_processing_type>(
                                input_tile, smem_x, smem_y, median_result);
#pragma unroll
                            for (unsigned int c = 0; c < OpHelper::num_channels; ++c)
                                cell(c, pixel_idx) = median_result[c];
                        }
                    }
                }
            }
        };


        // =============================================================================
        // Sharpen - Weights-based (corner_w, side_w, center_w). Clip at exgest for fused pipelines.
        // =============================================================================
        template<typename Weights, typename Layout, typename CellCfg, int SM, int BlockThreads>
        struct op_impl<Sharpen<Weights>, Layout, CellCfg, SM, BlockThreads> {
            using Operation = Sharpen<Weights>;
            using OpHelper  = FuncOpImplHelper<Operation, Layout, CellCfg>;

            template<typename T>
            NPPDX_DECL_SD void execute(T*, int, int) {
                static_assert(!NPPDX_STD::is_same_v<T, T>, "Sharpen is not supported for register-based execution");
            }

            __device__ static void execute(const typename OpHelper::TileStorageType& input_tile,
                                           typename OpHelper::TileStorageType&       output_tile, int, int) {
                using T = typename OpHelper::TileStorageType::ChannelSliceType::value_type;
                T                       regs[CellCfg::value.elements_per_cell()];
                NPPCellData<CellCfg, T> cell(regs);

                OpHelper::executor_type::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    const OuterPos smem_cell_start = OpHelper::smem_pos(cell_idx);
                    const int2     smem_cell_end   = smem_cell_start + int2(copy(CellCfg::CellSize::value));
                    constexpr int2 clip_size       = OpHelper::output_region.outer_right_bottom_pos();

                    const bool fully_valid = all(smem_cell_end <= clip_size);

                    if (fully_valid) {
                        execute_cell_fast(input_tile, cell, smem_cell_start);
                    } else {
                        execute_cell_slow(input_tile, cell, smem_cell_start);
                    }
                    store_cell_to_tile_clipped(cell, output_tile, smem_cell_start, uint2(clip_size));
                });
                __syncthreads();
            }

        private:
            template<typename CellType, typename PosType>
            __device__ static void execute_cell_fast(const typename OpHelper::TileStorageType& input_tile,
                                                     CellType& cell, const PosType& pos) {
                constexpr uint2 cell_size = copy(CellCfg::CellSize::value);
                backend::sharpen::compute_3x3_sharpen_cell<Weights, cell_size.x, cell_size.y,
                                                           typename OpHelper::TileStorageType, OpHelper::num_channels,
                                                           CellType>(input_tile, pos->x, pos->y, cell);
            }

            template<typename CellType, typename PosType>
            __device__ static void execute_cell_slow(const typename OpHelper::TileStorageType& input_tile,
                                                     CellType& cell, const PosType& pos) {
#pragma unroll
                for (unsigned int py = 0; py < CellCfg::CellSize::Y; ++py) {
                    const int smem_y = pos->y + static_cast<int>(py);
#pragma unroll
                    for (unsigned int px = 0; px < CellCfg::CellSize::X; ++px) {
                        const int          smem_x    = pos->x + static_cast<int>(px);
                        const unsigned int pixel_idx = py * CellCfg::CellSize::X + px;
                        if (copy(OpHelper::output_region).contains(OuterPos(int2 {smem_x, smem_y}))) {
                            float sharpen_result[OpHelper::num_channels];
                            backend::sharpen::compute_3x3_sharpen<Weights, typename OpHelper::TileStorageType,
                                                                  OpHelper::num_channels>(input_tile, smem_x, smem_y,
                                                                                          sharpen_result);
#pragma unroll
                            for (unsigned int c = 0; c < OpHelper::num_channels; ++c) {
                                cell(c, pixel_idx) = sharpen_result[c];
                            }
                        }
                    }
                }
            }
        };

        template<unsigned int J, unsigned int K, interpolation_method Method, typename Layout, typename CellCfg, int SM,
                 int BlockThreads>
        struct op_impl<Resize<J, K, Method>, Layout, CellCfg, SM, BlockThreads> {

            using Operation = Resize<J, K, Method>;
            using OpHelper  = FuncOpImplHelper<Operation, Layout, CellCfg>;

            // Spacing ratio: R = J/K (J input samples -> K output samples)
            static constexpr unsigned int         scale_J = J;
            static constexpr unsigned int         scale_K = K;
            static constexpr interpolation_method method  = Method;

            // Halo is the already-scaled radius from backend filter properties.
            static constexpr unsigned int halo = backend::resize::filter_computed_properties<J, K, Method>::radius_ceil;

            static constexpr region in_region       = make_input_region(Layout::value);
            static constexpr uint2  in_nominal_size = OpHelper::input_nominal_size;

            // Output tile size: N_O = ceil(N_I*K/J - rho_O) with rho_O = 0.5 (symmetric placement).
            // Assumes Layout::base_x and Layout::base_y are divisible by J (constant tile-local rho_O).
            // Precondition for compute_resize_output_nominal_tile: 2 * in_nominal * K >= J (else unsigned underflow).
            static_assert(2u * in_nominal_size.x * K >= J && 2u * in_nominal_size.y * K >= J,
                          "Resize configuration is invalid: tile is too small for the given downscale ratio (J/K). "
                          "Increase TileSize or reduce J/K.");
            static constexpr uint2 out_size = OpHelper::output_nominal_size;

            static constexpr uint2 inter_size = {out_size.x, in_region.size.y};

            // Resize-specific compile-time guards.
            // Large upscale factors can produce output/intermediate tiles that cannot be represented
            // by this shared-memory execution model.
            // When helper is ready move this to operation_database.hpp to generalize.
            static constexpr bool is_valid_output_tile = is_valid_tile_v(out_size, SM, packing_format::internal);
            static constexpr bool is_valid_intermediate_tile =
                is_valid_tile_v(inter_size, SM, packing_format::internal);

            static_assert(
                is_valid_output_tile,
                "Resize configuration is invalid: the derived output tile is not supported for this SM/cell layout. "
                "Reduce TileSize or resize ratio (K/J), or use a staged resize.");
            static_assert(is_valid_intermediate_tile,
                          "Resize configuration is invalid: the derived intermediate tile is not supported for this "
                          "SM/cell layout. Reduce TileSize or resize ratio (K/J), or use a staged resize.");

            using OutputTileStorageType       = NPPDX_STD::tuple_element_t<0, typename OpHelper::outputs>;
            using IntermediateTileStorageType = NPPDX_STD::tuple_element_t<0, typename OpHelper::temp>;

            // Register-based execute (not supported, requires shared memory)
            template<typename T>
            __device__ static void execute(T*, int, int) {
                constexpr bool DependentFalse = !NPPDX_STD::is_same_v<T, T>;
                static_assert(DependentFalse, "Register-based execute not supported for resize operations.");
            }

            // Non-separable resize execute (input tile -> output tile with different sizes)
            __device__ static void execute(const typename OpHelper::TileStorageType& input_tile,
                                           OutputTileStorageType&                    output_tile, int, int) {
                using internal_processing_type = typename OpHelper::TileStorageType::ChannelSliceType::value_type;

                // For a gather we require the output grid
                constexpr auto grid = aligned_cell_grid {out_size, CellCfg::value.size};
                using executor      = cell_executor<NPPDX_MAKE_UINT2D(grid.cell_count())>;

                // Checks
                static_assert(NPPDX_STD::is_same_v<NPPDX_STD::remove_cv_t<internal_processing_type>,
                                                   typename OpHelper::processing_type>,
                              "internal_processing_type must be the same as processing_type");

                // Compute offset from output coordinates to input shared memory coordinates
                // Input tile has halo, output tile starts at (0,0)
                constexpr int2 in_offset = int2(Layout::MemoryHalo::value.left_top);

                constexpr region out_region {int2::zero(), out_size};

                executor::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    const OuterPos cell_smem_pos = get_smem_pos(grid, cell_idx, int2::zero());
                    for (unsigned int py = 0; py < CellCfg::value.size.y; ++py) {
                        for (unsigned int px = 0; px < CellCfg::value.size.x; ++px) {
                            const int2 el_smem_pos = cell_smem_pos + int2(uint2 {px, py});
                            if (out_region.contains(OuterPos(el_smem_pos))) {
                                backend::resize::resize_2d_non_separable<J, K, Method, OpHelper::num_channels>(
                                    input_tile, output_tile, el_smem_pos.x, el_smem_pos.y, in_offset.x, in_offset.y);
                            }
                        }
                    }
                });
                __syncthreads();
            }

            // Double-buffered execute with separable resize (input -> intermediate -> output)
            // Uses O(2*lambda) operations instead of O(lambda^2) - critical for Lanczos on SM100
            // For a separable kernel: k(x,y) = k(x) * k(y)
            __device__ static void execute(const typename OpHelper::TileStorageType& input_tile,
                                           IntermediateTileStorageType&              intermediate_tile,
                                           OutputTileStorageType&                    output_tile, int, int) {
                using internal_processing_type = typename OpHelper::TileStorageType::ChannelSliceType::value_type;


                constexpr int2 in_offset = int2(Layout::MemoryHalo::value.left_top);

                // Grid and executor for intermediate region (horizontal pass)
                // Intermediate region for separable resize (output_width x (base_y + cumulative_halo))
                constexpr uint2 cell_size    = CellCfg::value.size;
                constexpr auto  inter_region = region {int2::zero(), inter_size};
                constexpr auto  inter_grid   = aligned_cell_grid {inter_size, cell_size};
                using inter_executor         = cell_executor<NPPDX_MAKE_UINT2D(inter_grid.cell_count())>;

                // =========== PASS 1: Horizontal Resize ===========
                // Iterate over intermediate buffer using executor
                inter_executor::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    const int2 cell_smem_pos = get_smem_pos(inter_grid, cell_idx, int2::zero());
                    for (unsigned int py = 0; py < inter_grid.cell_size.y; ++py) {
                        for (unsigned int px = 0; px < inter_grid.cell_size.x; ++px) {
                            const int out_x     = cell_smem_pos.x + static_cast<int>(px);
                            const int inter_row = cell_smem_pos.y + static_cast<int>(py);

                            if (inter_region.contains(OuterPos(int2 {out_x, inter_row}))) {
                                // Convert inter_row (0-based) to base-relative row for input gathering
                                const int base_rel_row =
                                    inter_row - static_cast<int>(Layout::CumulativeHalo::value.left_top.y);

                                backend::resize::resize_horizontal_row<J, K, Method, OpHelper::num_channels>(
                                    input_tile, intermediate_tile, base_rel_row, out_x, inter_row, in_offset.x,
                                    in_offset.y);
                            }
                        }
                    }
                });

                __syncthreads();

                // =========== PASS 2: Vertical Resize ===========
                // Each thread processes some (out_x, out_y) pairs
                // Reads from intermediate, writes to output
                constexpr auto   out_grid = aligned_cell_grid {out_size, CellCfg::value.size};
                constexpr region out_region {int2::zero(), out_size};
                using out_executor = cell_executor<NPPDX_MAKE_UINT2D(out_grid.cell_count())>;

                out_executor::template for_each_cell_idx</*Strided=*/true>([&](InnerCellIdx cell_idx) {
                    const int2 cell_smem_pos = get_smem_pos(out_grid, cell_idx, int2::zero());
                    for (unsigned int py = 0; py < CellCfg::value.size.y; ++py) {
                        for (unsigned int px = 0; px < CellCfg::value.size.x; ++px) {
                            const int2 el_smem_pos = cell_smem_pos + int2(uint2 {px, py});
                            if (out_region.contains(OuterPos(el_smem_pos))) {
                                // The offset at output is the resize halo
                                constexpr int inter_offset_y =
                                    static_cast<int>(Layout::CumulativeHalo::value.left_top.y);
                                backend::resize::resize_vertical_pixel<J, K, Method, OpHelper::num_channels>(
                                    intermediate_tile, output_tile, el_smem_pos.x, el_smem_pos.y, inter_offset_y);
                            }
                        }
                    }
                });


                __syncthreads();
            }
        };

    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_DATABASE_FUNCTION_OPERATION_IMPLEMENTATION_HPP
