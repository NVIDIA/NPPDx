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

#ifndef NPPDX_DETAIL_BACKEND_RESIZE_OPERATIONS_HPP
#define NPPDX_DETAIL_BACKEND_RESIZE_OPERATIONS_HPP

#include "nppdx/detail/backend/constexpr_math.hpp"
#include "nppdx/detail/backend/resize_geometry.hpp"
#include "nppdx/utils.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/types.hpp"
#include "nppdx/operators/function.hpp"

#include "nppdx/detail/config.hpp"
#include NPPDX_STD_INCLUDE_CMATH
#include NPPDX_STD_INCLUDE_UTILITY


namespace nppdx {
    namespace detail {
        namespace backend {
            namespace resize {

                // "taps" refers to the lowest even integer >= ceil(2 * radius).
                // Nearest-neighbor is a special case with taps = lambda = 1.

                template<unsigned int J, unsigned int K, interpolation_method Method>
                struct filter_computed_properties {
                    static constexpr filter_props props       = get_filter_props(Method);
                    static constexpr unsigned int mu_num      = props.mu_num;
                    static constexpr unsigned int mu_den      = props.mu_den;
                    static constexpr unsigned int radius_num  = props.radius_num(J, K);
                    static constexpr unsigned int radius_den  = props.radius_den(J, K);
                    static constexpr unsigned int radius_ceil = props.radius_ceil(J, K);
                    static constexpr unsigned int taps        = props.taps(J, K);
                };

                // =============================================================================
                // Filter kernel evaluation functions
                // =============================================================================

                // Sinc function: sin(pi*x)/(pi*x), with sinc(0) = 1
                __device__ __forceinline__ float sinc(float x) {
                    if (fabsf(x) < 1e-6f) {
                        return 1.0f;
                    }
                    const float pi_x = 3.14159265358979323846f * x;
                    return __sinf(pi_x) / pi_x;
                }

                // Nearest neighbor: just returns 1 for the closest pixel
                __device__ __forceinline__ float kernel_nearest(float x) {
                    return (fabsf(x) <= 0.5f) ? 1.0f : 0.0f;
                }

                // Bilinear (triangle/tent): 1 - |x| for |x| < 1
                __device__ __forceinline__ float kernel_bilinear(float x) {
                    const float ax = fabsf(x);
                    return (ax < 1.0f) ? (1.0f - ax) : 0.0f;
                }

                // Bicubic (Catmull-Rom / Mitchell-Netravali with B=0, C=0.5)
                __device__ __forceinline__ float kernel_bicubic(float x) {
                    const float ax = fabsf(x);
                    if (ax < 1.0f) {
                        return (1.5f * ax - 2.5f) * ax * ax + 1.0f;
                    } else if (ax < 2.0f) {
                        return ((-0.5f * ax + 2.5f) * ax - 4.0f) * ax + 2.0f;
                    }
                    return 0.0f;
                }

                // Lanczos-3: sinc(x) * sinc(x/3) for |x| < 3
                __device__ __forceinline__ float kernel_lanczos3(float x) {
                    const float ax = fabsf(x);
                    if (ax < 3.0f) {
                        return sinc(x) * sinc(x / 3.0f);
                    }
                    return 0.0f;
                }

                // Dispatch to appropriate 1D kernel based on method
                template<interpolation_method Method>
                __device__ __forceinline__ float evaluate_1d_kernel(float x) {
                    if constexpr (Method == interpolation_method::nearest) {
                        return kernel_nearest(x);
                    } else if constexpr (Method == interpolation_method::bilinear) {
                        return kernel_bilinear(x);
                    } else if constexpr (Method == interpolation_method::bicubic) {
                        return kernel_bicubic(x);
                    } else {
                        return kernel_lanczos3(x);
                    }
                }

                template<interpolation_method Method>
                __device__ __forceinline__ float evaluate_separable_2d_kernel(float x, float y) {
                    return evaluate_1d_kernel<Method>(x) * evaluate_1d_kernel<Method>(y);
                }

                template<interpolation_method Method>
                __device__ __forceinline__ float evaluate_2d_kernel(float x, float y) {
                    if constexpr (Method == interpolation_method::lanczos3) {
                        // Use radial distance for Lanczos in the non-separable path.
                        return evaluate_1d_kernel<Method>(sqrtf(x * x + y * y));
                    } else {
                        // Fallback to separable composition for methods without a dedicated 2D kernel.
                        return evaluate_separable_2d_kernel<Method>(x, y);
                    }
                }

                // =============================================================================
                // Phase and index calculations (integer arithmetic where possible)
                // From the math doc: theta = r1_plus((n_O + rho_O)R - 1/2)
                // For rational R = J/K with rho_O = 1/2: phase_m = (2*n_O*J + J - K) mod 2K / 2K
                // =============================================================================

                // Compute phase index m in [0, K) for output pixel n_O
                // Using rho_O = 1/2 (symmetric placement)
                template<unsigned int J, unsigned int K>
                __device__ __forceinline__ unsigned int compute_phase_index(int n_O) {
                    // theta = r1_plus((n_O + 0.5) * J/K - 0.5)
                    // With integer math: phase_num = (2*n_O*J + J - K) mod 2K
                    // m = phase_num / 2  (since phases are m/K for m in [0,K))
                    int phase_num = (2 * n_O * static_cast<int>(J) + static_cast<int>(J) - static_cast<int>(K));
                    // Proper modulo for negative numbers
                    phase_num =
                        ((phase_num % (2 * static_cast<int>(K))) + 2 * static_cast<int>(K)) % (2 * static_cast<int>(K));
                    return static_cast<unsigned int>(phase_num / 2);
                }

                // Compute phase theta in [0, 1) for output pixel n_O.
                // theta = frac((n_O + 0.5) * J/K - 0.5)
                template<unsigned int J, unsigned int K>
                __device__ __forceinline__ float compute_phase(int n_O) {
                    float val =
                        (static_cast<float>(n_O) + 0.5f) * (static_cast<float>(J) / static_cast<float>(K)) - 0.5f;
                    return val - floorf(val);
                }

                // Compute center input index n_Ic for output pixel n_O.
                // n_Ic = floor((n_O + 0.5) * J/K - 0.5)
                template<unsigned int J, unsigned int K>
                __device__ __forceinline__ int compute_center_input_index(int n_O) {
                    // Using integer arithmetic: n_c = floor((2*n_O*J + J - K) / (2*K))
                    int numerator = (2 * n_O + 1) * static_cast<int>(J) - static_cast<int>(K);
                    return (numerator >= 0) ? (numerator / (2 * static_cast<int>(K)))
                                            : ((numerator - 2 * static_cast<int>(K) + 1) / (2 * static_cast<int>(K)));
                }

                // Compute and return the (n_Ic, theta) pair from n_O.
                template<unsigned int J, unsigned int K>
                __device__ __forceinline__ NPPDX_STD::pair<int, float> compute_center_index_and_phase(int n_O) {
                    return {compute_center_input_index<J, K>(n_O), compute_phase<J, K>(n_O)};
                }

                // Delta term from the resampling formulation.
                // For even lambda: Delta = 2
                // For odd  lambda: Delta = 1 if theta < 0.5, else 3
                template<unsigned int Lambda>
                __device__ __forceinline__ int compute_delta(float theta) {
                    if constexpr (Lambda % 2 == 0) {
                        return 2;
                    } else {
                        return (theta < 0.5f) ? 1 : 3;
                    }
                }

                // Compute starting input index n_I_minus from center index and phase.
                // n_I_minus = n_c - (lambda - Delta(theta)) / 2
                template<unsigned int Lambda>
                __device__ __forceinline__ int compute_input_start_index(int n_Ic, float theta) {
                    const int delta = compute_delta<Lambda>(theta);
                    return n_Ic + (delta - static_cast<int>(Lambda)) / 2;
                }

                // Compute the x position (distance) for kernel evaluation
                // x_theta(j) = theta + L - j where L positions the kernel
                template<unsigned int Lambda>
                __device__ __forceinline__ float compute_kernel_sample_position(float theta, int j) {
                    const int delta = compute_delta<Lambda>(theta);
                    // L = (lambda - delta) / 2
                    float L = static_cast<float>(static_cast<int>(Lambda) - delta) / 2.0f;

                    return theta + L - static_cast<float>(j);
                }

                // =============================================================================
                // Constexpr Q tensor computation
                // =============================================================================

                // Compute kernel position for phase m, tap j
                template<unsigned int J, unsigned int K, unsigned int Lambda>
                constexpr __device__ __forceinline__ float compute_kernel_position_constexpr(unsigned int m,
                                                                                             unsigned int j) {
                    constexpr unsigned int epsilon = (J + K) % 2;
                    float                  theta   = static_cast<float>(2 * m + epsilon) / static_cast<float>(2 * K);

                    // L = (lambda - delta) / 2, where delta = 2 for even lambda
                    int   delta = (Lambda % 2 == 0) ? 2 : ((theta < 0.5f) ? 1 : 3);
                    float L     = static_cast<float>(static_cast<int>(Lambda) - delta) / 2.0f;

                    return theta + L - static_cast<float>(j);
                }

                // Compute single Q weight
                template<unsigned int J, unsigned int K, interpolation_method Method>
                constexpr __device__ __forceinline__ float compute_Q_weight(unsigned int m, unsigned int j) {
                    constexpr unsigned int lambda = filter_computed_properties<J, K, Method>::taps;
                    float                  x      = compute_kernel_position_constexpr<J, K, lambda>(m, j);

                    // Dispatch to appropriate kernel
                    if constexpr (Method == interpolation_method::nearest) {
                        return (constexpr_math::const_abs(x) <= 0.5f) ? 1.0f : 0.0f;
                    } else if constexpr (Method == interpolation_method::lanczos3) {
                        return constexpr_math::lanczos3(x);
                    } else if constexpr (Method == interpolation_method::bilinear) {
                        return constexpr_math::bilinear(x);
                    } else if constexpr (Method == interpolation_method::bicubic) {
                        return constexpr_math::bicubic(x);
                    } else {
                        return 0.0f;
                    }
                }

                // =============================================================================
                // QWeights with constexpr-computed values
                // =============================================================================

                template<unsigned int J, unsigned int K, interpolation_method Method>
                using QWeightsArray = Array<float, int(K* filter_computed_properties<J, K, Method>::taps)>;

                template<unsigned int J, unsigned int K, interpolation_method Method>
                struct QWeightsGenerator {
                    static constexpr unsigned int lambda = filter_computed_properties<J, K, Method>::taps;
                    static constexpr unsigned int V      = K * lambda;
                    static constexpr __device__ __forceinline__ auto make_data() {
                        Array<float, int(V)> arr {};
                        for (unsigned int phase_idx = 0; phase_idx < K; ++phase_idx) {
                            for (unsigned int tap_idx = 0; tap_idx < lambda; ++tap_idx) {
                                arr[int(phase_idx * lambda + tap_idx)] =
                                    compute_Q_weight<J, K, Method>(phase_idx, tap_idx);
                            }
                        }
                        return arr;
                    }
                };

                template<typename ProcessingType, unsigned int Lambda, typename DataArray>
                constexpr __device__ __forceinline__ ProcessingType lookup(const DataArray& data, unsigned int m,
                                                                           unsigned int j) {
                    return data[int(m * Lambda + j)];
                }

                // Tap loop helpers used to control unrolling policy by kernel size.
                template<unsigned int Lambda, typename Body>
                __device__ __forceinline__ void taps_unrolled(Body&& body) {
#pragma unroll
                    for (unsigned int j = 0; j < Lambda; ++j) {
                        body(j);
                    }
                }

                template<unsigned int Lambda, typename Body>
                __device__ __forceinline__ void taps_plain(Body&& body) {
                    for (unsigned int j = 0; j < Lambda; ++j) {
                        body(j);
                    }
                }

                template<unsigned int Lambda, typename Body>
                __device__ __forceinline__ void taps_for_lambda(Body&& body) {
                    // Due to resource constraints, we avoid unrolling for lambda > 9.
                    if constexpr (Lambda <= 9) {
                        taps_unrolled<Lambda>(static_cast<Body&&>(body));
                    } else {
                        taps_plain<Lambda>(static_cast<Body&&>(body));
                    }
                }


                // =============================================================================
                // 1D Gather Resize Operation
                // Computes: v_O(n_O) = sum_j k(j) * v_I(n_I_minus + j) for j in [0, lambda)
                // =============================================================================

                template<bool Vertical, unsigned int J, unsigned int K, interpolation_method Method,
                         typename InputSliceType, typename ProcessingType = float>
                __device__ __forceinline__ ProcessingType gather_1d(const InputSliceType& input_channel, int out_x,
                                                                    int out_y, int in_offset_x, int in_offset_y) {
                    constexpr unsigned int             lambda  = filter_computed_properties<J, K, Method>::taps;
                    const int                          out_pos = Vertical ? out_y : out_x;
                    const auto                         center_and_phase = compute_center_index_and_phase<J, K>(out_pos);
                    const int                          n_Ic             = center_and_phase.first;
                    const float                        theta            = center_and_phase.second;
                    const unsigned int                 m                = compute_phase_index<J, K>(out_pos);
                    const int                          n_I_start = compute_input_start_index<lambda>(n_Ic, theta);
                    const QWeightsArray<J, K, Method>& data      = QWeightsGenerator<J, K, Method>::make_data();

                    ProcessingType sum = ProcessingType(0), weight_sum = ProcessingType(0);

#pragma unroll
                    for (unsigned int j = 0; j < lambda; ++j) {
                        const float weight =
                            lookup<ProcessingType, lambda>(data, m, j); // J/K/Method inferred from data type.
                        const ProcessingType val =
                            Vertical
                                ? input_channel(n_I_start + static_cast<int>(j) + in_offset_y, out_x + in_offset_x)
                                : input_channel(out_y + in_offset_y, n_I_start + static_cast<int>(j) + in_offset_x);
                        sum += weight * val;
                        weight_sum += weight;
                    }

                    return (weight_sum > ProcessingType(1e-6)) ? (sum / weight_sum) : sum;
                }

                template<unsigned int J, unsigned int K, interpolation_method Method, typename InputSliceType,
                         typename ProcessingType = float>
                __device__ __forceinline__ ProcessingType
                gather_1d_horizontal(const InputSliceType& input_channel,
                                     int                   out_x,        // output x coordinate
                                     int                   base_rel_row, // row coordinate relative to base row 0
                                     int in_offset_x, // memory_halo.left_top.x - maps base-relative to physical
                                     int in_offset_y  // memory_halo.left_top.y - maps base-relative to physical
                ) {
                    return gather_1d<false, J, K, Method, InputSliceType, ProcessingType>(
                        input_channel, out_x, base_rel_row, in_offset_x, in_offset_y);
                }

                template<unsigned int J, unsigned int K, interpolation_method Method, typename InputSliceType,
                         typename ProcessingType = float>
                __device__ __forceinline__ ProcessingType gather_1d_vertical(const InputSliceType& input_channel,
                                                                             int out_x, int out_y, int in_offset_y) {
                    return gather_1d<true, J, K, Method, InputSliceType, ProcessingType>(input_channel, out_x, out_y, 0,
                                                                                         in_offset_y);
                }

                // =============================================================================
                // Separable 2-Pass Resize: O(2*lambda) instead of O(lambda^2)
                // Pass 1: Horizontal resize (input -> intermediate)
                // Pass 2: Vertical resize (intermediate -> output)
                //
                // Lambda choice:
                // We intentionally round lambda up to the nearest even value for non-nearest methods.
                // This can add a small amount of extra work for some J/K ratios, but avoids the
                // odd-lambda + phase-0.5 corner case and keeps the symmetry formulas/tables simple.
                // Nearest-neighbor is a special case: it stays canonical with lambda = 1.
                // =============================================================================

                // Pass 1: Horizontal resize for a single row
                // Processes one row from input, writes horizontally resized values to intermediate
                template<unsigned int J, unsigned int K, interpolation_method Method, unsigned int NumChannels,
                         typename InputTileStorageType, typename IntermediateTileStorageType,
                         typename ProcessingType = float>
                __device__ __forceinline__ void resize_horizontal_row(
                    const InputTileStorageType& input, IntermediateTileStorageType& intermediate,
                    int base_rel_row, // row coordinate relative to base row 0 (same convention as n_I_start for columns)
                    int out_x,        // output x position to compute
                    int inter_row,    // row index in intermediate buffer (0-based)
                    int in_offset_x,  // memory_halo.left_top.x - maps base-relative column to physical
                    int in_offset_y   // memory_halo.left_top.y - maps base-relative row to physical
                ) {
#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        ProcessingType val = gather_1d_horizontal<J, K, Method>(input.channel(c), out_x, base_rel_row,
                                                                                in_offset_x, in_offset_y);
                        intermediate.channel(c)(inter_row, out_x) = val;
                    }
                }

                // Pass 2: Vertical resize for a single output pixel
                // Reads from intermediate (already horizontally resized), writes to output
                template<unsigned int J, unsigned int K, interpolation_method Method, unsigned int NumChannels,
                         typename IntermediateTileStorageType, typename OutputTileStorageType,
                         typename ProcessingType = float>
                __device__ __forceinline__ void resize_vertical_pixel(
                    const IntermediateTileStorageType& intermediate, OutputTileStorageType& output, int out_x,
                    int out_y,
                    int inter_offset_y // offset to map virtual row 0 to real input row index in the input tile
                ) {
#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        ProcessingType val =
                            gather_1d_vertical<J, K, Method>(intermediate.channel(c), out_x, out_y, inter_offset_y);
                        output.channel(c)(out_y, out_x) = val;
                    }
                }


                // =============================================================================
                // 2D Non-Separable Resize
                // Uses the same lambda policy described in the separable resize block above.
                // For methods without a dedicated 2D form, this falls back to separable composition.
                // =============================================================================

                template<unsigned int J, unsigned int K, interpolation_method Method, unsigned int NumChannels,
                         typename InputTileStorageType, typename OutputTileStorageType, typename ProcessingType = float>
                __device__ void resize_2d_non_separable(
                    const InputTileStorageType& input, OutputTileStorageType& output, int out_x, int out_y,
                    int in_halo_x, // input tile halo (offset from output to input coordinates)
                    int in_halo_y) {
                    constexpr unsigned int lambda = filter_computed_properties<J, K, Method>::taps;

                    const auto center_and_phase_x = compute_center_index_and_phase<J, K>(out_x);
                    const auto center_and_phase_y = compute_center_index_and_phase<J, K>(out_y);
                    const int  n_Ix_start =
                        compute_input_start_index<lambda>(center_and_phase_x.first, center_and_phase_x.second);
                    const int n_Iy_start =
                        compute_input_start_index<lambda>(center_and_phase_y.first, center_and_phase_y.second);

#pragma unroll
                    for (unsigned int c = 0; c < NumChannels; ++c) {
                        ProcessingType result     = ProcessingType(0);
                        ProcessingType weight_sum = ProcessingType(0);

                        taps_for_lambda<lambda>([&](unsigned int jy) {
                            const int   in_y = n_Iy_start + static_cast<int>(jy);
                            const float y_pos =
                                compute_kernel_sample_position<lambda>(center_and_phase_y.second, static_cast<int>(jy));

                            taps_for_lambda<lambda>([&](unsigned int jx) {
                                const int   in_x   = n_Ix_start + static_cast<int>(jx);
                                const float x_pos  = compute_kernel_sample_position<lambda>(center_and_phase_x.second,
                                                                                            static_cast<int>(jx));
                                const float weight = evaluate_2d_kernel<Method>(x_pos, y_pos);
                                const ProcessingType val = input.channel(c)(in_y + in_halo_y, in_x + in_halo_x);
                                result += weight * val;
                                weight_sum += weight;
                            });
                        });

                        output.channel(c)(out_y, out_x) =
                            (weight_sum > ProcessingType(1e-6)) ? (result / weight_sum) : result;
                    }
                }

            } // namespace resize
        } // namespace backend
    } // namespace detail
} // namespace nppdx

#endif // NPPDX_DETAIL_BACKEND_RESIZE_OPERATIONS_HPP
