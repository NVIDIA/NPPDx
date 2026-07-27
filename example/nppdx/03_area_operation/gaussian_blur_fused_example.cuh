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

// Shared single-radius driver for the fused Gaussian blur + RGB->YUV422P examples.
//
// Background:
// (a) Radius encoding: RadiusTenths is exact (radius = RadiusTenths/10.f, no rounding). NPPDx uses odd-length
//     separable discrete Gaussian FIRs with kernel width 3..21 by band
//     (e.g. 3x3 for R<1.0 on the tenths grid, 5x5 for 1.0<=R<=2.0, up to 21x21 for large R).
//
// (b) NPP reference: nppiFilterGaussAdvancedBorder accepts a user 1D kernel; we pass the same normalized
//     discrete Gaussian taps as NPPDx (GaussianBlur::fir_1d_weights), then BT.601 YUV422 - so NPP and NPPDx apply
//     the same separable blur on RGB before conversion.
//
// (c) Halo grows with kernel width (half the 1D tap count, floored). Cellular path uses register separable FIR when
//     the cell is fully inside the ROI; edge pixels use full 2D FIR.
//
// (d) Outputs: standard writes cow_yuv422p_gaussian_fused_nppdx_rI_F.yuv422; wide inserts "wide_" before
//     rI_F (I_F = integer.tenths). When NPP comparison runs, a matching gaussian_twostep_npp file is written too.

#ifndef NPPDX_EXAMPLE_03_GAUSSIAN_BLUR_FUSED_EXAMPLE_CUH
#define NPPDX_EXAMPLE_03_GAUSSIAN_BLUR_FUSED_EXAMPLE_CUH

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../common.hpp"
#include "../common/gaussian_blur_npp_reference.cuh"

namespace G = common::gaussian_blur_npp_reference;

namespace common {
    namespace gaussian_blur_example {

        // Filename tag for radius: RadiusTenths 5 -> "r0_5" (0.5), 100 -> "r10_0" (10.0).
        inline std::string gaussian_radius_output_tag(int radius_tenths) {
            const int whole = radius_tenths / 10;
            const int frac  = radius_tenths % 10;
            return std::string("r") + std::to_string(whole) + "_" + std::to_string(frac);
        }

        // Run the fused Gaussian-blur + RGB->YUV422P example at a single RadiusTenths.
        // Tabular=false -> verbose single-shot output.
        // Tabular=true  -> one table row per call.
        // NppRefRadiusTenths != RadiusTenths -> gradient probe (NPP ref uses a different sigma);
        // such rows are not scored into pass/fail counts.
        template<int Arch, int RadiusTenths = 20, bool Tabular = false, int NppRefRadiusTenths = RadiusTenths,
                 typename GaussianFirKernelProvider = G::gaussian_sigma2_fir_kernel_provider,
                 nppdx::gaussian_tail_width TailWidth = nppdx::gaussian_tail_width::standard>
        int run_single_radius() {
            static_assert(RadiusTenths >= 1 && RadiusTenths <= 100, "RadiusTenths 1..100");
            static_assert(NppRefRadiusTenths >= 1 && NppRefRadiusTenths <= 100, "NppRefRadiusTenths 1..100");
            constexpr bool npp_ref_gradient_probe = (NppRefRadiusTenths != RadiusTenths);
            constexpr bool wide = (TailWidth == nppdx::gaussian_tail_width::wide);

            constexpr float radius  = static_cast<float>(RadiusTenths) / 10.f;
            constexpr bool  verbose = !Tabular;

            constexpr unsigned int warm_up_runs = 10;
            constexpr unsigned int runs         = 100;

            constexpr unsigned int width    = 312;
            constexpr unsigned int height   = 376;
            constexpr unsigned int channels = 3;
            constexpr size_t       rgb_size = width * height * channels;

            constexpr size_t y_size   = width * height;
            constexpr size_t uv_size  = (width / 2) * height;
            constexpr size_t yuv_size = y_size + 2 * uv_size;

            if (verbose) {
                std::cout << "\n>>> Radius " << radius << " (RadiusTenths=" << RadiusTenths << ") <<<" << std::endl;
                if (npp_ref_gradient_probe) {
                    std::cout << "NPP separable ref: R=" << (static_cast<float>(NppRefRadiusTenths) / 10.f)
                              << " (tenths " << NppRefRadiusTenths << ") - gradient probe vs NPPDx above\n";
                }
                std::cout << "=== NPPDx Fused Gaussian Blur + RGB to YUV422P Example (SM" << Arch
                          << ") ===" << std::endl;
                std::cout << "Image: " << width << "x" << height << " RGB24 (" << rgb_size << " bytes) -> YUV422P ("
                          << yuv_size << " bytes)" << std::endl;
                std::cout << "Gaussian radius: " << radius << ", tail=" << (wide ? "wide" : "standard")
                          << ", path=FIR separable (cellular or per-pixel)" << std::endl;
            }

            std::vector<uint8_t> h_input = common::load_or_generate_rgb_test_data(width, height);

            common::DevBuf d_input(rgb_size);
            common::DevBuf d_output(yuv_size);
            CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_input.data(), rgb_size, cudaMemcpyHostToDevice));

            auto nppdx_result = G::run_nppdx_fused<Arch, channels, RadiusTenths, TailWidth>(
                d_input.d_buf, d_output.d_buf, width, height, yuv_size, warm_up_runs, runs, verbose);

#ifdef NPPDX_EXAMPLE_HAS_NPP
            constexpr bool run_npp_reference = true;
#else
            constexpr bool run_npp_reference = false;
#endif
            constexpr bool npp_pixel_compare = run_npp_reference;
            auto           npp_result        = run_npp_reference
                                                   ? G::run_npp_two_step<NppRefRadiusTenths,
                                                                         GaussianFirKernelProvider, TailWidth>(
                                                         d_input.d_buf, width, height, yuv_size, warm_up_runs, runs,
                                                         verbose)
                                                   : common::nppdx_results<uint8_t> {std::vector<uint8_t>(), 0};

            constexpr int    max_err_limit = G::npp_compare_max_abs_error;
            constexpr double avg_err_limit = G::npp_compare_avg_abs_error;

            bool               passed      = true;
            float              speedup     = 0.f;
            int                max_diff    = 0;
            double             avg_diff    = 0.;
            size_t             error_count = 0;
            common::ErrorStats y_plane {};
            common::ErrorStats u_plane {};
            common::ErrorStats v_plane {};
            if (run_npp_reference && !npp_result.output.empty() && npp_result.avg_time_in_ms > 0.f &&
                nppdx_result.avg_time_in_ms > 0.f) {
                speedup = npp_result.avg_time_in_ms / nppdx_result.avg_time_in_ms;
            }
            if (!npp_result.output.empty()) {
                if (npp_pixel_compare) {
                    auto error_stats = common::check_error_detailed(
                        nppdx_result.output.data(), npp_result.output.data(), yuv_size, G::npp_compare_error_threshold,
                        max_err_limit, avg_err_limit, 1, 1, false, false);
                    passed      = error_stats.passed;
                    max_diff    = static_cast<int>(error_stats.max_abs_error);
                    avg_diff    = error_stats.avg_abs_error;
                    error_count = error_stats.error_count;
                }

                if (npp_pixel_compare && !npp_ref_gradient_probe) {
                    y_plane = common::check_error_detailed(nppdx_result.output.data(), npp_result.output.data(), y_size,
                                                           G::npp_compare_error_threshold, max_err_limit, avg_err_limit,
                                                           1, 1, false, false);
                    u_plane = common::check_error_detailed(
                        nppdx_result.output.data() + y_size, npp_result.output.data() + y_size, uv_size,
                        G::npp_compare_error_threshold, max_err_limit, avg_err_limit, 1, 1, false, false);
                    v_plane = common::check_error_detailed(
                        nppdx_result.output.data() + y_size + uv_size, npp_result.output.data() + y_size + uv_size,
                        uv_size, G::npp_compare_error_threshold, max_err_limit, avg_err_limit, 1, 1, false, false);
                }

                if (npp_pixel_compare && !Tabular && !npp_ref_gradient_probe) {
                    std::cout << "\n--- NPP (sigma=R separable ref) vs NPPDx: speed and error ---" << std::endl;
                    std::cout << "Y plane:  max_diff=" << y_plane.max_abs_error
                              << ", avg_diff=" << y_plane.avg_abs_error << ", mismatches=" << y_plane.error_count << "/"
                              << y_size << std::endl;
                    std::cout << "U plane:  max_diff=" << u_plane.max_abs_error
                              << ", avg_diff=" << u_plane.avg_abs_error << ", mismatches=" << u_plane.error_count << "/"
                              << uv_size << std::endl;
                    std::cout << "V plane:  max_diff=" << v_plane.max_abs_error
                              << ", avg_diff=" << v_plane.avg_abs_error << ", mismatches=" << v_plane.error_count << "/"
                              << uv_size << std::endl;
                    std::cout << "Overall:  max_diff=" << max_diff << ", avg_diff=" << std::setprecision(4) << avg_diff
                              << ", mismatches=" << error_count << "/" << yuv_size << " (limits max=" << max_err_limit
                              << " avg=" << avg_err_limit << ")" << std::endl;
                    std::cout << "Speed:    NPP " << std::fixed << std::setprecision(4) << npp_result.avg_time_in_ms
                              << " ms, NPPDx " << nppdx_result.avg_time_in_ms << " ms, speedup " << speedup << "x"
                              << std::endl;
                }
            }

            if (Tabular) {
                const char* path_str   = wide ? "FIR-wide" : "FIR";
                std::string radius_col = std::to_string(RadiusTenths / 10) + "." + std::to_string(RadiusTenths % 10);
                if (npp_ref_gradient_probe) {
                    radius_col += "|";
                    radius_col +=
                        std::to_string(NppRefRadiusTenths / 10) + "." + std::to_string(NppRefRadiusTenths % 10);
                }
                std::cout << std::right << std::setw(10) << radius_col << "  " << std::left << std::setw(10) << path_str
                          << std::right << std::fixed << std::setprecision(4) << std::setw(9)
                          << nppdx_result.avg_time_in_ms;
                if (run_npp_reference && !npp_result.output.empty()) {
                    std::cout << std::setw(9) << npp_result.avg_time_in_ms << std::setw(8) << speedup << "x";
                    if (npp_pixel_compare) {
                        std::cout << std::defaultfloat;
                        std::cout << "  " << std::left;
                        if (npp_ref_gradient_probe) {
                            std::cout << "GRAD" << std::setprecision(4) << "  max=" << max_diff << " avg=" << avg_diff;
                        } else {
                            std::cout << (passed ? "PASS" : "FAIL") << std::right << std::setw(4) << max_diff
                                      << std::setw(7) << std::setprecision(4) << avg_diff << std::setw(5)
                                      << y_plane.max_abs_error << std::setw(5) << u_plane.max_abs_error << std::setw(5)
                                      << v_plane.max_abs_error << std::setw(7) << std::setprecision(4)
                                      << y_plane.avg_abs_error << std::setw(7) << u_plane.avg_abs_error << std::setw(7)
                                      << v_plane.avg_abs_error;
                        }
                    }
                } else if (run_npp_reference) {
                    std::cout << std::setw(9) << "-" << std::setw(8) << "-" << std::left << "  (NPP run failed?)";
                } else {
                    std::cout << std::setw(9) << "-" << std::setw(8) << "-" << std::left << "  (no NPP in build)";
                }
                std::cout << std::endl;
            }

            {
                const std::string rtag        = gaussian_radius_output_tag(RadiusTenths);
                const std::string mode_tag    = wide ? "wide_" : "";
                const std::string fname_nppdx =
                    "cow_yuv422p_gaussian_fused_nppdx_" + mode_tag + rtag + ".yuv422";
                constexpr bool    list_output = true;
                common::write_output_file(fname_nppdx.c_str(),
                                          reinterpret_cast<const uint8_t*>(nppdx_result.output.data()), yuv_size,
                                          list_output);
                if (run_npp_reference && !npp_result.output.empty()) {
                    const std::string rtag_npp  = gaussian_radius_output_tag(NppRefRadiusTenths);
                    const std::string fname_npp =
                        "cow_yuv422p_gaussian_twostep_npp_" + mode_tag + rtag_npp + ".yuv422";
                    common::write_output_file(fname_npp.c_str(),
                                              reinterpret_cast<const uint8_t*>(npp_result.output.data()), yuv_size,
                                              list_output);
                }
            }

            if (verbose)
                std::cout << "\nResult: "
                          << (npp_ref_gradient_probe ? "GRAD (not scored)" : (passed ? "PASSED" : "FAILED"))
                          << std::endl;
            std::cout.flush();
            if (npp_ref_gradient_probe)
                return 0;
            return passed ? 0 : 1;
        }

    } // namespace gaussian_blur_example
} // namespace common

#endif // NPPDX_EXAMPLE_03_GAUSSIAN_BLUR_FUSED_EXAMPLE_CUH
