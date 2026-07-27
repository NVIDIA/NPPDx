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

// Host-side NPP reference and comparison vs NPPDx fused Gaussian blur + YUV422P.
// GPU fused launcher lives next to the example: ../03_area_operation/gaussian_blur_fused_gpu.cuh

#ifndef NPPDX_EXAMPLE_COMMON_GAUSSIAN_BLUR_NPP_REFERENCE_CUH
#define NPPDX_EXAMPLE_COMMON_GAUSSIAN_BLUR_NPP_REFERENCE_CUH

#include <nppdx.hpp>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../03_area_operation/gaussian_blur_fused_gpu.cuh"

#include "../common.hpp"

#ifdef NPPDX_EXAMPLE_HAS_NPP
#    include <nppi.h>
#endif

namespace common {
    namespace gaussian_blur_npp_reference {

        // NPPDx vs separable NPP using the same 1D taps. Residual error is mostly the YUV path.
        inline constexpr int    npp_compare_max_abs_error   = 32;
        inline constexpr double npp_compare_avg_abs_error   = 3.0;
        inline constexpr int    npp_compare_error_threshold = 2;

        template<int RadiusTenths>
        void set_gaussian_fir_kernel(float (&kernel)[21]) {
            static_assert(RadiusTenths == 20, "The public NPP reference provides taps for sigma=2.0");
            constexpr float weights[] = {0.152469144f, 0.221841296f, 0.251379121f, 0.221841296f, 0.152469144f};
            for (unsigned int i = 0; i < 5; ++i)
                kernel[i] = weights[i];
        }

        struct gaussian_sigma2_fir_kernel_provider {
            static constexpr nppdx::gaussian_tail_width tail_width = nppdx::gaussian_tail_width::standard;

            template<int RadiusTenths>
            static void set(float (&kernel)[21]) {
                set_gaussian_fir_kernel<RadiusTenths>(kernel);
            }
        };

        // NPP: FilterGaussAdvancedBorder using the selected example's Gaussian taps,
        // then RGB to full-range BT.601 YCbCr422P.
        // GaussianFirKernelProvider::tail_width must match TailWidth: nTaps comes from TailWidth while
        // coefficients come from the provider; a mismatch would send a wrong-sized / zero-padded kernel to NPP.
        template<int RadiusTenths, typename GaussianFirKernelProvider = gaussian_sigma2_fir_kernel_provider,
                 nppdx::gaussian_tail_width TailWidth = nppdx::gaussian_tail_width::standard>
        nppdx_results<uint8_t> run_npp_two_step(uint8_t* d_input, const unsigned int width, const unsigned int height,
                                                const size_t yuv_size, unsigned int warm_up_runs, unsigned int runs,
                                                bool verbose = true) {
            static_assert(GaussianFirKernelProvider::tail_width == TailWidth,
                          "GaussianFirKernelProvider::tail_width must match TailWidth (nTaps vs filled coeffs)");
#ifdef NPPDX_EXAMPLE_HAS_NPP
            if (verbose)
                std::cout << "\nRunning NPP two-step (Gaussian blur then convert)..." << std::endl;

            constexpr unsigned int channels = 3;
            constexpr unsigned int nTaps = nppdx::GaussianBlur<RadiusTenths, TailWidth>::value.kernel_size().x;
            static_assert(nTaps <= 21u, "NPP reference kernel buffer sized for up to 21 taps");

            int            blurred_pitch = width * channels;
            common::DevBuf d_blurred_rgb(static_cast<size_t>(blurred_pitch) * height);

            int            y_pitch = width;
            int            u_pitch = (width + 1) / 2;
            int            v_pitch = (width + 1) / 2;
            common::DevBuf d_y(static_cast<size_t>(y_pitch) * height);
            common::DevBuf d_u(static_cast<size_t>(u_pitch) * height);
            common::DevBuf d_v(static_cast<size_t>(v_pitch) * height);

            NppiSize  roi    = {static_cast<int>(width), static_cast<int>(height)};
            NppiPoint offset = {0, 0};

            float h_kernel[21] = {};
            GaussianFirKernelProvider::template set<RadiusTenths>(h_kernel);

            common::DevBuf d_kernel_buf(nTaps * sizeof(Npp32f));
            Npp32f*        d_kernel = reinterpret_cast<Npp32f*>(d_kernel_buf.d_buf);
            CUDA_CHECK_AND_EXIT(cudaMemcpy(d_kernel, h_kernel, nTaps * sizeof(Npp32f), cudaMemcpyHostToDevice));

            NppStreamContext streamContext = {};
            int              device        = 0;
            cudaGetDevice(&device);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);

            streamContext.hStream                            = cudaStream_t(0);
            streamContext.nCudaDeviceId                      = device;
            streamContext.nMultiProcessorCount               = prop.multiProcessorCount;
            streamContext.nMaxThreadsPerMultiProcessor       = prop.maxThreadsPerMultiProcessor;
            streamContext.nMaxThreadsPerBlock                = prop.maxThreadsPerBlock;
            streamContext.nSharedMemPerBlock                 = prop.sharedMemPerBlock;
            streamContext.nCudaDevAttrComputeCapabilityMajor = prop.major;
            streamContext.nCudaDevAttrComputeCapabilityMinor = prop.minor;

            Npp8u* pDstYUV[3]    = {d_y.d_buf, d_u.d_buf, d_v.d_buf};
            int    dstPitches[3] = {y_pitch, u_pitch, v_pitch};

            auto npp_execution = [&](cudaStream_t) {
                NppStatus blur_result = nppiFilterGaussAdvancedBorder_8u_C3R_Ctx(
                    d_input, width * channels, roi, offset, d_blurred_rgb.d_buf, blurred_pitch, roi, nTaps, d_kernel,
                    NPP_BORDER_REPLICATE, streamContext);
                NPP_CHECK(blur_result);

                auto convert_result = nppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx(d_blurred_rgb.d_buf, blurred_pitch, pDstYUV,
                                                                          dstPitches, roi, streamContext);
                NPP_CHECK(convert_result);
            };

            npp_execution(0);
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            const size_t y_size  = width * height;
            const size_t uv_size = (width / 2) * height;

            std::vector<uint8_t> h_npp_output(yuv_size);

            cudaMemcpy2D(h_npp_output.data(), width, d_y.d_buf, y_pitch, width, height, cudaMemcpyDeviceToHost);
            cudaMemcpy2D(h_npp_output.data() + y_size, width / 2, d_u.d_buf, u_pitch, width / 2, height,
                         cudaMemcpyDeviceToHost);
            cudaMemcpy2D(h_npp_output.data() + y_size + uv_size, width / 2, d_v.d_buf, v_pitch, width / 2, height,
                         cudaMemcpyDeviceToHost);

            auto npp_ms = measure_execution_ms(npp_execution, warm_up_runs, runs, 0);
            if (verbose)
                std::cout << "NPP two-step time: " << std::fixed << std::setprecision(6) << (npp_ms / runs) << " ms"
                          << std::endl;

            return nppdx_results<uint8_t> {h_npp_output, npp_ms / runs};
#else
            (void)width;
            (void)height;
            (void)yuv_size;
            (void)warm_up_runs;
            (void)runs;
            (void)d_input;
            (void)verbose;
            if (verbose)
                std::cout << "\n(NPP not available - skipping comparison)" << std::endl;
            return nppdx_results<uint8_t> {std::vector<uint8_t>(), 0};
#endif
        }

        // Compare NPPDx fused pipeline to NPP separable reference for one radius (RadiusTenths 1..100).
        template<int Arch, int RadiusTenths, unsigned int NumChannels = 3,
                 typename GaussianFirKernelProvider = gaussian_sigma2_fir_kernel_provider,
                 nppdx::gaussian_tail_width TailWidth = nppdx::gaussian_tail_width::standard>
        bool fused_yuv422p_matches_npp_separable_reference(uint8_t* d_input, uint8_t* d_output, unsigned int width,
                                                           unsigned int height, size_t yuv_size,
                                                           unsigned int warm_up_runs, unsigned int runs, bool verbose,
                                                           int* out_max_diff = nullptr, double* out_avg_diff = nullptr,
                                                           size_t* out_error_count = nullptr) {
            const int    max_lim = npp_compare_max_abs_error;
            const double avg_lim = npp_compare_avg_abs_error;

            auto nppdx_result = run_nppdx_fused<Arch, NumChannels, RadiusTenths, TailWidth>(
                d_input, d_output, width, height, yuv_size, warm_up_runs, runs, verbose);
#ifdef NPPDX_EXAMPLE_HAS_NPP
            auto npp_result = run_npp_two_step<RadiusTenths, GaussianFirKernelProvider, TailWidth>(
                d_input, width, height, yuv_size, warm_up_runs, runs, verbose);
            if (npp_result.output.empty()) {
                if (out_max_diff)
                    *out_max_diff = -1;
                return false;
            }
            auto error_stats = check_error_detailed(nppdx_result.output.data(), npp_result.output.data(), yuv_size,
                                                    npp_compare_error_threshold, max_lim, avg_lim, 1, 1, false, false);
            if (out_max_diff)
                *out_max_diff = static_cast<int>(error_stats.max_abs_error);
            if (out_avg_diff)
                *out_avg_diff = error_stats.avg_abs_error;
            if (out_error_count)
                *out_error_count = error_stats.error_count;
            return error_stats.passed;
#else
            (void)d_output;
            (void)verbose;
            if (out_max_diff)
                *out_max_diff = 0;
            if (out_avg_diff)
                *out_avg_diff = 0.;
            if (out_error_count)
                *out_error_count = 0;
            return true;
#endif
        }

    } // namespace gaussian_blur_npp_reference
} // namespace common

#endif // NPPDX_EXAMPLE_COMMON_GAUSSIAN_BLUR_NPP_REFERENCE_CUH
