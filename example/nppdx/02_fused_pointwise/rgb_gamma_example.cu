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


#include <nppdx.hpp>

#include <iostream>
#include <tuple>
#include <vector>
#include <string>

#include "../common.hpp"

#include <type_traits>

namespace {

    // Kernels

    template<typename RGBIngest, typename RGBExgest, typename GammaForward,
             typename GammaInverse = void>
    __global__ void rgb24_ingest_exgest_simple_gamma_kernel(const uint8_t* rgb_input,  // RGB24 input
                                                            uint8_t*       rgb_output, // RGB24 output
                                                            int width, int height) {
        // Internal processing format (RGB float)
        float rgb_data[RGBIngest::elements_per_thread];

        // Step 1: INGEST - Load RGB24 and convert to internal format
        RGBIngest().execute(rgb_input, rgb_data, width, height);

        // Step 2: Forward gamma transform is applied.
        GammaForward().execute(rgb_data, width, height);
        if constexpr (!std::is_same_v<GammaInverse, void>) {
            // Step 2a: Inverse gamma transform is applied.
            // The output of the Inverse transform has to be the same as the input.
            GammaInverse().execute(rgb_data, width, height);
        }

        // Step 3: EXGEST - Convert internal format back to RGB24 and store
        RGBExgest().execute(rgb_data, rgb_output, width, height);
    }


    // Image dimensions (matching cow image size for consistency)
    constexpr int width    = 312;
    constexpr int height   = 376;
    constexpr int rgb_size = width * height * 3;

    static auto check_output_error(const std::vector<uint8_t>& h_rgb24_input,
                                   uint8_t* d_output) -> std::tuple<std::vector<uint8_t>, std::vector<uint8_t>> {
        std::vector<uint8_t> h_output_image(rgb_size), difference_image(rgb_size);

        // Get image data from device
        CUDA_CHECK_AND_EXIT(cudaMemcpy(h_output_image.data(), d_output, rgb_size, cudaMemcpyDeviceToHost));

        // Build difference image for visualization
        for (int i = 0; i < rgb_size; ++i) {
            difference_image[i] = static_cast<uint8_t>(
                std::abs(static_cast<int>(h_rgb24_input[i]) - static_cast<int>(h_output_image[i])));
        }

        // Compute and print error statistics
        common::compute_conversion_stats(h_rgb24_input.data(), h_output_image.data(), rgb_size,
                                         /*error_threshold=*/0, /*max_error_limit=*/40, /*avg_error_limit=*/3.0,
                                         /*print=*/true, /*test_name=*/"Gamma roundtrip");

        return {h_output_image, difference_image};
    }

    using transfer_fun = nppdx::gamma_transfer_function;
    using nppdx::gamma_dir;

#define ASSERT_SUPPORTED(Arch, Op) \
    static_assert(nppdx::is_supported_v<Op, Arch>, #Op " invalid or not supported for this architecture")

    template<int Arch>
    struct ExampleContext {
        using BaseOp = decltype(nppdx::TileSize<32, 16> {} + nppdx::SM<Arch> {});

        using GammaBaseOp = decltype(nppdx::Function<nppdx::function::gamma> {} + BaseOp {});

        template<nppdx::gamma_dir Dir, transfer_fun TransferFun>
        using GammaOp = decltype(nppdx::GammaTransform<nppdx::bit_depth::bpp_8u, Dir, TransferFun> {} + GammaBaseOp {} +
                                 nppdx::Block {});

        using RGBIngest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                                   nppdx::InputFormat<nppdx::packing_format::rgb24>() + BaseOp() + nppdx::Block {});
        ASSERT_SUPPORTED(Arch, RGBIngest);

        using RGBExgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                                   nppdx::OutputFormat<nppdx::packing_format::rgb24>() + BaseOp() + nppdx::Block {});
        ASSERT_SUPPORTED(Arch, RGBExgest);

        static constexpr dim3 block_dim = RGBIngest::block_dim;
        static constexpr dim3 grid_dim  = RGBIngest::calculate_grid_dim(width, height);

        int rgb_gamma_example();

        int operator()() { return rgb_gamma_example(); }

    private:
        template<typename FirstGammaOp, typename SecondGammaOp = void>
        void do_gamma_test(std::string const& prefix, const std::vector<uint8_t>& h_rgb24_input) {
            constexpr bool is_verification = !std::is_same_v<SecondGammaOp, void>;

            // Allocate buffers and populate input buffer with RGB values from the image
            common::DevBuf d_input(rgb_size);
            common::DevBuf d_output(rgb_size);

            CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_rgb24_input.data(), rgb_size, cudaMemcpyHostToDevice));

            // Launch HLG kernel
            rgb24_ingest_exgest_simple_gamma_kernel<RGBIngest, RGBExgest, FirstGammaOp, SecondGammaOp>
                <<<grid_dim, block_dim>>>(d_input.d_buf, d_output.d_buf, width, height);

            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            std::vector<uint8_t> h_output_image;

            // Check and output results
            if constexpr (is_verification) {
                std::vector<uint8_t> difference_image;
                std::tie(h_output_image, difference_image) = check_output_error(h_rgb24_input, d_output.d_buf);
                const std::string difference_filename      = prefix + "_diff.rgb";
                common::write_output_file(difference_filename.c_str(), difference_image.data(),
                                          difference_image.size());
            } else {
                // Copy results to host
                h_output_image = std::vector<uint8_t>(rgb_size);
                CUDA_CHECK_AND_EXIT(
                    cudaMemcpy(h_output_image.data(), d_output.d_buf, rgb_size, cudaMemcpyDeviceToHost));
            }

            const std::string output_filename = prefix + "_result.rgb";
            common::write_output_file(output_filename.c_str(), h_output_image.data(), h_output_image.size());
        }
    };

    // Actual Example function

    template<int Arch>
    int ExampleContext<Arch>::rgb_gamma_example() {
        std::cout << "=== NPPDx RGB Gamma Transforms Example (SM" << Arch << ") ===" << std::endl;
        // ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ = ~ =

        std::cout << "Image dimensions: " << width << "x" << height << std::endl;

        using GammaFwd_SDR = GammaOp<gamma_dir::forward, transfer_fun::SDR>;
        using GammaInv_SDR = GammaOp<gamma_dir::inverse, transfer_fun::SDR>;
        using GammaFwd_HLG = GammaOp<gamma_dir::forward, transfer_fun::HLG>;
        using GammaInv_HLG = GammaOp<gamma_dir::inverse, transfer_fun::HLG>;
        using GammaFwd_PQ  = GammaOp<gamma_dir::forward, transfer_fun::PQ>;
        using GammaInv_PQ  = GammaOp<gamma_dir::inverse, transfer_fun::PQ>;

        ASSERT_SUPPORTED(Arch, GammaFwd_SDR);
        ASSERT_SUPPORTED(Arch, GammaInv_SDR);
        ASSERT_SUPPORTED(Arch, GammaFwd_HLG);
        ASSERT_SUPPORTED(Arch, GammaInv_HLG);
        ASSERT_SUPPORTED(Arch, GammaFwd_PQ);
        ASSERT_SUPPORTED(Arch, GammaInv_PQ);

        std::vector<uint8_t> h_rgb24_input = common::load_or_generate_rgb_test_data(width, height);

        std::cout << "Launch Config - Grid: " << grid_dim.x << "x" << grid_dim.y << ", Block: " << block_dim.x << "x"
                  << block_dim.y << std::endl;

        std::cout << "\n=== [EXAMPLE 1.1] : SDR Gamma Correctness === \n";
        do_gamma_test<GammaFwd_SDR, GammaInv_SDR>("SDR_correctness", h_rgb24_input);

        std::cout << "\n=== [EXAMPLE 1.2] : HLG Gamma Correctness === \n";
        do_gamma_test<GammaFwd_HLG, GammaInv_HLG>("HLG_correctness", h_rgb24_input);

        std::cout << "\n=== [EXAMPLE 1.3] : PQ Gamma Correctness === \n";
        do_gamma_test<GammaFwd_PQ, GammaInv_PQ>("PQ_correctness", h_rgb24_input);

        std::cout << "\n=== [EXAMPLE 2.1] : Apply SDR Gamma forward === \n";
        do_gamma_test<GammaFwd_SDR>("SDR_forward", h_rgb24_input);

        std::cout << "\n=== [EXAMPLE 2.2] : Apply HLG Gamma forward === \n";
        do_gamma_test<GammaFwd_HLG>("HLG_forward", h_rgb24_input);

        std::cout << "\n=== [EXAMPLE 2.3] : Apply PQ Gamma forward === \n";
        do_gamma_test<GammaFwd_PQ>("PQ_forward", h_rgb24_input);

        std::cout << "\n=== [EXAMPLE 3.1] : Apply SDR Gamma inverse === \n";
        do_gamma_test<GammaInv_SDR>("SDR_inverse", h_rgb24_input);

        std::cout << "\n=== [EXAMPLE 3.2] : Apply HLG Gamma inverse === \n";
        do_gamma_test<GammaInv_HLG>("HLG_inverse", h_rgb24_input);

        std::cout << "\n=== [EXAMPLE 3.3] : Apply PQ Gamma inverse === \n";
        do_gamma_test<GammaInv_PQ>("PQ_inverse", h_rgb24_input);

        std::cout << "\n=== NPPDx RGB Gamma Transforms Example Complete" << std::endl;

        return 0;
    }

} // namespace

//  Main function

int main() {
    // Use the DX-style SM runner to dispatch based on device architecture
    return common::run_example_with_sm<ExampleContext>();
}
