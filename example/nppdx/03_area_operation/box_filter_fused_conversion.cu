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


// Fused Box Blur + RGB to YUV422P Example - NPPDx vs NPP comparison
// Demonstrates fusing box blur with color conversion in a single NPPDx kernel
// compared to NPP's two-step approach.

#include <nppdx.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "../common.hpp"

#ifdef NPPDX_EXAMPLE_HAS_NPP
#    include <nppi.h>
#endif

// =============================================================================
// NPPDx Fused Kernel: Ingest -> Box Blur -> Color Convert -> Exgest
// =============================================================================

template<typename Ingest, typename BoxBlur, typename ColorConvert, typename Exgest, typename InputTileStorageType,
         typename BlurredTileStorageType>
__global__ void fused_blur_convert_kernel(const uint8_t* input, uint8_t* output, const unsigned int width,
                                          const unsigned int height) {
    extern __shared__ unsigned char smem[];

    auto tile_storage =
        nppdx::shared_memory::slice_into_tile_storage<InputTileStorageType, BlurredTileStorageType>(smem);
    auto input_channels   = std::get<0>(tile_storage);
    auto blurred_channels = std::get<1>(tile_storage);

    // Step 1: Ingest RGB24 with halo
    Ingest().execute(input, input_channels, width, height);

    // Step 2: Box blur
    BoxBlur().execute(input_channels, blurred_channels, width, height);

    // Step 3: Color convert RGB -> YUV (in-place on blurred_channels)
    ColorConvert().execute(blurred_channels, width, height);

    // Step 4: Exgest to YUV422P
    Exgest().execute(blurred_channels, output, width, height);
}

// =============================================================================
// NPPDx Fused Execution
// =============================================================================

template<int Arch, unsigned int NumChannels, unsigned int KernelW, unsigned int KernelH>
common::nppdx_results<uint8_t> run_nppdx_fused(uint8_t* d_input, uint8_t* d_output, const unsigned int width,
                                               const unsigned int height, const size_t yuv_size,
                                               unsigned int warm_up_runs, unsigned int runs) {
    constexpr unsigned int tile_x = 32;
    constexpr unsigned int tile_y = 32;
    constexpr unsigned int halo_x = (KernelW - 1) / 2;
    constexpr unsigned int halo_y = (KernelH - 1) / 2;

    using UniHalo = nppdx::Int2D<halo_x, halo_y>;

    // Ingest RGB24 with halo for box blur
    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                            nppdx::MemoryHalo<UniHalo, UniHalo>() + nppdx::SM<Arch>() + nppdx::Block());

    // Box blur operation
    using BoxBlur = decltype(nppdx::Function<nppdx::function::box_blur>() + nppdx::BoxBlur<KernelW, KernelH>() +
                             nppdx::TileSize<tile_x, tile_y>() + nppdx::MemoryHalo<UniHalo, UniHalo>() +

                             nppdx::CumulativeHalo<UniHalo, UniHalo>() + nppdx::SM<Arch>() + nppdx::Block());

    // Color conversion RGB -> YUV BT.601
    using ColorConvert = decltype(nppdx::Function<nppdx::function::color_convert>() +
                                  nppdx::ColorConvert<nppdx::color_space::rgb, nppdx::color_space::yuv_bt601,
                                                      nppdx::bit_depth::bpp_8u, nppdx::bit_depth::bpp_8u>() +
                                  nppdx::MemoryHalo<UniHalo, UniHalo>() + nppdx::EmptyCumulativeHalo() +
                                  nppdx::TileSize<tile_x, tile_y>() + nppdx::SM<Arch>() + nppdx::Block());

    // Exgest to YUV422P planar format
    using Exgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                            nppdx::OutputFormat<nppdx::packing_format::yuv422p>() + nppdx::TileSize<tile_x, tile_y>() +
                            nppdx::MemoryHalo<UniHalo, UniHalo>() + nppdx::EmptyCumulativeHalo() + nppdx::SM<Arch>() +
                            nppdx::Block());

    static_assert(nppdx::is_supported_v<Ingest, Arch>, "Ingest not supported");
    static_assert(nppdx::is_supported_v<BoxBlur, Arch>, "BoxBlur not supported");
    static_assert(nppdx::is_supported_v<ColorConvert, Arch>, "ColorConvert not supported");
    static_assert(nppdx::is_supported_v<Exgest, Arch>, "Exgest not supported");

    using InputTileStorageType   = nppdx::input_storage_of_t<Ingest>;
    using BlurredTileStorageType = nppdx::output_storage_of_t<BoxBlur>;

    std::cout << "Tile: " << tile_x << "x" << tile_y << " with halo=" << halo_x << "," << halo_y << "," << halo_x << ","
              << halo_y << std::endl;
    std::cout << "Input storage tile: " << Ingest::input_storage.size.x << "x" << Ingest::input_storage.size.y
              << std::endl;
    std::cout << "Blurred storage tile: " << BoxBlur::output_storage.size.x << "x" << BoxBlur::output_storage.size.y
              << std::endl;

    constexpr size_t smem_size =
        nppdx::shared_memory::compute_total_tile_storage<InputTileStorageType, BlurredTileStorageType>();
    std::cout << "Shared memory: " << smem_size << " bytes per block" << std::endl;

    constexpr dim3 block_dim = Ingest::block_dim;
    const dim3     grid_dim  = Ingest::calculate_grid_dim(width, height);

    std::cout << "Grid: " << grid_dim.x << "x" << grid_dim.y << " | Block: " << block_dim.x << "x" << block_dim.y
              << std::endl;

    auto kernel_ptr =
        fused_blur_convert_kernel<Ingest, BoxBlur, ColorConvert, Exgest, InputTileStorageType, BlurredTileStorageType>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    std::cout << "\nRunning NPPDx fused (blur + convert)..." << std::endl;

    auto nppdx_execution = [&](cudaStream_t stream) {
        kernel_ptr<<<grid_dim, block_dim, smem_size, stream>>>(d_input, d_output, width, height);
    };

    // Correctness run
    nppdx_execution(0);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<uint8_t> h_nppdx_output(yuv_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_nppdx_output.data(), d_output, yuv_size, cudaMemcpyDeviceToHost));

    auto ms = common::measure_execution_ms(nppdx_execution, warm_up_runs, runs, 0);
    std::cout << "NPPDx fused time: " << ms / runs << " ms" << std::endl;

    return common::nppdx_results<uint8_t> {h_nppdx_output, ms / runs};
}

// =============================================================================
// NPP Two-Step Execution: Box Blur + RGB to YUV422P
// =============================================================================

template<unsigned int KernelW, unsigned int KernelH>
common::nppdx_results<uint8_t> run_npp_two_step(uint8_t* d_input, const unsigned int width, const unsigned int height,
                                                const size_t yuv_size, unsigned int warm_up_runs, unsigned int runs) {
#ifdef NPPDX_EXAMPLE_HAS_NPP
    std::cout << "\nRunning NPP two-step (blur then convert)..." << std::endl;

    constexpr unsigned int channels = 3;

    // Allocate device buffers (contiguous memory - pitch equals row width)
    // Blurred RGB intermediate buffer
    int            blurred_pitch = width * channels;
    common::DevBuf d_blurred_rgb(blurred_pitch * height);

    // YUV422P output planes: Y is width x height, U and V are (width/2) x height
    int            y_pitch = width;
    int            u_pitch = (width + 1) / 2;
    int            v_pitch = (width + 1) / 2;
    common::DevBuf d_y(y_pitch * height);
    common::DevBuf d_u(u_pitch * height);
    common::DevBuf d_v(v_pitch * height);

    NppiSize  roi    = {static_cast<int>(width), static_cast<int>(height)};
    NppiSize  mask   = {static_cast<int>(KernelW), static_cast<int>(KernelH)};
    NppiPoint anchor = {static_cast<int>(KernelW / 2), static_cast<int>(KernelH / 2)};
    NppiPoint offset = {0, 0};

    // Properly initialize NPP stream context with device properties
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
        // Step 1: Box blur on RGB
        auto blur_result =
            nppiFilterBoxBorder_8u_C3R_Ctx(d_input, width * channels, roi, offset, d_blurred_rgb.d_buf, blurred_pitch,
                                           roi, mask, anchor, NPP_BORDER_REPLICATE, streamContext);
        NPP_CHECK(blur_result);

        // Step 2: RGB to full-range BT.601 YCbCr422P conversion.
        auto convert_result = nppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx(d_blurred_rgb.d_buf, blurred_pitch, pDstYUV,
                                                                  dstPitches, roi, streamContext);
        NPP_CHECK(convert_result);
    };

    // Correctness run
    npp_execution(0);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy YUV422P output to host (contiguous format: Y, then U, then V)
    const size_t y_size  = width * height;
    const size_t uv_size = (width / 2) * height;

    std::vector<uint8_t> h_npp_output(yuv_size);

    // Copy Y plane
    cudaMemcpy2D(h_npp_output.data(), width, d_y.d_buf, y_pitch, width, height, cudaMemcpyDeviceToHost);
    // Copy U plane
    cudaMemcpy2D(h_npp_output.data() + y_size, width / 2, d_u.d_buf, u_pitch, width / 2, height,
                 cudaMemcpyDeviceToHost);
    // Copy V plane
    cudaMemcpy2D(h_npp_output.data() + y_size + uv_size, width / 2, d_v.d_buf, v_pitch, width / 2, height,
                 cudaMemcpyDeviceToHost);

    auto npp_ms = common::measure_execution_ms(npp_execution, warm_up_runs, runs, 0);
    std::cout << "NPP two-step time: " << npp_ms / runs << " ms" << std::endl;

    return common::nppdx_results<uint8_t> {h_npp_output, npp_ms / runs};
#else
    std::cout << "\n(NPP not available - skipping comparison)" << std::endl;
    return common::nppdx_results<uint8_t> {std::vector<uint8_t>(), 0};
#endif
}

// =============================================================================
// Main Example Function
// =============================================================================

template<int Arch>
int box_filter_fused_conversion_example() {
    std::cout << "=== NPPDx Fused Box Blur + RGB to YUV422P Example (SM" << Arch << ") ===" << std::endl;

    constexpr unsigned int warm_up_runs = 10;
    constexpr unsigned int runs         = 100;

    // Same input dimensions as box_blur_filter.cu
    constexpr unsigned int KernelW  = 5;
    constexpr unsigned int KernelH  = 5;
    constexpr unsigned int width    = 312;
    constexpr unsigned int height   = 376;
    constexpr unsigned int channels = 3;
    constexpr size_t       rgb_size = width * height * channels;

    // YUV422P output sizes
    constexpr size_t y_size   = width * height;
    constexpr size_t uv_size  = (width / 2) * height;
    constexpr size_t yuv_size = y_size + 2 * uv_size;

    std::cout << "Image: " << width << "x" << height << " RGB24 (" << rgb_size << " bytes) -> YUV422P (" << yuv_size
              << " bytes)" << std::endl;
    std::cout << "Box filter: " << KernelW << "x" << KernelH << std::endl;

    // Load input data (same as box_blur_filter.cu)
    std::vector<uint8_t> h_input = common::load_or_generate_rgb_test_data(width, height);

    // Allocate device memory
    common::DevBuf d_input(rgb_size);
    common::DevBuf d_output(yuv_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_input.data(), rgb_size, cudaMemcpyHostToDevice));

    // Run NPPDx fused pipeline
    auto nppdx_result = run_nppdx_fused<Arch, channels, KernelW, KernelH>(d_input.d_buf, d_output.d_buf, width, height,
                                                                          yuv_size, warm_up_runs, runs);

    // Run NPP two-step pipeline
    auto npp_result = run_npp_two_step<KernelW, KernelH>(d_input.d_buf, width, height, yuv_size, warm_up_runs, runs);

    // Compare results
    bool passed = true;
    if (!npp_result.output.empty()) {
        std::cout << "\n--- Comparison (per-plane) ---" << std::endl;

        // Compare Y plane
        auto y_stats = common::check_error_detailed(nppdx_result.output.data(), npp_result.output.data(), y_size, 2, 10,
                                                    2.0, 1, 1, false, false);
        std::cout << "Y plane:  max_diff=" << y_stats.max_abs_error << ", avg_diff=" << y_stats.avg_abs_error
                  << ", mismatches=" << y_stats.error_count << "/" << y_size << std::endl;

        // Compare U plane
        auto u_stats =
            common::check_error_detailed(nppdx_result.output.data() + y_size, npp_result.output.data() + y_size,
                                         uv_size, 2, 10, 2.0, 1, 1, false, false);
        std::cout << "U plane:  max_diff=" << u_stats.max_abs_error << ", avg_diff=" << u_stats.avg_abs_error
                  << ", mismatches=" << u_stats.error_count << "/" << uv_size << std::endl;

        // Compare V plane
        auto v_stats = common::check_error_detailed(nppdx_result.output.data() + y_size + uv_size,
                                                    npp_result.output.data() + y_size + uv_size, uv_size, 2, 10, 2.0, 1,
                                                    1, false, false);
        std::cout << "V plane:  max_diff=" << v_stats.max_abs_error << ", avg_diff=" << v_stats.avg_abs_error
                  << ", mismatches=" << v_stats.error_count << "/" << uv_size << std::endl;

        // Overall comparison
        auto error_stats = common::check_error_detailed(nppdx_result.output.data(), npp_result.output.data(), yuv_size,
                                                        2, 10, 2.0, 1, 1, false, false);
        std::cout << "Overall:  max_diff=" << error_stats.max_abs_error << ", avg_diff=" << error_stats.avg_abs_error
                  << ", mismatches=" << error_stats.error_count << "/" << yuv_size << std::endl;

        // Calculate speedup
        if (npp_result.avg_time_in_ms > 0) {
            float speedup = npp_result.avg_time_in_ms / nppdx_result.avg_time_in_ms;
            std::cout << "Speedup (NPP/NPPDx): " << speedup << "x" << std::endl;
        }

        passed = error_stats.passed;
    }

    // Write output files for inspection
    common::write_output_file("cow_yuv422p_fused_nppdx.yuv422",
                              reinterpret_cast<const uint8_t*>(nppdx_result.output.data()), yuv_size);

    if (!npp_result.output.empty()) {
        common::write_output_file("cow_yuv422p_twostep_npp.yuv422",
                                  reinterpret_cast<const uint8_t*>(npp_result.output.data()), yuv_size);
    }

    std::cout << "\nResult: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}

template<int Arch>
struct box_filter_fused_conversion_functor {
    int operator()() const { return box_filter_fused_conversion_example<Arch>(); }
};

int main() {
    return common::run_example_with_sm<box_filter_fused_conversion_functor>();
}
