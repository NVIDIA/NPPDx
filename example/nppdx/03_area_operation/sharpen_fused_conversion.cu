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


// Fused 3x3 Sharpen (Rosenfeld additive Laplacian) + RGB to YUV422P Example
// NPP path: NppiFilterBorder (3x3 additive kernel) + full-range BT.601 YCbCr422.
// NPPDx fused path: Ingest -> Sharpen (generic 3x3) -> ColorConvert -> Exgest.

#include <nppdx.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../common.hpp"

#ifdef NPPDX_EXAMPLE_HAS_NPP
#    include <nppi.h>
#endif

// =============================================================================
// NPPDx Fused Kernel: Ingest -> Sharpen -> Color Convert -> Exgest
// =============================================================================

template<typename Ingest, typename SharpenOp, typename ColorConvert, typename Exgest, typename InputTileStorageType,
         typename SharpenedTileStorageType>
__global__ void fused_sharpen_convert_kernel(const uint8_t* input, uint8_t* output, const unsigned int width,
                                             const unsigned int height) {
    extern __shared__ unsigned char smem[];

    auto tile_storage =
        nppdx::shared_memory::slice_into_tile_storage<InputTileStorageType, SharpenedTileStorageType>(smem);
    auto input_channels     = std::get<0>(tile_storage);
    auto sharpened_channels = std::get<1>(tile_storage);

    Ingest().execute(input, input_channels, width, height);
    SharpenOp().execute(input_channels, sharpened_channels, width, height);
    ColorConvert().execute(sharpened_channels, width, height);
    Exgest().execute(sharpened_channels, output, width, height);
}

// Kernel: Ingest -> Sharpen -> Exgest RGB24 (for debug comparison of sharpened RGB)
template<typename Ingest, typename SharpenOp, typename ExgestRGB, typename InputTileStorageType,
         typename SharpenedTileStorageType>
__global__ void sharpen_exgest_rgb_kernel(const uint8_t* input, uint8_t* output_rgb, const unsigned int width,
                                          const unsigned int height) {
    extern __shared__ unsigned char smem[];

    auto tile_storage =
        nppdx::shared_memory::slice_into_tile_storage<InputTileStorageType, SharpenedTileStorageType>(smem);
    auto input_channels     = std::get<0>(tile_storage);
    auto sharpened_channels = std::get<1>(tile_storage);

    Ingest().execute(input, input_channels, width, height);
    SharpenOp().execute(input_channels, sharpened_channels, width, height);
    ExgestRGB().execute(sharpened_channels, output_rgb, width, height);
}

// =============================================================================
// NPPDx Fused Execution (Weights-based: 3 floats corner_w, side_w, center_w)
// =============================================================================

template<int Arch, unsigned int NumChannels, typename SharpenWeights>
common::nppdx_results<uint8_t> run_nppdx_fused(uint8_t* d_input, uint8_t* d_output, const unsigned int width,
                                               const unsigned int height, const size_t yuv_size,
                                               unsigned int warm_up_runs, unsigned int runs) {
    constexpr unsigned int tile_x   = 24;
    constexpr unsigned int tile_y   = 42;
    constexpr unsigned int halo_x   = 1;
    constexpr unsigned int halo_y   = 1;
    constexpr nppdx::int2  uni_halo = nppdx::int2(nppdx::uint2(halo_x, halo_y));
    constexpr nppdx::Halo4 halo     = nppdx::Halo4 {uni_halo, uni_halo};

    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() + nppdx::SM<Arch>() + nppdx::Block());

    using SharpenOp = decltype(nppdx::Function<nppdx::function::sharpen>() + nppdx::Sharpen<SharpenWeights>() +
                               nppdx::TileSize<tile_x, tile_y>() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() +
                               NPPDX_MAKE_HALO(nppdx::CumulativeHalo, halo)() + nppdx::SM<Arch>() + nppdx::Block());

    using ColorConvert = decltype(nppdx::Function<nppdx::function::color_convert>() +
                                  nppdx::ColorConvert<nppdx::color_space::rgb, nppdx::color_space::yuv_bt601,
                                                      nppdx::bit_depth::bpp_8u, nppdx::bit_depth::bpp_8u>() +
                                  NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() + nppdx::EmptyCumulativeHalo() +
                                  nppdx::TileSize<tile_x, tile_y>() + nppdx::SM<Arch>() + nppdx::Block());

    using Exgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                            nppdx::OutputFormat<nppdx::packing_format::yuv422p>() + nppdx::TileSize<tile_x, tile_y>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() + nppdx::EmptyCumulativeHalo() +
                            nppdx::SM<Arch>() + nppdx::Block());

    static_assert(nppdx::is_supported_v<Ingest, Arch>, "Ingest not supported");
    static_assert(nppdx::is_supported_v<SharpenOp, Arch>, "Sharpen not supported");
    static_assert(nppdx::is_supported_v<ColorConvert, Arch>, "ColorConvert not supported");
    static_assert(nppdx::is_supported_v<Exgest, Arch>, "Exgest not supported");

    using InputTileStorageType     = nppdx::input_storage_of_t<Ingest>;
    using SharpenedTileStorageType = nppdx::output_storage_of_t<SharpenOp>;

    constexpr size_t smem_size =
        nppdx::shared_memory::compute_total_tile_storage<InputTileStorageType, SharpenedTileStorageType>();

    constexpr dim3 block_dim = Ingest::block_dim;
    const dim3     grid_dim  = Ingest::calculate_grid_dim(width, height);

    auto kernel_ptr = fused_sharpen_convert_kernel<Ingest, SharpenOp, ColorConvert, Exgest, InputTileStorageType,
                                                   SharpenedTileStorageType>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    std::cout << "\nRunning NPPDx fused (sharpen + convert)..." << std::endl;

    auto nppdx_execution = [&](cudaStream_t stream) {
        kernel_ptr<<<grid_dim, block_dim, smem_size, stream>>>(d_input, d_output, width, height);
    };

    nppdx_execution(0);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<uint8_t> h_nppdx_output(yuv_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_nppdx_output.data(), d_output, yuv_size, cudaMemcpyDeviceToHost));

    auto ms = common::measure_execution_ms(nppdx_execution, warm_up_runs, runs, 0);
    std::cout << "NPPDx fused time: " << std::fixed << std::setprecision(6) << (ms / runs) << " ms" << std::endl;

    return common::nppdx_results<uint8_t> {h_nppdx_output, ms / runs};
}

// Run NPPDx Ingest -> Sharpen -> Exgest(RGB24) and write to file (for debug comparison)
template<int Arch, unsigned int NumChannels, typename SharpenWeights>
void run_nppdx_sharpen_rgb_and_write(uint8_t* d_input, const unsigned int width, const unsigned int height,
                                     const char* filename) {
    constexpr unsigned int tile_x   = 24;
    constexpr unsigned int tile_y   = 42;
    constexpr unsigned int halo_x   = 1;
    constexpr unsigned int halo_y   = 1;
    constexpr nppdx::int2  uni_halo = nppdx::int2(nppdx::uint2(halo_x, halo_y));
    constexpr nppdx::Halo4 halo     = nppdx::Halo4 {uni_halo, uni_halo};

    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() + nppdx::SM<Arch>() + nppdx::Block());

    using SharpenOp = decltype(nppdx::Function<nppdx::function::sharpen>() + nppdx::Sharpen<SharpenWeights>() +
                               nppdx::TileSize<tile_x, tile_y>() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() +
                               NPPDX_MAKE_HALO(nppdx::CumulativeHalo, halo)() + nppdx::SM<Arch>() + nppdx::Block());

    using ExgestRGB = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                               nppdx::OutputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                               NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() + nppdx::EmptyCumulativeHalo() +
                               nppdx::SM<Arch>() + nppdx::Block());

    using InputTileStorageType     = nppdx::input_storage_of_t<Ingest>;
    using SharpenedTileStorageType = nppdx::output_storage_of_t<SharpenOp>;

    constexpr size_t smem_size =
        nppdx::shared_memory::compute_total_tile_storage<InputTileStorageType, SharpenedTileStorageType>();
    constexpr dim3 block_dim = Ingest::block_dim;
    const dim3     grid_dim  = Ingest::calculate_grid_dim(width, height);

    const size_t   rgb_size = width * height * NumChannels;
    common::DevBuf d_rgb(rgb_size);

    auto kernel_ptr =
        sharpen_exgest_rgb_kernel<Ingest, SharpenOp, ExgestRGB, InputTileStorageType, SharpenedTileStorageType>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel_ptr<<<grid_dim, block_dim, smem_size, 0>>>(d_input, d_rgb.d_buf, width, height);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<uint8_t> h_rgb(rgb_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_rgb.data(), d_rgb.d_buf, rgb_size, cudaMemcpyDeviceToHost));
    common::write_output_file(filename, h_rgb.data(), rgb_size);
}

// =============================================================================
// NPP Two-Step Execution: 3x3 Sharpen (FilterBorder) + RGB to YUV422P
// =============================================================================
//
// NPPDx uses 3-float Weights (corner_w, side_w, center_w). NPP uses an explicit
// integer kernel + divisor for nppiFilterBorder.
// Set NPPDX_SHARPEN_IDENTITY to 1 to use identity Weights (phasing check).
#define NPPDX_SHARPEN_IDENTITY 0

#if NPPDX_SHARPEN_IDENTITY
constexpr int npp_sharpen_corner  = 0;
constexpr int npp_sharpen_side    = 0;
constexpr int npp_sharpen_center  = 144;
constexpr int npp_sharpen_divisor = 144;
#else
constexpr int npp_sharpen_corner  = -2;
constexpr int npp_sharpen_side    = -5;
constexpr int npp_sharpen_center  = 173;
constexpr int npp_sharpen_divisor = 144;
#endif

static void set_sharpen_kernel_from_params(Npp32s kernel[9], int corner, int side, int center) {
    // Row-major 3x3: [corner, side, corner; side, center, side; corner, side, corner]
    kernel[0] = corner;
    kernel[1] = side;
    kernel[2] = corner;
    kernel[3] = side;
    kernel[4] = center;
    kernel[5] = side;
    kernel[6] = corner;
    kernel[7] = side;
    kernel[8] = corner;
}

common::nppdx_results<uint8_t> run_npp_sharpen_convert(uint8_t* d_input, const unsigned int width,
                                                       const unsigned int height, const size_t yuv_size,
                                                       unsigned int warm_up_runs, unsigned int runs) {
#ifdef NPPDX_EXAMPLE_HAS_NPP
    std::cout << "\nRunning NPP two-step (sharpen FilterBorder then convert)..." << std::endl;

    constexpr unsigned int channels = 3;

    Npp32s kernel[9];
    set_sharpen_kernel_from_params(kernel, npp_sharpen_corner, npp_sharpen_side, npp_sharpen_center);
    common::DevBuf d_kernel_buf(9 * sizeof(Npp32s));
    Npp32s*        d_kernel = reinterpret_cast<Npp32s*>(d_kernel_buf.d_buf);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_kernel, kernel, 9 * sizeof(Npp32s), cudaMemcpyHostToDevice));

    int            sharpened_pitch = width * channels;
    common::DevBuf d_sharpened_rgb(static_cast<size_t>(sharpened_pitch) * height);

    int            y_pitch = width;
    int            u_pitch = (width + 1) / 2;
    int            v_pitch = (width + 1) / 2;
    common::DevBuf d_y(static_cast<size_t>(y_pitch) * height);
    common::DevBuf d_u(static_cast<size_t>(u_pitch) * height);
    common::DevBuf d_v(static_cast<size_t>(v_pitch) * height);

    NppiSize  roi        = {static_cast<int>(width), static_cast<int>(height)};
    NppiSize  kernelSize = {3, 3};
    NppiPoint anchor     = {1, 1};
    NppiPoint offset     = {0, 0};

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
        // Step 1: 3x3 sharpen on RGB - nppiFilterBorder_8u_C3R_Ctx (kernel/divisor for 50% additive weight)
        auto filter_result = nppiFilterBorder_8u_C3R_Ctx(d_input, width * channels, roi, offset, d_sharpened_rgb.d_buf,
                                                         sharpened_pitch, roi, d_kernel, kernelSize, anchor,
                                                         npp_sharpen_divisor, NPP_BORDER_REPLICATE, streamContext);
        NPP_CHECK(filter_result);

        // Step 2: RGB to full-range BT.601 YCbCr422P conversion.
        auto convert_result = nppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx(d_sharpened_rgb.d_buf, sharpened_pitch, pDstYUV,
                                                                  dstPitches, roi, streamContext);
        NPP_CHECK(convert_result);
    };

    auto npp_function_only = [&](cudaStream_t) {
        auto filter_result = nppiFilterBorder_8u_C3R_Ctx(d_input, width * channels, roi, offset, d_sharpened_rgb.d_buf,
                                                         sharpened_pitch, roi, d_kernel, kernelSize, anchor,
                                                         npp_sharpen_divisor, NPP_BORDER_REPLICATE, streamContext);
        NPP_CHECK(filter_result);
    };

    npp_execution(0);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Debug: write NPP sharpened RGB for comparison
    const size_t         rgb_size = width * height * channels;
    std::vector<uint8_t> h_npp_sharpened_rgb(rgb_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy2D(h_npp_sharpened_rgb.data(), sharpened_pitch, d_sharpened_rgb.d_buf,
                                     sharpened_pitch, width * channels, height, cudaMemcpyDeviceToHost));
    common::write_output_file("cow_sharpen_npp.rgb", h_npp_sharpened_rgb.data(), rgb_size);

    const size_t y_size  = width * height;
    const size_t uv_size = (width / 2) * height;

    std::vector<uint8_t> h_npp_output(yuv_size);
    cudaMemcpy2D(h_npp_output.data(), width, d_y.d_buf, y_pitch, width, height, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(h_npp_output.data() + y_size, width / 2, d_u.d_buf, u_pitch, width / 2, height,
                 cudaMemcpyDeviceToHost);
    cudaMemcpy2D(h_npp_output.data() + y_size + uv_size, width / 2, d_v.d_buf, v_pitch, width / 2, height,
                 cudaMemcpyDeviceToHost);

    auto npp_ms      = common::measure_execution_ms(npp_execution, warm_up_runs, runs, 0);
    auto npp_func_ms = common::measure_execution_ms(npp_function_only, warm_up_runs, runs, 0);
    std::cout << "NPP two-step time: " << std::fixed << std::setprecision(6) << (npp_ms / runs) << " ms" << std::endl;
    std::cout << "NPP function time: " << std::fixed << std::setprecision(6) << (npp_func_ms / runs) << " ms"
              << std::endl;

    return common::nppdx_results<uint8_t> {h_npp_output, npp_ms / runs};
#else
    std::cout << "\n(NPP not available)" << std::endl;
    return common::nppdx_results<uint8_t> {std::vector<uint8_t>(), 0};
#endif
}

// =============================================================================
// Main Example Function
// =============================================================================

template<int Arch>
int sharpen_fused_conversion_example() {
    std::cout << "=== Sharpen (3x3 Rosenfeld additive) + RGB to YUV422P Example (SM" << Arch << ") ===" << std::endl;

    constexpr unsigned int warm_up_runs = 10;
    constexpr unsigned int runs         = 100;

    constexpr unsigned int width    = 312;
    constexpr unsigned int height   = 376;
    constexpr unsigned int channels = 3;
    constexpr size_t       rgb_size = width * height * channels;

    constexpr size_t y_size   = width * height;
    constexpr size_t uv_size  = (width / 2) * height;
    constexpr size_t yuv_size = y_size + 2 * uv_size;

    std::cout << "Image: " << width << "x" << height << " RGB24 (" << rgb_size << " bytes) -> YUV422P (" << yuv_size
              << " bytes)" << std::endl;
    std::cout << "Sharpen: 3-float Weights (20%); NPP uses an explicit 3x3 integer kernel, divisor 144" << std::endl;

#if NPPDX_SHARPEN_IDENTITY
    using SharpenWeights = nppdx::sharpen_weights::identity_weights;
#else
    using SharpenWeights = nppdx::sharpen_weights::rosenfeld_generalized_weights<20>;
#endif

    std::vector<uint8_t> h_input = common::load_or_generate_rgb_test_data(width, height);

    common::DevBuf d_input(rgb_size);
    common::DevBuf d_output(yuv_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_input.data(), rgb_size, cudaMemcpyHostToDevice));

    auto nppdx_result = run_nppdx_fused<Arch, channels, SharpenWeights>(d_input.d_buf, d_output.d_buf, width, height,
                                                                        yuv_size, warm_up_runs, runs);

    // Debug: write NPPDx sharpened RGB for comparison with cow_sharpen_npp.rgb
    run_nppdx_sharpen_rgb_and_write<Arch, channels, SharpenWeights>(d_input.d_buf, width, height,
                                                                    "cow_sharpen_nppdx.rgb");

#ifdef NPPDX_EXAMPLE_HAS_NPP
    auto npp_result = run_npp_sharpen_convert(d_input.d_buf, width, height, yuv_size, warm_up_runs, runs);

    bool passed = true;
    if (!npp_result.output.empty()) {
        std::cout << "\n--- Comparison (per-plane) ---" << std::endl;
        constexpr size_t y_size  = width * height;
        constexpr size_t uv_size = (width / 2) * height;

        // Unlike box/median (outputs stay in the input hull, so NPP vs NPPDx clamp ordering rarely matters), sharpen can
        // produce out-of-gamut linear RGB before YUV; NPP and NPPDx then differ in where they saturate. After functional
        // validation, comparison vs NPP uses relaxed per-plane limits; color conversion / 4:2:2 averaging adds slack on U/V.
        constexpr int    y_max_err  = 20;
        constexpr double y_avg_err  = 1.0;
        constexpr int    uv_max_err = 32;
        constexpr double uv_avg_err = 3.0;

        auto y_stats = common::check_error_detailed(nppdx_result.output.data(), npp_result.output.data(), y_size, 2,
                                                    y_max_err, y_avg_err, 1, 1, false, false);
        std::cout << "Y plane:  max_diff=" << y_stats.max_abs_error << ", avg_diff=" << y_stats.avg_abs_error
                  << ", mismatches=" << y_stats.error_count << "/" << y_size << " (limit max=" << y_max_err
                  << " avg=" << y_avg_err << ")" << std::endl;

        auto u_stats =
            common::check_error_detailed(nppdx_result.output.data() + y_size, npp_result.output.data() + y_size,
                                         uv_size, 2, uv_max_err, uv_avg_err, 1, 1, false, false);
        std::cout << "U plane:  max_diff=" << u_stats.max_abs_error << ", avg_diff=" << u_stats.avg_abs_error
                  << ", mismatches=" << u_stats.error_count << "/" << uv_size << " (limit max=" << uv_max_err
                  << " avg=" << uv_avg_err << ")" << std::endl;

        auto v_stats = common::check_error_detailed(nppdx_result.output.data() + y_size + uv_size,
                                                    npp_result.output.data() + y_size + uv_size, uv_size, 2, uv_max_err,
                                                    uv_avg_err, 1, 1, false, false);
        std::cout << "V plane:  max_diff=" << v_stats.max_abs_error << ", avg_diff=" << v_stats.avg_abs_error
                  << ", mismatches=" << v_stats.error_count << "/" << uv_size << " (limit max=" << uv_max_err
                  << " avg=" << uv_avg_err << ")" << std::endl;

        auto error_stats = common::check_error_detailed(nppdx_result.output.data(), npp_result.output.data(), yuv_size,
                                                        2, uv_max_err, uv_avg_err, 1, 1, false, false);
        std::cout << "Overall:  max_diff=" << error_stats.max_abs_error << ", avg_diff=" << error_stats.avg_abs_error
                  << ", mismatches=" << error_stats.error_count << "/" << yuv_size << std::endl;

        if (npp_result.avg_time_in_ms > 0) {
            float speedup = npp_result.avg_time_in_ms / nppdx_result.avg_time_in_ms;
            std::cout << "Speedup (NPP/NPPDx): " << speedup << "x" << std::endl;
        }
        passed = y_stats.passed && u_stats.passed && v_stats.passed;
    }

    common::write_output_file("cow_yuv422p_sharpen_nppdx.yuv422",
                              reinterpret_cast<const uint8_t*>(nppdx_result.output.data()), yuv_size);
#    ifdef NPPDX_EXAMPLE_HAS_NPP
    if (!npp_result.output.empty()) {
        common::write_output_file("cow_yuv422p_sharpen_npp.yuv422",
                                  reinterpret_cast<const uint8_t*>(npp_result.output.data()), yuv_size);
    }
#    endif
#endif

#ifdef NPPDX_EXAMPLE_HAS_NPP
    std::cout << "\nResult: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
#else
    std::cout << "\nResult: PASSED (NPPDx only; NPP not available)" << std::endl;
    return 0;
#endif
}

template<int Arch>
struct sharpen_fused_conversion_functor {
    int operator()() const { return sharpen_fused_conversion_example<Arch>(); }
};

int main() {
    return common::run_example_with_sm<sharpen_fused_conversion_functor>();
}
