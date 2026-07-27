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

// Fused 3x3 Median + RGB to YUV422P Example - NPPDx vs NPP comparison
// Uses median_radius::r1_0 (3x3) which has a direct NPP equivalent.

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
// NPPDx Fused Kernel: Ingest -> Median (3x3) -> Color Convert -> Exgest
// =============================================================================

template<typename Ingest, typename Median, typename ColorConvert, typename Exgest, typename InputTileStorageType,
         typename MedianTileStorageType>
__global__ void fused_median_convert_kernel(const uint8_t* input, uint8_t* output, const unsigned int width,
                                            const unsigned int height) {
    extern __shared__ unsigned char smem[];

    auto tile_storage =
        nppdx::shared_memory::slice_into_tile_storage<InputTileStorageType, MedianTileStorageType>(smem);
    auto input_channels  = std::get<0>(tile_storage);
    auto median_channels = std::get<1>(tile_storage);

    Ingest().execute(input, input_channels, width, height);
    Median().execute(input_channels, median_channels, width, height);
    ColorConvert().execute(median_channels, width, height);
    Exgest().execute(median_channels, output, width, height);
}

// =============================================================================
// NPPDx Fused Execution
// =============================================================================

template<int Arch, unsigned int NumChannels>
common::nppdx_results<uint8_t> run_nppdx_fused(uint8_t* d_input, uint8_t* d_output, const unsigned int width,
                                               const unsigned int height, const size_t yuv_size,
                                               unsigned int warm_up_runs, unsigned int runs) {
    constexpr unsigned int tile_x   = 24;
    constexpr unsigned int tile_y   = 42;
    constexpr unsigned int halo_x   = 1; // 3x3 median
    constexpr unsigned int halo_y   = 1;
    constexpr nppdx::int2  uni_halo = nppdx::int2(nppdx::uint2(halo_x, halo_y));
    constexpr nppdx::Halo4 halo     = nppdx::Halo4 {uni_halo, uni_halo};

    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() + nppdx::SM<Arch>() + nppdx::Block());

    using Median = decltype(nppdx::Function<nppdx::function::median>() + nppdx::Median<nppdx::median_radius::r1_0>() +
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
    static_assert(nppdx::is_supported_v<Median, Arch>, "Median not supported");
    static_assert(nppdx::is_supported_v<ColorConvert, Arch>, "ColorConvert not supported");
    static_assert(nppdx::is_supported_v<Exgest, Arch>, "Exgest not supported");

    using InputTileStorageType  = nppdx::input_storage_of_t<Ingest>;
    using MedianTileStorageType = nppdx::output_storage_of_t<Median>;

    std::cout << "Tile: " << tile_x << "x" << tile_y << " halo=1 (3x3 median)" << std::endl;
    std::cout << "Input storage tile: " << Ingest::input_storage.size.x << "x" << Ingest::input_storage.size.y
              << std::endl;
    std::cout << "Median storage tile: " << Median::output_storage.size.x << "x" << Median::output_storage.size.y
              << std::endl;

    constexpr size_t smem_size =
        nppdx::shared_memory::compute_total_tile_storage<InputTileStorageType, MedianTileStorageType>();
    std::cout << "Shared memory: " << smem_size << " bytes per block" << std::endl;

    constexpr dim3 block_dim = Ingest::block_dim;
    const dim3     grid_dim  = Ingest::calculate_grid_dim(width, height);

    auto kernel_ptr =
        fused_median_convert_kernel<Ingest, Median, ColorConvert, Exgest, InputTileStorageType, MedianTileStorageType>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    std::cout << "\nRunning NPPDx fused (3x3 median + convert)..." << std::endl;

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

// =============================================================================
// NPP Two-Step: 3x3 Median + RGB to YUV422P
// =============================================================================

common::nppdx_results<uint8_t> run_npp_two_step(uint8_t* d_input, const unsigned int width, const unsigned int height,
                                                const size_t yuv_size, unsigned int warm_up_runs, unsigned int runs) {
#ifdef NPPDX_EXAMPLE_HAS_NPP
    // NPP comparison uses only _Ctx API (nppiFilterMedianBorder_8u_C3R_Ctx); non-_Ctx versions are deprecated.
    std::cout << "\nRunning NPP two-step (median then convert)..." << std::endl;

    constexpr unsigned int channels = 3;
    constexpr int          maskW = 3, maskH = 3;
    constexpr int          anchorX = 1, anchorY = 1;

    int            median_pitch = width * channels;
    common::DevBuf d_median_rgb(static_cast<size_t>(median_pitch) * height);

    int            y_pitch = width, u_pitch = (width + 1) / 2, v_pitch = (width + 1) / 2;
    common::DevBuf d_y(static_cast<size_t>(y_pitch) * height);
    common::DevBuf d_u(static_cast<size_t>(u_pitch) * height);
    common::DevBuf d_v(static_cast<size_t>(v_pitch) * height);

    NppiSize  roi    = {static_cast<int>(width), static_cast<int>(height)};
    NppiSize  mask   = {maskW, maskH};
    NppiPoint anchor = {anchorX, anchorY};
    NppiPoint offset = {0, 0};

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

    Npp32u median_buf_size = 0;
    NPP_CHECK(nppiFilterMedianBorderGetBufferSize_8u_C3R_Ctx(roi, mask, &median_buf_size, NPP_BORDER_REPLICATE,
                                                             streamContext));
    common::DevBuf d_median_buf(median_buf_size);

    Npp8u* pDstYUV[3]    = {d_y.d_buf, d_u.d_buf, d_v.d_buf};
    int    dstPitches[3] = {y_pitch, u_pitch, v_pitch};

    // _Ctx API: pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, pBuffer, eBorderType, ctx
    auto npp_execution = [&](cudaStream_t) {
        auto median_result = nppiFilterMedianBorder_8u_C3R_Ctx(d_input, width * channels, roi, offset,
                                                               d_median_rgb.d_buf, median_pitch, roi, mask, anchor,
                                                               d_median_buf.d_buf, NPP_BORDER_REPLICATE, streamContext);
        NPP_CHECK(median_result);
        auto convert_result = nppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx(d_median_rgb.d_buf, median_pitch, pDstYUV, dstPitches,
                                                                  roi, streamContext);
        NPP_CHECK(convert_result);
    };

    auto npp_function_only = [&](cudaStream_t) {
        auto median_result = nppiFilterMedianBorder_8u_C3R_Ctx(d_input, width * channels, roi, offset,
                                                               d_median_rgb.d_buf, median_pitch, roi, mask, anchor,
                                                               d_median_buf.d_buf, NPP_BORDER_REPLICATE, streamContext);
        NPP_CHECK(median_result);
    };

    npp_execution(0);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    const size_t         y_size  = width * height;
    const size_t         uv_size = (width / 2) * height;
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
    (void)width;
    (void)height;
    (void)yuv_size;
    (void)warm_up_runs;
    (void)runs;
    std::cout << "\n(NPP not available - skipping comparison)" << std::endl;
    return common::nppdx_results<uint8_t> {std::vector<uint8_t>(), 0};
#endif
}

// =============================================================================
// CPU reference: 3x3 median (replicate border) + BT.601 RGB -> YUV422
// Match NPPDx: same float BT.601 (0.299,0.587,0.114; U,V + 128), round +0.5 clamp; YUV422 chroma = average of pair.
// =============================================================================

static inline uint8_t clamp_u8(float v) {
    return static_cast<uint8_t>(std::max(0.f, std::min(255.f, v + 0.5f)));
}

static inline uint8_t clamp_u8_trunc(float v) {
    int i = static_cast<int>(v);
    return static_cast<uint8_t>(std::max(0, std::min(255, i)));
}

// Median of 9 using std::sort (no custom network).
static inline uint8_t cpu_median9(uint8_t r00, uint8_t r01, uint8_t r02, uint8_t r10, uint8_t r11, uint8_t r12,
                                  uint8_t r20, uint8_t r21, uint8_t r22) {
    uint8_t v[9] = {r00, r01, r02, r10, r11, r12, r20, r21, r22};
    std::sort(v, v + 9);
    return v[4];
}

// BT.601 (same coeffs as color_conversion_operations.hpp); RGB 0..255 -> Y,U,V float (U,V with 128 offset).
static inline void rgb_to_yuv_bt601_float(uint8_t R, uint8_t G, uint8_t B, float& y, float& u, float& v) {
    float r = static_cast<float>(R), g = static_cast<float>(G), b = static_cast<float>(B);
    y = 0.29900f * r + 0.58700f * g + 0.11400f * b;
    u = -0.16873600f * r - 0.33126399f * g + 0.50000000f * b + 128.0f;
    v = 0.50000000f * r - 0.41868800f * g - 0.08131200f * b + 128.0f;
}

static void cpu_median3x3_rgb_to_yuv422p(const uint8_t* rgb, unsigned int width, unsigned int height,
                                         uint8_t* yuv_out) {
    const unsigned int y_size  = width * height;
    const unsigned int uv_w    = (width + 1) / 2;
    const unsigned int uv_size = uv_w * height;
    uint8_t*           y       = yuv_out;
    uint8_t*           u       = yuv_out + y_size;
    uint8_t*           v       = yuv_out + y_size + uv_size;

    auto at = [&](int x, int y, int c) -> uint8_t {
        int ix = std::max(0, std::min(static_cast<int>(width) - 1, x));
        int iy = std::max(0, std::min(static_cast<int>(height) - 1, y));
        return rgb[(iy * width + ix) * 3 + c];
    };

    for (unsigned int py = 0; py < height; ++py) {
        for (unsigned int px = 0; px < width; ++px) {
            int     x = static_cast<int>(px), yi = static_cast<int>(py);
            uint8_t r00 = at(x - 1, yi - 1, 0), r01 = at(x, yi - 1, 0), r02 = at(x + 1, yi - 1, 0);
            uint8_t r10 = at(x - 1, yi, 0), r11 = at(x, yi, 0), r12 = at(x + 1, yi, 0);
            uint8_t r20 = at(x - 1, yi + 1, 0), r21 = at(x, yi + 1, 0), r22 = at(x + 1, yi + 1, 0);
            uint8_t R = cpu_median9(r00, r01, r02, r10, r11, r12, r20, r21, r22);
            r00       = at(x - 1, yi - 1, 1);
            r01       = at(x, yi - 1, 1);
            r02       = at(x + 1, yi - 1, 1);
            r10       = at(x - 1, yi, 1);
            r11       = at(x, yi, 1);
            r12       = at(x + 1, yi, 1);
            r20       = at(x - 1, yi + 1, 1);
            r21       = at(x, yi + 1, 1);
            r22       = at(x + 1, yi + 1, 1);
            uint8_t G = cpu_median9(r00, r01, r02, r10, r11, r12, r20, r21, r22);
            r00       = at(x - 1, yi - 1, 2);
            r01       = at(x, yi - 1, 2);
            r02       = at(x + 1, yi - 1, 2);
            r10       = at(x - 1, yi, 2);
            r11       = at(x, yi, 2);
            r12       = at(x + 1, yi, 2);
            r20       = at(x - 1, yi + 1, 2);
            r21       = at(x, yi + 1, 2);
            r22       = at(x + 1, yi + 1, 2);
            uint8_t B = cpu_median9(r00, r01, r02, r10, r11, r12, r20, r21, r22);

            float fy, fu, fv;
            rgb_to_yuv_bt601_float(R, G, B, fy, fu, fv);
            y[py * width + px] = clamp_u8(fy);
            if (px % 2 == 0) {
                float fu1 = fu, fv1 = fv;
                if (px + 1 < width) {
                    int     x1 = x + 1;
                    uint8_t R1 = cpu_median9(at(x1 - 1, yi - 1, 0), at(x1, yi - 1, 0), at(x1 + 1, yi - 1, 0),
                                             at(x1 - 1, yi, 0), at(x1, yi, 0), at(x1 + 1, yi, 0), at(x1 - 1, yi + 1, 0),
                                             at(x1, yi + 1, 0), at(x1 + 1, yi + 1, 0));
                    uint8_t G1 = cpu_median9(at(x1 - 1, yi - 1, 1), at(x1, yi - 1, 1), at(x1 + 1, yi - 1, 1),
                                             at(x1 - 1, yi, 1), at(x1, yi, 1), at(x1 + 1, yi, 1), at(x1 - 1, yi + 1, 1),
                                             at(x1, yi + 1, 1), at(x1 + 1, yi + 1, 1));
                    uint8_t B1 = cpu_median9(at(x1 - 1, yi - 1, 2), at(x1, yi - 1, 2), at(x1 + 1, yi - 1, 2),
                                             at(x1 - 1, yi, 2), at(x1, yi, 2), at(x1 + 1, yi, 2), at(x1 - 1, yi + 1, 2),
                                             at(x1, yi + 1, 2), at(x1 + 1, yi + 1, 2));
                    float   y1;
                    rgb_to_yuv_bt601_float(R1, G1, B1, y1, fu1, fv1);
                }
                // GPU: fclampf each pixel's U,V then average pair then (T) truncate
                float u_avg               = (clamp_u8(fu) + clamp_u8(fu1)) * 0.5f;
                float v_avg               = (clamp_u8(fv) + clamp_u8(fv1)) * 0.5f;
                u[(py * uv_w) + (px / 2)] = clamp_u8_trunc(u_avg);
                v[(py * uv_w) + (px / 2)] = clamp_u8_trunc(v_avg);
            }
        }
    }
}

// =============================================================================
// Main
// =============================================================================

template<int Arch>
int median_fused_conversion_example() {
    std::cout << "=== NPPDx Fused 3x3 Median + RGB to YUV422P Example (SM" << Arch << ") ===" << std::endl;

    constexpr unsigned int warm_up_runs = 10;
    constexpr unsigned int runs         = 100;

    constexpr unsigned int width    = 312;
    constexpr unsigned int height   = 376;
    constexpr unsigned int channels = 3;
    constexpr size_t       rgb_size = width * height * channels;
    constexpr size_t       y_size   = width * height;
    constexpr size_t       uv_size  = (width / 2) * height;
    constexpr size_t       yuv_size = y_size + 2 * uv_size;

    std::cout << "Image: " << width << "x" << height << " RGB24 -> YUV422P, median radius r1_0 (3x3)" << std::endl;

    std::vector<uint8_t> h_input = common::load_or_generate_rgb_test_data(width, height);

    common::DevBuf d_input(rgb_size);
    common::DevBuf d_output(yuv_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_input.data(), rgb_size, cudaMemcpyHostToDevice));

    auto nppdx_result =
        run_nppdx_fused<Arch, channels>(d_input.d_buf, d_output.d_buf, width, height, yuv_size, warm_up_runs, runs);
    auto npp_result = run_npp_two_step(d_input.d_buf, width, height, yuv_size, warm_up_runs, runs);

    std::vector<uint8_t> cpu_yuv(yuv_size);
    cpu_median3x3_rgb_to_yuv422p(h_input.data(), width, height, cpu_yuv.data());

    std::cout << "\n--- Perf / timing ---" << std::endl;
    std::cout << "NPPDx fused: " << nppdx_result.avg_time_in_ms << " ms" << std::endl;
    if (!npp_result.output.empty() && npp_result.avg_time_in_ms > 0) {
        std::cout << "NPP two-step: " << npp_result.avg_time_in_ms << " ms" << std::endl;
        std::cout << "Speedup (NPP/NPPDx): " << (npp_result.avg_time_in_ms / nppdx_result.avg_time_in_ms) << "x"
                  << std::endl;
    } else {
        std::cout << "NPP two-step: (skipped)" << std::endl;
    }

    constexpr int    y_max_err  = 10;
    constexpr double y_avg_err  = 2.0;
    constexpr int    uv_max_err = 32;
    constexpr double uv_avg_err = 3.0;
    bool             passed     = true;

    std::cout << "\n--- Comparison vs CPU reference (3x3 median + BT.601, same tolerance as 5x5) ---" << std::endl;
    auto nppdx_y = common::check_error_detailed(nppdx_result.output.data(), cpu_yuv.data(), y_size, 2, y_max_err,
                                                y_avg_err, 1, 1, false, false);
    auto nppdx_u = common::check_error_detailed(nppdx_result.output.data() + y_size, cpu_yuv.data() + y_size, uv_size,
                                                2, uv_max_err, uv_avg_err, 1, 1, false, false);
    auto nppdx_v =
        common::check_error_detailed(nppdx_result.output.data() + y_size + uv_size, cpu_yuv.data() + y_size + uv_size,
                                     uv_size, 2, uv_max_err, uv_avg_err, 1, 1, false, false);
    std::cout << "NPPDx vs CPU: Y max=" << nppdx_y.max_abs_error << " mismatches=" << nppdx_y.error_count
              << "  U max=" << nppdx_u.max_abs_error << " mismatches=" << nppdx_u.error_count
              << "  V max=" << nppdx_v.max_abs_error << " mismatches=" << nppdx_v.error_count << std::endl;
    passed = passed && nppdx_y.passed && nppdx_u.passed && nppdx_v.passed;

    if (!npp_result.output.empty()) {
        auto npp_y = common::check_error_detailed(npp_result.output.data(), cpu_yuv.data(), y_size, 2, y_max_err,
                                                  y_avg_err, 1, 1, false, false);
        auto npp_u = common::check_error_detailed(npp_result.output.data() + y_size, cpu_yuv.data() + y_size, uv_size,
                                                  2, uv_max_err, uv_avg_err, 1, 1, false, false);
        auto npp_v =
            common::check_error_detailed(npp_result.output.data() + y_size + uv_size, cpu_yuv.data() + y_size + uv_size,
                                         uv_size, 2, uv_max_err, uv_avg_err, 1, 1, false, false);
        std::cout << "NPP vs CPU:   Y max=" << npp_y.max_abs_error << " mismatches=" << npp_y.error_count
                  << "  U max=" << npp_u.max_abs_error << " mismatches=" << npp_u.error_count
                  << "  V max=" << npp_v.max_abs_error << " mismatches=" << npp_v.error_count << std::endl;
        passed = passed && npp_y.passed && npp_u.passed && npp_v.passed;
    }

    common::write_output_file("cow_median_yuv422p_nppdx.yuv422", nppdx_result.output.data(), yuv_size);
    if (!npp_result.output.empty())
        common::write_output_file("cow_median_yuv422p_npp.yuv422", npp_result.output.data(), yuv_size);

    std::cout << "\nResult: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}

template<int Arch>
struct median_fused_conversion_functor {
    int operator()() const { return median_fused_conversion_example<Arch>(); }
};

int main() {
    return common::run_example_with_sm<median_fused_conversion_functor>();
}
