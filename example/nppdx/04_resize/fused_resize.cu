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


// Fused Box Blur + 2x Resize + Color Convert Example - NPPDx vs NPP comparison
// Pipeline: RGB24 (312x376) -> Blur -> 2x Resize -> YUV -> YUV422P (624x752)
//
// Placeholder architecture: this kernel keeps a full resized RGB tile in shared memory before color convert
// and exgest. That only works for modest tiles; in practice there is often not enough SMEM to stage the
// entire resized image, so a dedicated path (e.g. exgestResize - resize while writing output / streaming)
// will replace this pattern once the sampling math is nailed down.
//
// Pass/fail: NPPDx vs CPU reference (NPPDx clamp model: no RGB clamp before BT.601; Y/U/V fclamp at exgest).
// NPP three-step: timing vs NPPDx; CPU also builds an NPP-style RGB-saturate-before-YCbCr path for sanity.
//
// Matching policy (NPP vs NPPDx clamp order): For box blur and median, each output sample is a convex combination of
// inputs in [0,255], so sharpened/high-pass aside, linear RGB before YUV stays in-gamut - NPP's pre-YUV clamp and NPPDx's
// exgest-only clamp often agree in practice (intermediate-value / hull argument). Sharpen and similar filters can
// overshoot; those examples use looser NPP comparison after validation (see sharpen_fused_conversion.cu). The same
// applies to strict color-convert / round-trip limits when the pipeline can push values outside nominal 8-bit range.

#include <nppdx.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../common.hpp"
#include "../common/cpu_resize_reference.hpp"

#ifdef NPPDX_EXAMPLE_HAS_NPP
#    include <nppi.h>
#endif

namespace {

    constexpr float scale_8bit_to_10bit = 4.0f;
    constexpr float scale_10bit_to_8bit = 0.25f;
    constexpr float uv_offset_10bit     = 512.0f;

    // 5x5 box blur in float (matches NPPDx smem path: average of 25 samples, no uint8 rounding between ops).
    inline std::vector<float> cpu_box_blur_5x5_float(const std::vector<uint8_t>& input, unsigned int width,
                                                     unsigned int height, unsigned int ch) {
        constexpr int      halo  = 2;
        constexpr float    inv25 = 1.0f / 25.0f;
        std::vector<float> out(input.size());
        const int          w = static_cast<int>(width);
        const int          h = static_cast<int>(height);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                for (unsigned int c = 0; c < ch; ++c) {
                    float sum = 0.f;
                    for (int dy = -halo; dy <= halo; ++dy) {
                        for (int dx = -halo; dx <= halo; ++dx) {
                            int nx = std::clamp(x + dx, 0, w - 1);
                            int ny = std::clamp(y + dy, 0, h - 1);
                            sum += static_cast<float>(
                                input[(static_cast<size_t>(ny) * width + static_cast<unsigned>(nx)) * ch + c]);
                        }
                    }
                    out[(static_cast<size_t>(y) * width + static_cast<unsigned>(x)) * ch + c] = sum * inv25;
                }
            }
        }
        return out;
    }

    // Match ingest_exgest fclampf<bpp_8u> (used when writing Y/U/V at exgest, not for RGB before color_convert).
    inline float fclampf_8u(float value) {
        return std::fmin(std::fmax(value + 0.5f, 0.0f), 255.0f);
    }

    // NPP-style: saturate linear RGB to [0,255] before RGB->YCbCr. NPPDx does not do this.
    inline float saturate_rgb_npp_style(float x) {
        return std::fmin(std::fmax(x, 0.0f), 255.0f);
    }

    inline void rgb_float_to_yuv_bt601_8u_path(float r, float g, float b, float& y_out, float& u_out, float& v_out) {
        const float r10 = r * scale_8bit_to_10bit;
        const float g10 = g * scale_8bit_to_10bit;
        const float b10 = b * scale_8bit_to_10bit;
        const float y10 = 0.29900f * r10 + 0.58700f * g10 + 0.11400f * b10;
        const float u10 = -0.16873600f * r10 - 0.33126399f * g10 + 0.50000000f * b10 + uv_offset_10bit;
        const float v10 = 0.50000000f * r10 - 0.41868800f * g10 - 0.08131200f * b10 + uv_offset_10bit;
        y_out           = y10 * scale_10bit_to_8bit;
        u_out           = u10 * scale_10bit_to_8bit;
        v_out           = v10 * scale_10bit_to_8bit;
    }

    // Blur -> resize -> ColorConvert(rgb,yuv_bt601,8u,8u) -> YUV422P exgest.
    //
    // ClampRgbBeforeYuv == false (NPPDx): unclamped float RGB through color_convert; Y/U/V clamp at exgest (fclampf per
    // channel, then 4:2:2 average of clamped U/V). With blur+nearest here, RGB stays in [0,255] anyway, so this matches
    // NPP's pre-YUV clamp in practice; the flag still documents the library contract.
    //
    // ClampRgbBeforeYuv == true (NPP): saturate R,G,B to [0,255] before BT.601, then same exgest (for pipelines where RGB
    // might leave [0,255], e.g. after sharpen - see sharpen_fused_conversion.cu for looser NPP tolerances there).
    template<bool ClampRgbBeforeYuv, unsigned int J, unsigned int K, nppdx::interpolation_method Method>
    std::vector<uint8_t> cpu_fused_blur_resize_yuv422p_reference(const std::vector<uint8_t>& rgb_input,
                                                                 unsigned int in_w, unsigned int in_h,
                                                                 unsigned int out_w, unsigned int out_h) {
        constexpr unsigned int ch      = 3;
        auto                   blurred = cpu_box_blur_5x5_float(rgb_input, in_w, in_h, ch);
        auto resized = nppdx::example::cpu_reference::resize<J, K, Method, ch, /*UseRadialLanczos2D=*/false, float>(
            blurred, static_cast<int>(in_w), static_cast<int>(in_h));

        const size_t         y_plane_size  = static_cast<size_t>(out_w) * out_h;
        const size_t         uv_w          = out_w / 2;
        const size_t         uv_plane_size = uv_w * out_h;
        std::vector<uint8_t> yuv(y_plane_size + 2 * uv_plane_size);
        uint8_t*             y_dst = yuv.data();
        uint8_t*             u_dst = yuv.data() + y_plane_size;
        uint8_t*             v_dst = yuv.data() + y_plane_size + uv_plane_size;

        std::vector<float> yf(y_plane_size), uf(y_plane_size), vf(y_plane_size);
        for (unsigned int y = 0; y < out_h; ++y) {
            for (unsigned int x = 0; x < out_w; ++x) {
                const size_t ri = (static_cast<size_t>(y) * out_w + x) * ch;
                float        r  = resized[ri];
                float        g  = resized[ri + 1];
                float        b  = resized[ri + 2];
                if constexpr (ClampRgbBeforeYuv) {
                    r = saturate_rgb_npp_style(r);
                    g = saturate_rgb_npp_style(g);
                    b = saturate_rgb_npp_style(b);
                }
                float yv, uv, vv;
                rgb_float_to_yuv_bt601_8u_path(r, g, b, yv, uv, vv);
                const size_t ti = static_cast<size_t>(y) * out_w + x;
                // Exgest: clamp Y/U/V (same order as GPU - per-pixel clamp, then 4:2:2 pair average on clamped U/V).
                yf[ti] = fclampf_8u(yv);
                uf[ti] = fclampf_8u(uv);
                vf[ti] = fclampf_8u(vv);
            }
        }

        for (unsigned int y = 0; y < out_h; ++y) {
            for (unsigned int x = 0; x < out_w; ++x) {
                y_dst[static_cast<size_t>(y) * out_w + x] =
                    static_cast<uint8_t>(yf[static_cast<size_t>(y) * out_w + x]);
            }
        }

        for (unsigned int y = 0; y < out_h; ++y) {
            for (unsigned int j = 0; j < uv_w; ++j) {
                const unsigned int x0 = j * 2;
                const size_t       t0 = static_cast<size_t>(y) * out_w + x0;
                float              ua, va;
                if (x0 + 1 < out_w) {
                    ua = (uf[t0] + uf[t0 + 1]) * 0.5f;
                    va = (vf[t0] + vf[t0 + 1]) * 0.5f;
                } else {
                    ua = uf[t0];
                    va = vf[t0];
                }
                const size_t ui = static_cast<size_t>(y) * uv_w + j;
                u_dst[ui]       = static_cast<uint8_t>(ua);
                v_dst[ui]       = static_cast<uint8_t>(va);
            }
        }

        return yuv;
    }

} // namespace

// =============================================================================
// NPPDx Fused Kernel: Ingest -> Box Blur -> Resize -> Color Convert -> Exgest
// (SMEM holds resized RGB until exgest; see file header - expect exgestResize-style fusion later.)
// =============================================================================

template<typename Ingest, typename BoxBlur, typename Resize, typename ColorConvert, typename Exgest,
         typename InputTileStorageType, typename BlurredTileStorageType, typename IntermediateTileStorageType,
         typename ResizedTileStorageType, unsigned int NumThreads>
__launch_bounds__(NumThreads) __global__
    void fused_blur_resize_convert_kernel(const uint8_t* input, uint8_t* output, const unsigned int input_width,
                                          const unsigned int input_height, const unsigned int output_width,
                                          const unsigned int output_height) {

    extern __shared__ unsigned char smem[];


    auto tile_storage =
        nppdx::shared_memory::slice_into_tile_storage<InputTileStorageType, BlurredTileStorageType,
                                                      IntermediateTileStorageType, ResizedTileStorageType>(smem);
    auto input_tile        = std::get<0>(tile_storage);
    auto blurred_tile      = std::get<1>(tile_storage);
    auto intermediate_tile = std::get<2>(tile_storage);
    auto resized_tile      = std::get<3>(tile_storage);

    // Step 1: Ingest RGB24 with halo
    Ingest().execute(input, input_tile, input_width, input_height);

    // Step 2: Box blur (same tile size)
    BoxBlur().execute(input_tile, blurred_tile, input_width, input_height);

    // Step 3: Resize 2x (different output tile size!)
    Resize().execute(blurred_tile, intermediate_tile, resized_tile, input_width, input_height);

    // Step 4: Color convert RGB -> YUV (in-place on resized_tile)
    ColorConvert().execute(resized_tile, output_width, output_height);

    // Step 5: Exgest to YUV422P
    Exgest().execute(resized_tile, output, output_width, output_height);
}

// =============================================================================
// NPPDx Fused Execution
// =============================================================================

template<int Arch, unsigned int NumChannels, unsigned int KernelW, unsigned int KernelH, unsigned int J, unsigned int K,
         nppdx::interpolation_method Method>
common::nppdx_results<uint8_t> run_nppdx_fused(uint8_t* d_input, uint8_t* d_output, const unsigned int input_width,
                                               const unsigned int input_height, const unsigned int output_width,
                                               const unsigned int output_height, const size_t yuv_size,
                                               unsigned int warm_up_runs, unsigned int runs) {

    // Input tile size (must be divisible by scale ratio for output)
    constexpr unsigned int tile_x = 16;
    constexpr unsigned int tile_y = 16;

    // Precompute the halo for the resize operation
    using ResizePre            = decltype(nppdx::Function<nppdx::function::resize>() + nppdx::Resize<J, K, Method>() +
                               nppdx::TileSize<tile_x, tile_y>() + nppdx::SM<Arch>() + nppdx::Block());
    constexpr auto resize_halo = ResizePre::local_halo;

    using BoxBlurPre = decltype(nppdx::Function<nppdx::function::box_blur>() + nppdx::BoxBlur<KernelW, KernelH>() +
                                nppdx::TileSize<tile_x, tile_y>() + nppdx::SM<Arch>() + nppdx::Block());
    constexpr auto box_blur_halo = BoxBlurPre::local_halo;

    constexpr auto total_halo = box_blur_halo + resize_halo;

    // Ingest RGB24 with combined halo
    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, total_halo)() +
                            NPPDX_MAKE_HALO(nppdx::CumulativeHalo, total_halo)() + nppdx::SM<Arch>() + nppdx::Block());

    // Box blur (consumes blur_halo, leaves resize_halo for next op)
    using BoxBlur = decltype(BoxBlurPre() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, total_halo)() +
                             NPPDX_MAKE_HALO(nppdx::CumulativeHalo, total_halo)());


    // Resize 2x (consumes resize_halo from remaining halo)
    using Resize = decltype(ResizePre() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, total_halo)() +
                            NPPDX_MAKE_HALO(nppdx::CumulativeHalo, resize_halo)());

    using InputTileStorageType        = nppdx::input_storage_of_t<Ingest>;
    using BlurredTileStorageType      = nppdx::output_storage_of_t<BoxBlur>;
    using IntermediateTileStorageType = nppdx::temp_storage_of_t<Resize>;
    using ResizedTileStorageType      = nppdx::output_storage_of_t<Resize>;

    constexpr unsigned int eff_tile_x = InputTileStorageType::ChannelSliceType::width;
    constexpr unsigned int eff_tile_y = InputTileStorageType::ChannelSliceType::height;
    constexpr unsigned int out_tile_x = ResizedTileStorageType::ChannelSliceType::width;
    constexpr unsigned int out_tile_y = ResizedTileStorageType::ChannelSliceType::height;

    [[maybe_unused]] constexpr auto empty_halo = nppdx::Halo4::make_empty();

    // Color convert RGB -> YUV (operates on resized tile)
    using ColorConvert =
        decltype(nppdx::Function<nppdx::function::color_convert>() +
                 nppdx::ColorConvert<nppdx::color_space::rgb, nppdx::color_space::yuv_bt601, nppdx::bit_depth::bpp_8u,
                                     nppdx::bit_depth::bpp_8u>() +
                 nppdx::TileSize<out_tile_x, out_tile_y>() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, empty_halo)() +
                 NPPDX_MAKE_HALO(nppdx::CumulativeHalo, empty_halo)() + nppdx::SM<Arch>() + nppdx::Block());

    // Exgest to YUV422P (output tile size)
    using Exgest =
        decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                 nppdx::OutputFormat<nppdx::packing_format::yuv422p>() + nppdx::TileSize<out_tile_x, out_tile_y>() +
                 NPPDX_MAKE_HALO(nppdx::MemoryHalo, empty_halo)() +
                 NPPDX_MAKE_HALO(nppdx::CumulativeHalo, empty_halo)() + nppdx::SM<Arch>() + nppdx::Block());

    std::cout << "Input tile: " << tile_x << "x" << tile_y << " with total_halo=" << total_halo.left_top.x << ", "
              << total_halo.right_bottom.x << ", " << total_halo.left_top.y << ", " << total_halo.right_bottom.y
              << std::endl;
    std::cout << "Effective input tile: " << eff_tile_x << "x" << eff_tile_y << std::endl;
    std::cout << "Output tile (after 2x resize): " << out_tile_x << "x" << out_tile_y << std::endl;
    std::cout << "Scale: " << J << ":" << K << std::endl;

    constexpr size_t smem_size =
        nppdx::shared_memory::compute_total_tile_storage<InputTileStorageType, BlurredTileStorageType,
                                                         IntermediateTileStorageType, ResizedTileStorageType>();
    std::cout << "Shared memory: " << smem_size << " bytes per block" << std::endl;

    //Need to get the maximum block dimension for the exgest operation
    constexpr dim3 block_dim = (J > K) ? Ingest::block_dim : Exgest::block_dim;
    const dim3     grid_dim  = (J > K) ? Ingest::calculate_grid_dim(input_width, input_height)
                                       : Exgest::calculate_grid_dim(output_width, output_height);

    constexpr unsigned int num_threads = block_dim.x * block_dim.y * block_dim.z;


    std::cout << "Grid: " << grid_dim.x << "x" << grid_dim.y << " | Block: " << block_dim.x << std::endl;

    auto kernel_ptr =
        fused_blur_resize_convert_kernel<Ingest, BoxBlur, Resize, ColorConvert, Exgest, InputTileStorageType,
                                         BlurredTileStorageType, IntermediateTileStorageType, ResizedTileStorageType,
                                         num_threads>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    std::cout << "\nRunning NPPDx fused (blur + resize + convert)..." << std::endl;

    auto nppdx_execution = [&](cudaStream_t stream) {
        kernel_ptr<<<grid_dim, block_dim, smem_size, stream>>>(d_input, d_output, input_width, input_height,
                                                               output_width, output_height);
    };

    // Correctness run
    nppdx_execution(0);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<uint8_t> h_nppdx_output(yuv_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_nppdx_output.data(), d_output, yuv_size, cudaMemcpyDeviceToHost));

    auto ms = common::measure_execution_ms(nppdx_execution, warm_up_runs, runs, 0);
    std::cout << "NPPDx fused time: " << std::fixed << std::setprecision(6) << (ms / runs) << " ms" << std::endl;

    return common::nppdx_results<uint8_t> {h_nppdx_output, ms / runs};
}

// =============================================================================
// NPP Three-Step Execution: Box Blur -> Resize -> RGB to YUV422P
// =============================================================================

template<unsigned int KernelW, unsigned int KernelH, nppdx::interpolation_method Method>
common::nppdx_results<uint8_t> run_npp_three_step(uint8_t* d_input, const unsigned int input_width,
                                                  const unsigned int input_height, const unsigned int output_width,
                                                  const unsigned int output_height, const size_t yuv_size,
                                                  unsigned int warm_up_runs, unsigned int runs) {

#ifdef NPPDX_EXAMPLE_HAS_NPP
    std::cout << "\nRunning NPP three-step (blur -> resize -> convert)..." << std::endl;

    constexpr unsigned int channels = 3;

    // Allocate intermediate buffers
    int            blurred_pitch = input_width * channels;
    int            resized_pitch = output_width * channels;
    common::DevBuf d_blurred_rgb(blurred_pitch * input_height);
    common::DevBuf d_resized_rgb(resized_pitch * output_height);

    // YUV422P output planes
    int            y_pitch = output_width;
    int            u_pitch = (output_width + 1) / 2;
    int            v_pitch = (output_width + 1) / 2;
    common::DevBuf d_y(y_pitch * output_height);
    common::DevBuf d_u(u_pitch * output_height);
    common::DevBuf d_v(v_pitch * output_height);

    NppiSize  input_roi   = {static_cast<int>(input_width), static_cast<int>(input_height)};
    NppiSize  output_roi  = {static_cast<int>(output_width), static_cast<int>(output_height)};
    NppiSize  mask        = {static_cast<int>(KernelW), static_cast<int>(KernelH)};
    NppiPoint anchor      = {static_cast<int>(KernelW / 2), static_cast<int>(KernelH / 2)};
    NppiPoint offset      = {0, 0};
    NppiRect  input_rect  = {0, 0, static_cast<int>(input_width), static_cast<int>(input_height)};
    NppiRect  output_rect = {0, 0, static_cast<int>(output_width), static_cast<int>(output_height)};

    // NPP stream context
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

    NppiInterpolationMode resize_method = NPPI_INTER_LINEAR;
    switch (Method) {
        case nppdx::interpolation_method::nearest: resize_method = NPPI_INTER_NN; break;
        case nppdx::interpolation_method::bilinear: resize_method = NPPI_INTER_LINEAR; break;
        case nppdx::interpolation_method::bicubic: resize_method = NPPI_INTER_CUBIC; break;
        // Throughput comparison only; pixel match vs NPPDx is not asserted for Lanczos (see file header).
        case nppdx::interpolation_method::lanczos3: resize_method = NPPI_INTER_LANCZOS; break;
        default: break;
    }

    Npp8u* pDstYUV[3]    = {d_y.d_buf, d_u.d_buf, d_v.d_buf};
    int    dstPitches[3] = {y_pitch, u_pitch, v_pitch};

    auto npp_execution = [&](cudaStream_t) {
        // Step 1: Box blur on RGB
        auto blur_result =
            nppiFilterBoxBorder_8u_C3R_Ctx(d_input, input_width * channels, input_roi, offset, d_blurred_rgb.d_buf,
                                           blurred_pitch, input_roi, mask, anchor, NPP_BORDER_REPLICATE, streamContext);
        NPP_CHECK(blur_result);

        // Step 2: Resize
        auto resize_result =
            nppiResize_8u_C3R_Ctx(d_blurred_rgb.d_buf, blurred_pitch, input_roi, input_rect, d_resized_rgb.d_buf,
                                  resized_pitch, output_roi, output_rect, resize_method, streamContext);
        NPP_CHECK(resize_result);

        // Step 3: RGB to full-range BT.601 YCbCr422P conversion.
        auto convert_result = nppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx(d_resized_rgb.d_buf, resized_pitch, pDstYUV,
                                                                  dstPitches, output_roi, streamContext);
        NPP_CHECK(convert_result);
    };

    // Correctness run
    npp_execution(0);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy YUV422P output to host
    const size_t y_size  = output_width * output_height;
    const size_t uv_size = (output_width / 2) * output_height;

    std::vector<uint8_t> h_npp_output(yuv_size);
    cudaMemcpy2D(h_npp_output.data(), output_width, d_y.d_buf, y_pitch, output_width, output_height,
                 cudaMemcpyDeviceToHost);
    cudaMemcpy2D(h_npp_output.data() + y_size, output_width / 2, d_u.d_buf, u_pitch, output_width / 2, output_height,
                 cudaMemcpyDeviceToHost);
    cudaMemcpy2D(h_npp_output.data() + y_size + uv_size, output_width / 2, d_v.d_buf, v_pitch, output_width / 2,
                 output_height, cudaMemcpyDeviceToHost);

    auto npp_ms = common::measure_execution_ms(npp_execution, warm_up_runs, runs, 0);
    std::cout << "NPP three-step time: " << std::fixed << std::setprecision(6) << (npp_ms / runs) << " ms" << std::endl;

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
int fused_resize_example() {
    std::cout << "=== NPPDx Fused Blur + Resize + Convert Example (SM" << Arch << ") ===" << std::endl;

    constexpr unsigned int warm_up_runs = 10;
    constexpr unsigned int runs         = 100;

    // Input dimensions (cow image)
    constexpr unsigned int KernelW      = 5;
    constexpr unsigned int KernelH      = 5;
    constexpr unsigned int input_width  = 312;
    constexpr unsigned int input_height = 376;
    constexpr unsigned int channels     = 3;

    // Scale factor: J=1, K=2 -> 2x upscale
    constexpr unsigned int J      = 1;
    constexpr unsigned int K      = 2;
    constexpr auto         method = nppdx::interpolation_method::nearest;

    // Output dimensions (2x)
    constexpr unsigned int output_width  = input_width * K / J;  // 624
    constexpr unsigned int output_height = input_height * K / J; // 752

    constexpr size_t rgb_size = input_width * input_height * channels;

    // YUV422P output sizes
    constexpr size_t y_size   = output_width * output_height;
    constexpr size_t uv_size  = (output_width / 2) * output_height;
    constexpr size_t yuv_size = y_size + 2 * uv_size;

    std::cout << "Input: " << input_width << "x" << input_height << " RGB24 (" << rgb_size << " bytes)" << std::endl;
    std::cout << "Output: " << output_width << "x" << output_height << " YUV422P (" << yuv_size << " bytes)"
              << std::endl;
    std::cout << "Box filter: " << KernelW << "x" << KernelH << std::endl;
    std::cout << "Resize: " << J << ":" << K << " " << common::interpolation_method_to_string(method)
              << " interpolation method" << std::endl;

    // Load cow test image
    std::vector<uint8_t> h_input = common::load_or_generate_rgb_test_data(input_width, input_height);

    // Allocate device memory
    common::DevBuf d_input(rgb_size);
    common::DevBuf d_output(yuv_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_input.data(), rgb_size, cudaMemcpyHostToDevice));

    // Run NPPDx fused pipeline
    auto nppdx_result = run_nppdx_fused<Arch, channels, KernelW, KernelH, J, K, method>(
        d_input.d_buf, d_output.d_buf, input_width, input_height, output_width, output_height, yuv_size, warm_up_runs,
        runs);

    // Run NPP three-step pipeline (timing / optional cross-check vs CPU; not the pass/fail golden).
    auto npp_result = run_npp_three_step<KernelW, KernelH, method>(
        d_input.d_buf, input_width, input_height, output_width, output_height, yuv_size, warm_up_runs, runs);

    std::vector<uint8_t> cpu_ref_nppdx =
        cpu_fused_blur_resize_yuv422p_reference</*ClampRgbBeforeYuv=*/false, J, K, method>(
            h_input, input_width, input_height, output_width, output_height);

    std::cout << "\n--- Validation vs CPU (NPPDx semantics: exgest-only YUV clamp, no pre-YUV RGB clamp) ---"
              << std::endl;
    auto y_stats = common::check_error_detailed(nppdx_result.output.data(), cpu_ref_nppdx.data(), y_size, 3, 10, 2.0, 1,
                                                1, false, false);
    std::cout << "Y plane:  max_diff=" << y_stats.max_abs_error << ", avg_diff=" << y_stats.avg_abs_error
              << ", mismatches=" << y_stats.error_count << "/" << y_size << std::endl;

    auto u_stats = common::check_error_detailed(nppdx_result.output.data() + y_size, cpu_ref_nppdx.data() + y_size,
                                                uv_size, 3, 10, 2.0, 1, 1, false, false);
    std::cout << "U plane:  max_diff=" << u_stats.max_abs_error << ", avg_diff=" << u_stats.avg_abs_error
              << ", mismatches=" << u_stats.error_count << "/" << uv_size << std::endl;

    auto v_stats =
        common::check_error_detailed(nppdx_result.output.data() + y_size + uv_size,
                                     cpu_ref_nppdx.data() + y_size + uv_size, uv_size, 3, 10, 2.0, 1, 1, false, false);
    std::cout << "V plane:  max_diff=" << v_stats.max_abs_error << ", avg_diff=" << v_stats.avg_abs_error
              << ", mismatches=" << v_stats.error_count << "/" << uv_size << std::endl;

    auto overall_stats = common::check_error_detailed(nppdx_result.output.data(), cpu_ref_nppdx.data(), yuv_size, 3, 10,
                                                      2.0, 1, 1, false, false);
    std::cout << "Overall:  max_diff=" << overall_stats.max_abs_error << ", avg_diff=" << overall_stats.avg_abs_error
              << ", mismatches=" << overall_stats.error_count << "/" << yuv_size << std::endl;

    bool passed = overall_stats.passed;

    if (!npp_result.output.empty()) {
        if (npp_result.avg_time_in_ms > 0 && nppdx_result.avg_time_in_ms > 0) {
            float speedup = npp_result.avg_time_in_ms / nppdx_result.avg_time_in_ms;
            std::cout << "\nSpeedup (NPP/NPPDx): " << speedup << "x" << std::endl;
        }
        std::vector<uint8_t> cpu_ref_npp =
            cpu_fused_blur_resize_yuv422p_reference</*ClampRgbBeforeYuv=*/true, J, K, method>(
                h_input, input_width, input_height, output_width, output_height);

        std::cout << "\n--- NPP three-step vs CPU (global pipeline sanity) ---" << std::endl;
        auto npp_vs_nppdx_cpu = common::check_error_detailed(npp_result.output.data(), cpu_ref_nppdx.data(), yuv_size,
                                                             3, 10, 2.0, 1, 1, false, false);
        auto npp_vs_npp_rgb_clamp_cpu = common::check_error_detailed(npp_result.output.data(), cpu_ref_npp.data(),
                                                                     yuv_size, 3, 10, 2.0, 1, 1, false, false);
        std::cout << "vs CPU (NPPDx semantics, no pre-YUV RGB clamp): max_diff=" << npp_vs_nppdx_cpu.max_abs_error
                  << ", mismatches=" << npp_vs_nppdx_cpu.error_count << "/" << yuv_size << std::endl;
        std::cout << "vs CPU (NPP-style RGB saturate before YUV):     max_diff="
                  << npp_vs_npp_rgb_clamp_cpu.max_abs_error << ", mismatches=" << npp_vs_npp_rgb_clamp_cpu.error_count
                  << "/" << yuv_size << std::endl;
        if (npp_vs_nppdx_cpu.error_count == npp_vs_npp_rgb_clamp_cpu.error_count &&
            npp_vs_nppdx_cpu.max_abs_error == npp_vs_npp_rgb_clamp_cpu.max_abs_error) {
            std::cout << "  (Same stats: blur+nearest keeps RGB in [0,255], so NPP pre-YUV clamp is a no-op vs NPPDx "
                         "path.)\n";
        }

        if (!passed && npp_vs_nppdx_cpu.error_count == 0) {
            std::cout << "Note: NPP matches global CPU reference; remaining error is fused NPPDx vs that reference "
                         "(tiling / resize / ordering), not NPP vs NPPDx clamp policy on this input.\n";
        }
    }

    // Write output files
    common::write_output_file("cow_blur_resize_convert_nppdx.yuv422", nppdx_result.output.data(), yuv_size);
    common::write_output_file("cow_blur_resize_convert_cpu_ref_nppdx.yuv422", cpu_ref_nppdx.data(), yuv_size);
    if (!npp_result.output.empty()) {
        common::write_output_file("cow_blur_resize_convert_npp.yuv422", npp_result.output.data(), yuv_size);
    }

    std::cout << "\nResult: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}

template<int Arch>
struct fused_resize_example_functor {
    int operator()() const { return fused_resize_example<Arch>(); }
};

int main() {
    return common::run_example_with_sm<fused_resize_example_functor>();
}
