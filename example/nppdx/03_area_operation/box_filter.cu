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


// Box Blur Example - NPPDx vs NPP comparison

#include <nppdx.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

#include "../common.hpp"

#ifdef NPPDX_EXAMPLE_HAS_NPP
#    include <nppi.h>
#endif

template<typename Ingest, typename BoxBlur, typename Exgest, typename InputTileStorageType,
         typename OutputTileStorageType>
__global__ void box_blur_kernel(const uint8_t* input, uint8_t* output, const unsigned int width,
                                const unsigned int height) {
    extern __shared__ unsigned char smem[];

    auto tile_storage =
        nppdx::shared_memory::slice_into_tile_storage<InputTileStorageType, OutputTileStorageType>(smem);
    auto input_channels  = std::get<0>(tile_storage);
    auto output_channels = std::get<1>(tile_storage);

    Ingest().execute(input, input_channels, width, height);

    BoxBlur().execute(input_channels, output_channels, width, height);

    Exgest().execute(output_channels, output, width, height);
}

template<unsigned int KernelW, unsigned int KernelH>
inline common::nppdx_results<uint8_t> cpu_box_blur(const std::vector<uint8_t>& input, const unsigned int width,
                                                   const unsigned int height, const unsigned int channels) {
    std::vector<uint8_t> output(input.size());

    constexpr int area_w = static_cast<int>(KernelW) / 2;
    constexpr int area_h = static_cast<int>(KernelH) / 2;

    for (unsigned int y = 0; y < height; ++y) {
        for (unsigned int x = 0; x < width; ++x) {
            for (unsigned int c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int dy = -area_h; dy <= area_h; ++dy) {
                    for (int dx = -area_w; dx <= area_w; ++dx) {
                        int nx = std::clamp(static_cast<int>(x) + dx, 0, static_cast<int>(width) - 1);
                        int ny = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(height) - 1);
                        sum += input[(ny * width + nx) * channels + c];
                    }
                }
                output[(y * width + x) * channels + c] = static_cast<uint8_t>(sum / (KernelW * KernelH) + 0.5f);
            }
        }
    }
    return common::nppdx_results<uint8_t> {output, 0};
}

template<int Arch, unsigned int NumChannels, unsigned int KernelW, unsigned int KernelH>
common::nppdx_results<uint8_t> run_nppdx_box_blur(uint8_t* d_input, uint8_t* d_output, const unsigned int width,
                                                  const unsigned int height, const size_t rgb_size,
                                                  unsigned int warm_up_runs, unsigned int runs) {
    constexpr unsigned int tile_x = 48;
    constexpr unsigned int tile_y = 48;
    constexpr unsigned int halo_x = (KernelW - 1) / 2;
    constexpr unsigned int halo_y = (KernelH - 1) / 2;

    using UniHalo = nppdx::Int2D<halo_x, halo_y>;

    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                            nppdx::MemoryHalo<UniHalo, UniHalo>() + nppdx::SM<Arch>() + nppdx::Block());

    using BoxBlur = decltype(nppdx::Function<nppdx::function::box_blur>() + nppdx::BoxBlur<3, 3>() +
                             nppdx::TileSize<tile_x, tile_y>() + nppdx::MemoryHalo<UniHalo, UniHalo>() +
                             nppdx::CumulativeHalo<UniHalo, UniHalo>() + nppdx::SM<Arch>() + nppdx::Block());


    using Exgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                            nppdx::OutputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                            nppdx::MemoryHalo<UniHalo, UniHalo>() + nppdx::SM<Arch>() + nppdx::Block());

    static_assert(nppdx::is_supported_v<Ingest, Arch>, "Ingest not supported");
    static_assert(nppdx::is_supported_v<BoxBlur, Arch>, "BoxBlur not supported");
    static_assert(nppdx::is_supported_v<Exgest, Arch>, "Exgest not supported");

    using InputTileStorageType  = nppdx::input_storage_of_t<Ingest>;
    using OutputTileStorageType = nppdx::output_storage_of_t<BoxBlur>;

    std::cout << "Image: " << width << "x" << height << " RGB24 (" << rgb_size << " bytes)" << std::endl;
    std::cout << "Tile: " << tile_x << "x" << tile_y << " with halo=" << halo_x << "," << halo_y << "," << halo_x << ","
              << halo_y << std::endl;
    std::cout << "Input storage tile: " << Ingest::input_storage.size.x << "x" << Ingest::input_storage.size.y
              << std::endl;
    std::cout << "Output storage tile: " << BoxBlur::output_storage.size.x << "x" << BoxBlur::output_storage.size.y
              << std::endl;

    constexpr size_t smem_size =
        nppdx::shared_memory::compute_total_tile_storage<InputTileStorageType, OutputTileStorageType>();
    std::cout << "Shared memory: " << smem_size << " bytes per block" << std::endl;

    constexpr dim3 block_dim = Ingest::block_dim;
    const dim3     grid_dim  = Ingest::calculate_grid_dim(width, height);

    std::cout << "Grid: " << grid_dim.x << "x" << grid_dim.y << " | Block: " << block_dim.x << "x" << block_dim.y
              << std::endl;

    auto kernel_ptr = box_blur_kernel<Ingest, BoxBlur, Exgest, InputTileStorageType, OutputTileStorageType>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    std::cout << "\nRunning NPPDx box blur..." << std::endl;

    auto nppdx_execution = [&](cudaStream_t stream) {
        kernel_ptr<<<grid_dim, block_dim, smem_size, stream>>>(d_input, d_output, width, height);
    };

    //Correctness run
    nppdx_execution(0);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<uint8_t> h_nppdx_output(rgb_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_nppdx_output.data(), d_output, rgb_size, cudaMemcpyDeviceToHost));


    auto ms = common::measure_execution_ms(nppdx_execution, warm_up_runs, runs, 0);
    std::cout << "NPPDx time: " << ms / runs << " ms" << std::endl;


    return common::nppdx_results<uint8_t> {h_nppdx_output, ms / runs};
}
template<unsigned int KernelW, unsigned int KernelH>
common::nppdx_results<uint8_t> run_npp_box_blur(uint8_t* d_input, const unsigned int width, const unsigned int height,
                                                const unsigned int channels, const size_t rgb_size,
                                                unsigned int warm_up_runs, unsigned int runs) {
#ifdef NPPDX_EXAMPLE_HAS_NPP
    std::cout << "\nRunning NPP box blur..." << std::endl;

    common::DevBuf d_npp_output(rgb_size);

    NppiSize         roi    = {static_cast<int>(width), static_cast<int>(height)};
    NppiSize         mask   = {KernelW, KernelH};
    NppiPoint        anchor = {1, 1};
    NppStreamContext streamContext;
    streamContext.hStream = cudaStream_t(0);
    NppiPoint offset      = {0, 0};

    auto npp_execution = [&](cudaStream_t) {
        auto npp_result =
            nppiFilterBoxBorder_8u_C3R_Ctx(d_input, width * channels, roi, offset, d_npp_output.d_buf, width * channels,
                                           roi, mask, anchor, NPP_BORDER_REPLICATE, streamContext);
        NPP_CHECK(npp_result);
    };

    npp_execution(0);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());


    std::vector<uint8_t> h_npp_output(rgb_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_npp_output.data(), d_npp_output.d_buf, rgb_size, cudaMemcpyDeviceToHost));

    auto npp_ms = common::measure_execution_ms(npp_execution, warm_up_runs, runs, 0);
    std::cout << "NPP time: " << npp_ms / runs << " ms" << std::endl;

    return common::nppdx_results<uint8_t> {h_npp_output, npp_ms / runs};
#else
    std::cout << "\n(NPP not available - skipping comparison)" << std::endl;
    return common::nppdx_results<uint8_t> {std::vector<uint8_t>(), 0};
#endif
}

template<int Arch>
int box_filter_example() {
    std::cout << "=== NPPDx Box Blur Example (SM" << Arch << ") ===" << std::endl;

    constexpr unsigned int warm_up_runs = 10;
    constexpr unsigned int runs         = 100;

    constexpr unsigned int KernelW  = 3;
    constexpr unsigned int KernelH  = 3;
    constexpr unsigned int width    = 312;
    constexpr unsigned int height   = 376;
    constexpr unsigned int channels = 3;
    constexpr size_t       rgb_size = width * height * channels;

    std::vector<uint8_t> h_input = common::load_or_generate_rgb_test_data(width, height);

    common::DevBuf d_input(rgb_size);
    common::DevBuf d_output(rgb_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_input.data(), rgb_size, cudaMemcpyHostToDevice));

    std::cout << "\nRunning CPU reference..." << std::endl;
    auto cpu_result = cpu_box_blur<KernelW, KernelH>(h_input, width, height, channels);

    auto nppdx_result = run_nppdx_box_blur<Arch, channels, KernelW, KernelH>(d_input.d_buf, d_output.d_buf, width,
                                                                             height, rgb_size, warm_up_runs, runs);


    auto nppdx_error_stats = common::check_error_detailed(nppdx_result.output.data(), cpu_result.output.data(),
                                                          rgb_size, 1, 1, 1.0, 1, 1, false, false);

    std::cout << "NPPDx vs CPU: max_diff=" << nppdx_error_stats.max_abs_error
              << ", mismatches=" << nppdx_error_stats.error_count << std::endl;

    auto npp_result =
        run_npp_box_blur<KernelW, KernelH>(d_input.d_buf, width, height, channels, rgb_size, warm_up_runs, runs);
    bool npp_passed = true;
    if (!npp_result.output.empty()) {
        auto npp_error_stats = common::check_error_detailed(npp_result.output.data(), cpu_result.output.data(),
                                                            rgb_size, 1, 1, 1.0, 1, 1, false, false);
        std::cout << "NPP vs CPU: max_diff=" << npp_error_stats.max_abs_error
                  << ", mismatches=" << npp_error_stats.error_count << std::endl;
        npp_passed = npp_error_stats.passed;
    }

    common::write_output_file("cow_rgb8_output_blur.rgb", reinterpret_cast<const uint8_t*>(nppdx_result.output.data()),
                              rgb_size);

    common::write_output_file("cow_rgb8_output_blur_npp.rgb",
                              reinterpret_cast<const uint8_t*>(npp_result.output.data()), rgb_size);

    bool passed = (nppdx_error_stats.passed && npp_passed);
    std::cout << "\nResult: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}

template<int Arch>
struct box_filter_functor {
    int operator()() const { return box_filter_example<Arch>(); }
};

int main() {
    return common::run_example_with_sm<box_filter_functor>();
}
