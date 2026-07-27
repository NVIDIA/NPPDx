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

// Resize Example - NPPDx
// Demonstrates image resizing with various interpolation methods using the cow test image

#include <nppdx.hpp>

#include <iostream>
#include <vector>

#include "../common.hpp"
#include "../common/cpu_resize_reference.hpp"

// =============================================================================
// NPPDx Resize Kernel: Ingest -> Resize -> Exgest
// =============================================================================

template<typename Ingest, typename Resize, typename Exgest, typename InputTileStorageType,
         typename OutputTileStorageType, typename IntermediateTileStorageType>
__global__ void resize_kernel(const uint8_t* input, uint8_t* output, const unsigned int input_width,
                              const unsigned int input_height, const unsigned int output_width,
                              const unsigned int output_height) {
    extern __shared__ unsigned char smem[];


    auto tile_storage      = nppdx::shared_memory::slice_into_tile_storage<InputTileStorageType, OutputTileStorageType,
                                                                           IntermediateTileStorageType>(smem);
    auto input_tile        = std::get<0>(tile_storage);
    auto output_tile       = std::get<1>(tile_storage);
    auto intermediate_tile = std::get<2>(tile_storage);

    // Step 1: Ingest input with halo
    Ingest().execute(input, input_tile, input_width, input_height);

    // Step 2: Resize with intermediate tile
    Resize().execute(input_tile, intermediate_tile, output_tile, input_width, input_height);

    // Step 3: Exgest resized output
    Exgest().execute(output_tile, output, output_width, output_height);
}

// =============================================================================
// NPPDx Resize Execution - 2x Upscale with Cow Image
// =============================================================================
//
// Validation uses the shared CPU golden in ../common/cpu_resize_reference.hpp
// (nppdx::example::cpu_reference::resize<J, K, Method, Channels, ...>).

template<int Arch, unsigned int NumChannels, unsigned int J, unsigned int K, nppdx::interpolation_method Method>
common::nppdx_results<uint8_t> run_nppdx_resize(uint8_t* d_input, uint8_t* d_output, const unsigned int input_width,
                                                const unsigned int input_height, const unsigned int output_width,
                                                const unsigned int output_height, const size_t output_size) {
    // Use tile size that divides 312x376 (cow image dimensions)
    constexpr unsigned int tile_x = 16;
    constexpr unsigned int tile_y = 16;

    constexpr bool downscale = (J > K);

    // Query required resize halo from the operation itself (depends on J/K and method).
    using ResizePre            = decltype(nppdx::Function<nppdx::function::resize>() + nppdx::Resize<J, K, Method>() +
                               nppdx::TileSize<tile_x, tile_y>() + nppdx::SM<Arch>() + nppdx::Block());
    constexpr auto resize_halo = ResizePre::local_halo;

    // Ingest RGB24 with halo for resize
    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, resize_halo)() + nppdx::SM<Arch>() + nppdx::Block());

    // Resize operation
    using Resize                = decltype(ResizePre() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, resize_halo)() +
                            NPPDX_MAKE_HALO(nppdx::CumulativeHalo, resize_halo)());
    using OutputTileStorageType = nppdx::output_storage_of_t<Resize>;

    [[maybe_unused]] constexpr auto empty_halo = nppdx::Halo4::make_empty();

    // Exgest to RGB24 (output tile is larger)
    using Exgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                            nppdx::OutputFormat<nppdx::packing_format::rgb24>() +
                            nppdx::TileSize<OutputTileStorageType::ChannelSliceType::width,
                                            OutputTileStorageType::ChannelSliceType::height>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, empty_halo)() +
                            NPPDX_MAKE_HALO(nppdx::CumulativeHalo, empty_halo)() + nppdx::SM<Arch>() + nppdx::Block());

    using InputTileStorageType        = nppdx::input_storage_of_t<Ingest>;
    using IntermediateTileStorageType = nppdx::temp_storage_of_t<Resize>;

    std::cout << "Input tile: " << tile_x << "x" << tile_y << " with halo=" << resize_halo.left_top.x << ","
              << resize_halo.right_bottom.x << "," << resize_halo.left_top.y << "," << resize_halo.right_bottom.y
              << std::endl;
    std::cout << "Input storage tile: " << Ingest::input_storage.size.x << "x" << Ingest::input_storage.size.y
              << std::endl;
    std::cout << "Output storage tile: " << Resize::output_storage.size.x << "x" << Resize::output_storage.size.y
              << std::endl;
    std::cout << "Intermediate storage tile: " << Resize::temp_storage.size.x << "x" << Resize::temp_storage.size.y
              << std::endl;
    std::cout << "Scale: " << J << "->" << K << " (" << (downscale ? "downscale" : "upscale") << ")" << std::endl;

    constexpr size_t smem_size =
        nppdx::shared_memory::compute_total_tile_storage<InputTileStorageType, OutputTileStorageType,
                                                         IntermediateTileStorageType>();
    std::cout << "Shared memory: " << smem_size << " bytes per block" << std::endl;

    //Need to get the maximum block dimension for the exgest operation
    constexpr dim3 block_dim = downscale ? Ingest::block_dim : Exgest::block_dim;
    const dim3     grid_dim  = downscale ? Ingest::calculate_grid_dim(input_width, input_height)
                                         : Exgest::calculate_grid_dim(output_width, output_height);

    std::cout << "Grid: " << grid_dim.x << "x" << grid_dim.y << " | Block: " << block_dim.x << std::endl;

    auto kernel_ptr =
        resize_kernel<Ingest, Resize, Exgest, InputTileStorageType, OutputTileStorageType, IntermediateTileStorageType>;
    // Required only when requesting dynamic shared memory above the default 48 KB limit.
    if constexpr (smem_size > 48 * 1024) {
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    std::cout << "\nRunning NPPDx resize ..." << std::endl;

    kernel_ptr<<<grid_dim, block_dim, smem_size>>>(d_input, d_output, input_width, input_height, output_width,
                                                   output_height);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy output to host
    std::vector<uint8_t> h_output(output_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost));

    return common::nppdx_results<uint8_t> {h_output, 0.0f};
}

// =============================================================================
// Main Example Function
// =============================================================================

template<int Arch>
int resize_example() {
    std::cout << "=== NPPDx Resize Example (SM" << Arch << ") ===" << std::endl;

    // Cow image dimensions
    constexpr unsigned int input_width  = 312;
    constexpr unsigned int input_height = 376;
    constexpr unsigned int channels     = 3;

    constexpr unsigned int j      = 2; // Number of input samples
    constexpr unsigned int k      = 1; // Number of output samples
    constexpr auto         method = nppdx::interpolation_method::bilinear;

    constexpr unsigned int output_width  = input_width * k / j;
    constexpr unsigned int output_height = input_height * k / j;

    constexpr size_t input_size  = input_width * input_height * channels;
    constexpr size_t output_size = output_width * output_height * channels;

    std::cout << "Input: " << input_width << "x" << input_height << " RGB24 (" << input_size << " bytes)" << std::endl;
    std::cout << "Output: " << output_width << "x" << output_height << " RGB24 (" << output_size << " bytes)"
              << std::endl;

    // Load cow test image
    std::vector<uint8_t> h_input = common::load_or_generate_rgb_test_data(input_width, input_height);

    // Allocate device memory
    common::DevBuf d_input(input_size);
    common::DevBuf d_output(output_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_input.data(), input_size, cudaMemcpyHostToDevice));

    // Run NPPDx resize
    auto nppdx_result = run_nppdx_resize<Arch, channels, j, k, method>(
        d_input.d_buf, d_output.d_buf, input_width, input_height, output_width, output_height, output_size);

    // Generate CPU reference for comparison (shared golden; honors the `method` constexpr above).
    std::cout << "\nGenerating CPU reference (separable " << static_cast<int>(method) << ")..." << std::endl;
    std::vector<uint8_t> cpu_reference =
        nppdx::example::cpu_reference::resize<j, k, method, channels, /*UseRadialLanczos2D=*/false, uint8_t>(
            h_input, static_cast<int>(input_width), static_cast<int>(input_height));

    // Compare NPPDx result against CPU reference
    std::cout << "\n--- Validation against CPU reference ---" << std::endl;
    auto error_stats = common::check_error_detailed(nppdx_result.output.data(), cpu_reference.data(), output_size,
                                                    3,    // tolerance per pixel
                                                    20,   // max errors to print
                                                    2.0,  // max avg error threshold
                                                    1, 1, // block size (ignored)
                                                    true, // print summary
                                                    false // don't print all errors
    );

    std::cout << "Max pixel difference: " << error_stats.max_abs_error << std::endl;
    std::cout << "Avg pixel difference: " << error_stats.avg_abs_error << std::endl;
    std::cout << "Pixels exceeding tolerance: " << error_stats.error_count << "/" << output_size << std::endl;

    // Write output files for visual inspection
    common::write_output_file("cow_resized_2x_nppdx.rgb", nppdx_result.output.data(), output_size);
    common::write_output_file("cow_resized_2x_cpu_ref.rgb", cpu_reference.data(), output_size);
    std::cout << "\nOutput dimensions: " << output_width << "x" << output_height << " (use raw RGB24 viewer)"
              << std::endl;

    bool passed = error_stats.passed;
    std::cout << "\nResult: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}

template<int Arch>
struct resize_example_functor {
    int operator()() const { return resize_example<Arch>(); }
};

int main() {
    return common::run_example_with_sm<resize_example_functor>();
}
