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

#include <iostream>
#include <vector>

#include <nppdx.hpp>
#include <nppdx/shared_memory.hpp>
#include "../common.hpp"
#include "../common/example_sm_runner.hpp"


using namespace nppdx;


// Kernel using shared memory Channels API instead of register arrays
template<typename IngestOp, typename ExgestOp>
__global__ void image_processing_kernel_with_channels(const uint8_t* global_input, uint8_t* global_output, int width,
                                                      int height) {
    extern __shared__ unsigned char smem[];

    using TileStorageType       = nppdx::input_storage_of_t<IngestOp>;
    using ExgestTileStorageType = nppdx::input_storage_of_t<ExgestOp>;
    static_assert(TileStorageType::ChannelSliceType::width == ExgestTileStorageType::ChannelSliceType::width,
                  "Ingest and exgest shared-memory tile widths must match");
    static_assert(TileStorageType::ChannelSliceType::height == ExgestTileStorageType::ChannelSliceType::height,
                  "Ingest and exgest shared-memory tile heights must match");

    auto channels = TileStorageType::from_memory(smem);

    // INGEST: Load image data and convert to intermediate format in shared memory
    IngestOp().execute(global_input, channels, width, height);

    // TRANSFORM HERE: If you add transformations, add __syncthreads() after them
    // Example: channels.channel<0>()(y, x) = transform(channels.channel<0>()(y, x));
    // __syncthreads(); // Uncomment if you add transformations above

    // EXGEST: Convert from intermediate format and store to global memory
    ExgestOp().execute(channels, global_output, width, height);
}

// DX-style example functor template
template<int SM>
struct introduction_example {
    int operator()() {
        std::cout << "Running NPPDx Introduction Example with Shared Memory Channels API" << std::endl;
        std::cout << "SM Architecture: " << SM << std::endl;


        // Image dimensions
        const int width    = 640;
        const int height   = 480;
        const int rgb_size = width * height * 3;

        // Host memory
        std::vector<uint8_t> h_input(rgb_size);
        std::vector<uint8_t> h_output(rgb_size, 0);

        // Initialize input with a simple pattern
        // RGB input is RGB
        for (int i = 0; i < rgb_size; i += 3) {
            h_input[i + 0] = 0 + i % 256;   // R
            h_input[i + 1] = 128 + i % 256; // G
            h_input[i + 2] = 64 + i % 256;  // B
        }

        // Device memory
        common::DevBuf d_input(rgb_size);
        common::DevBuf d_output(rgb_size);

        CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_input.data(), rgb_size, cudaMemcpyHostToDevice));

        // Define NPPDx operations with shared memory support
        using IngestOperation = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                                         nppdx::InputFormat<nppdx::packing_format::rgb24>() +
                                         nppdx::TileSize<48, 48>() + nppdx::Block() + nppdx::SM<SM>());

        using ExgestOperation = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                                         nppdx::OutputFormat<nppdx::packing_format::rgb24>() +
                                         nppdx::TileSize<48, 48>() + nppdx::Block() + nppdx::SM<SM>());


        // Query traits using the new trait system
        constexpr auto ingest_tile_size           = IngestOperation::suggested_tile_size;
        constexpr auto ingest_shared_memory       = IngestOperation::shared_memory_size;
        constexpr auto ingest_elements_per_thread = IngestOperation::elements_per_thread;
        constexpr auto ingest_block_dim           = IngestOperation::block_dim;

        constexpr auto exgest_tile_size           = ExgestOperation::suggested_tile_size;
        constexpr auto exgest_shared_memory       = ExgestOperation::shared_memory_size;
        constexpr auto exgest_elements_per_thread = ExgestOperation::elements_per_thread;
        constexpr auto exgest_block_dim           = ExgestOperation::block_dim;

        // Query format and direction traits
        constexpr auto ingest_direction = nppdx::input_output_direction_of_v<IngestOperation>;
        constexpr auto ingest_format    = nppdx::input_format_of_v<IngestOperation>;
        constexpr auto exgest_direction = nppdx::input_output_direction_of_v<ExgestOperation>;
        constexpr auto exgest_format    = nppdx::output_format_of_v<ExgestOperation>;

        // Demonstrate the convenient packing_format_of trait
        constexpr auto ingest_active_format = nppdx::packing_format_of_v<IngestOperation>;
        constexpr auto exgest_active_format = nppdx::packing_format_of_v<ExgestOperation>;

        using TileStorageType                   = nppdx::input_storage_of_t<IngestOperation>;
        using ExgestTileStorageType             = nppdx::input_storage_of_t<ExgestOperation>;
        constexpr size_t channels_shared_memory = nppdx::shared_memory::compute_total_tile_storage<TileStorageType>();

        std::cout << "\n=== Configuration ===" << std::endl;
        std::cout << "Tile size: " << ingest_tile_size.x << "x" << ingest_tile_size.y << std::endl;
        std::cout << "Block dim: " << ingest_block_dim.x << "x" << ingest_block_dim.y << "x" << ingest_block_dim.z
                  << std::endl;
        std::cout << "Ingest shared memory: " << ingest_shared_memory << " bytes" << std::endl;
        std::cout << "Exgest shared memory: " << exgest_shared_memory << " bytes" << std::endl;
        std::cout << "Channels shared memory (max): " << channels_shared_memory << " bytes" << std::endl;
        std::cout << "Elements per thread: " << ingest_elements_per_thread << std::endl;

        std::cout << "\n=== Format Information ===" << std::endl;
        std::cout << "Ingest direction: "
                  << (ingest_direction == nppdx::input_output_direction::ingest ? "ingest" : "exgest") << std::endl;
        std::cout << "Ingest format: " << (ingest_format == nppdx::packing_format::rgb24 ? "rgb24" : "other")
                  << std::endl;
        std::cout << "Exgest direction: "
                  << (exgest_direction == nppdx::input_output_direction::exgest ? "exgest" : "ingest") << std::endl;
        std::cout << "Exgest format: " << (exgest_format == nppdx::packing_format::rgb24 ? "rgb24" : "other")
                  << std::endl;

        // Ingest and Exgest operations should have compatible parameters
        static_assert(ingest_tile_size.x == exgest_tile_size.x && ingest_tile_size.y == exgest_tile_size.y,
                      "Ingest and Exgest tile sizes must be the same");
        static_assert(TileStorageType::ChannelSliceType::width == ExgestTileStorageType::ChannelSliceType::width &&
                          TileStorageType::ChannelSliceType::height == ExgestTileStorageType::ChannelSliceType::height,
                      "Ingest and Exgest storage tile sizes must be the same");
        static_assert(ingest_elements_per_thread == exgest_elements_per_thread,
                      "Ingest and Exgest elements per thread must be the same");
        static_assert(ingest_block_dim.x == exgest_block_dim.x && ingest_block_dim.y == exgest_block_dim.y &&
                          ingest_block_dim.z == exgest_block_dim.z,
                      "Ingest and Exgest block dimensions must be the same");
        // Note: directions and formats should be different (ingest vs exgest)
        static_assert(ingest_direction != exgest_direction, "Ingest and Exgest directions must be different");
        static_assert(ingest_format == exgest_format, "For this example, input and output formats are the same");
        static_assert(ingest_active_format == exgest_active_format, "For this example, active formats are the same");

        // Calculate grid dimensions
        const auto grid_dim = IngestOperation::calculate_grid_dim(width, height);

        std::cout << "\n=== Launching Kernel ===" << std::endl;
        std::cout << "Grid: " << grid_dim.x << "x" << grid_dim.y << " blocks" << std::endl;
        std::cout << "Using " << channels_shared_memory << " bytes of shared memory per block" << std::endl;

        // Launch kernel with shared memory
        image_processing_kernel_with_channels<IngestOperation, ExgestOperation>
            <<<grid_dim, ingest_block_dim, channels_shared_memory>>>(d_input.d_buf, d_output.d_buf, width, height);

        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        // Copy result back
        CUDA_CHECK_AND_EXIT(cudaMemcpy(h_output.data(), d_output.d_buf, rgb_size, cudaMemcpyDeviceToHost));

        // Verify result - should be identical to input since we're just doing ingest->exgest
        std::cout << "\n=== Verification ===" << std::endl;
        std::cout << "Image size: " << width << "x" << height << std::endl;

        bool success    = true;
        int  mismatches = 0;
        for (int i = 0; i < rgb_size && success; i += 3) {
            if (h_output[i + 0] != h_input[i + 0] || h_output[i + 1] != h_input[i + 1] ||
                h_output[i + 2] != h_input[i + 2]) {
                mismatches++;
                if (mismatches <= 5) { // Show first 5 mismatches
                    std::cout << "Mismatch at pixel " << (i / 3) << ": " << "input=(" << (int)h_input[i] << ","
                              << (int)h_input[i + 1] << "," << (int)h_input[i + 2] << ") " << "output=("
                              << (int)h_output[i] << "," << (int)h_output[i + 1] << "," << (int)h_output[i + 2] << ")"
                              << std::endl;
                }
                if (mismatches > 100) {
                    success = false;
                    break;
                }
            }
        }

        if (mismatches > 0) {
            std::cout << "Total mismatches: " << mismatches << std::endl;
        }

        std::cout << "Example " << (success ? "succeeded" : "failed") << std::endl;

        return success ? 0 : 1;
    }
};

int main() {
    std::cout << "===========================================================" << std::endl;
    std::cout << "NPPDx Introduction Example - Shared Memory Channels API" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << std::endl;

    return common::run_example_with_sm<introduction_example>();
}
