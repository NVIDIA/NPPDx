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
#include "../common.hpp"
#include "../common/example_sm_runner.hpp"


using namespace nppdx;

template<typename IngestOp, typename ExgestOp>
__global__ void image_processing_kernel(const uint8_t* global_input, uint8_t* global_output, int width, int height) {

    using processing_t = nppdx::processing_type_of_t<IngestOp>;
    // Query register requirements at compile time - for 3-channel intermediate format
    constexpr size_t elements_per_thread = nppdx::elements_per_thread_of_v<IngestOp>;


    // Register arrays for the intermediate processing (RGB float)
    processing_t intermediate_data[elements_per_thread];

    // INGEST: Load image data and convert to intermediate format
    IngestOp().execute(global_input, intermediate_data, width, height);

    // TRANSFORM HERE: Additional transformations can be done on intermediate_data

    // EXGEST: Convert from intermediate format and store to global memory
    ExgestOp().execute(intermediate_data, global_output, width, height);
}

// DX-style example functor template
template<int SM>
struct introduction_example {
    int operator()() {
        std::cout << "Running NPPDx Introduction Example with SM<" << SM << ">" << std::endl;


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

        // Define NPPDx operations
        using IngestOperation = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                                         nppdx::InputFormat<nppdx::packing_format::rgb24>() +
                                         nppdx::TileSize<48, 48>() + nppdx::Block() + nppdx::SM<SM>());

        using ExgestOperation = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                                         nppdx::OutputFormat<nppdx::packing_format::rgb24>() +
                                         nppdx::TileSize<48, 48>() + nppdx::Block() + nppdx::SM<SM>());


        // Query traits using the new trait system
        constexpr auto ingest_tile_size           = IngestOperation::suggested_tile_size;
        constexpr auto ingest_elements_per_thread = IngestOperation::elements_per_thread;
        constexpr auto ingest_block_dim           = IngestOperation::block_dim;

        constexpr auto exgest_tile_size           = ExgestOperation::suggested_tile_size;
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

        std::cout << "Ingest elements per thread: " << ingest_elements_per_thread << std::endl;
        std::cout << "Tile size: " << ingest_tile_size.x << "x" << ingest_tile_size.y << std::endl;
        std::cout << "Ingest direction: "
                  << (ingest_direction == nppdx::input_output_direction::ingest ? "ingest" : "exgest") << std::endl;
        std::cout << "Ingest format: " << (ingest_format == nppdx::packing_format::rgb24 ? "rgb24" : "other")
                  << std::endl;
        std::cout << "Exgest direction: "
                  << (exgest_direction == nppdx::input_output_direction::exgest ? "exgest" : "ingest") << std::endl;
        std::cout << "Exgest format: " << (exgest_format == nppdx::packing_format::rgb24 ? "rgb24" : "other")
                  << std::endl;

        std::cout << "Ingest block dim: " << ingest_block_dim.x << "x" << ingest_block_dim.y << "x"
                  << ingest_block_dim.z << std::endl;
        std::cout << "Exgest block dim: " << exgest_block_dim.x << "x" << exgest_block_dim.y << "x"
                  << exgest_block_dim.z << std::endl;

        // Ingest and Exgest operations should have compatible parameters
        static_assert(ingest_tile_size.x == exgest_tile_size.x && ingest_tile_size.y == exgest_tile_size.y,
                      "Ingest and Exgest tile sizes must be the same");
        static_assert(ingest_elements_per_thread == exgest_elements_per_thread,
                      "Ingest and Exgest elements per thread must be the same");
        static_assert(ingest_block_dim.x == exgest_block_dim.x && ingest_block_dim.y == exgest_block_dim.y &&
                          ingest_block_dim.z == exgest_block_dim.z,
                      "Ingest and Exgest block dimensions must be the same");
        // Note: directions and formats should be different (ingest vs exgest)
        static_assert(ingest_direction != exgest_direction, "Ingest and Exgest directions must be different");
        static_assert(ingest_format == exgest_format, "For this example, input and output formats are the same");
        static_assert(ingest_active_format == exgest_active_format, "For this example, active formats are the same");

        const dim3 grid_dim = IngestOperation::calculate_grid_dim(width, height);

        // Launch kernel
        image_processing_kernel<IngestOperation, ExgestOperation>
            <<<grid_dim, ingest_block_dim>>>(d_input.d_buf, d_output.d_buf, width, height);

        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        // Copy result back
        CUDA_CHECK_AND_EXIT(cudaMemcpy(h_output.data(), d_output.d_buf, rgb_size, cudaMemcpyDeviceToHost));

        // Verify result, check it is the same as the input -1
        bool success = true;
        for (int i = 0; i < rgb_size && success; i += 3) {
            if (h_output[i + 0] != h_input[i + 0] || h_output[i + 1] != h_input[i + 1] ||
                h_output[i + 2] != h_input[i + 2]) {
                success = false;
            }
        }

        std::cout << "Image size: " << width << "x" << height << std::endl;
        std::cout << "Test " << (success ? "PASSED" : "FAILED") << std::endl;

        if (!success) {
            for (int i = 0; i < rgb_size; i += 3) {
                std::cout << "h_output[" << i + 0 << "] = " << static_cast<int>(h_output[i + 0]) << ", h_input["
                          << i + 0 << "] = " << static_cast<int>(h_input[i + 0]) << std::endl;
                std::cout << "h_output[" << i + 1 << "] = " << static_cast<int>(h_output[i + 1]) << ", h_input["
                          << i + 1 << "] = " << static_cast<int>(h_input[i + 1]) << std::endl;
                std::cout << "h_output[" << i + 2 << "] = " << static_cast<int>(h_output[i + 2]) << ", h_input["
                          << i + 2 << "] = " << static_cast<int>(h_input[i + 2]) << std::endl;
            }
        }
        return success ? 0 : 1;
    }
};

int main() {
    std::cout << "NPPDx Introduction Example" << std::endl;
    std::cout << "This example demonstrates basic ingest/exgest operations using operator composition" << std::endl;

    // Use the DX-style SM runner to dispatch based on device architecture
    return common::run_example_with_sm<introduction_example>();
}
