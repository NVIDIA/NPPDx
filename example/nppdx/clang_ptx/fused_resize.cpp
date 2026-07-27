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

#include "device_management.hpp"
#include "test_data_host.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

    struct FusedResizeConfig {
        unsigned int shared_memory = 0;
        unsigned int block_x       = 1;
        unsigned int block_y       = 1;
        unsigned int block_z       = 1;
        unsigned int out_tile_x    = 1;
        unsigned int out_tile_y    = 1;
    };

    FusedResizeConfig query_fused_resize_config(const common::cuda_driver::Driver&      driver,
                                                const common::cuda_driver::ModuleGuard& module) {
        using namespace common::cuda_driver;

        std::array<unsigned int, 6> host_config {};
        DeviceBuffer                device_config(driver, sizeof(host_config));
        DevicePtr                   config_pointer = device_config.get();
        void*                       query_args[]   = {&config_pointer};

        KernelLauncher query_kernel(driver, module.function("nppdx_clang_fused_resize_query"));
        query_kernel.launch(LaunchConfig {}, query_args);
        driver.synchronize();
        device_config.copy_to_host(host_config.data(), sizeof(host_config));

        return FusedResizeConfig {host_config[0], host_config[1], host_config[2],
                                  host_config[3], host_config[4], host_config[5]};
    }

    std::uint64_t checksum(const std::vector<std::uint8_t>& values) {
        std::uint64_t result = 1469598103934665603ull;
        for (std::uint8_t value : values) {
            result ^= value;
            result *= 1099511628211ull;
        }
        return result;
    }

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <ptx-file>\n";
        return 2;
    }

    using namespace common::cuda_driver;

    constexpr unsigned int kernel_w      = 5;
    constexpr unsigned int kernel_h      = 5;
    constexpr unsigned int input_width   = 312;
    constexpr unsigned int input_height  = 376;
    constexpr unsigned int channels      = 3;
    constexpr unsigned int scale_j       = 1;
    constexpr unsigned int scale_k       = 2;
    constexpr unsigned int output_width  = input_width * 2;
    constexpr unsigned int output_height = input_height * 2;
    constexpr std::size_t  rgb_size      = static_cast<std::size_t>(input_width) * input_height * channels;
    constexpr std::size_t  y_size        = static_cast<std::size_t>(output_width) * output_height;
    constexpr std::size_t  uv_size       = static_cast<std::size_t>(output_width / 2) * output_height;
    constexpr std::size_t  yuv_size      = y_size + 2 * uv_size;

    std::cout << "=== NPPDx Fused Blur + Resize + Convert Example (Clang PTX) ===\n";
    std::cout << "Input: " << input_width << "x" << input_height << " RGB24 (" << rgb_size << " bytes)\n";
    std::cout << "Output: " << output_width << "x" << output_height << " YUV422P (" << yuv_size << " bytes)\n";
    std::cout << "Box filter: " << kernel_w << "x" << kernel_h << '\n';
    std::cout << "Resize: " << scale_j << ":" << scale_k << " nearest interpolation method\n";

    std::vector<std::uint8_t> host_input = common::load_or_generate_rgb_test_data(input_width, input_height);
    std::vector<std::uint8_t> host_output(yuv_size, 0);

    try {
        Driver driver = Driver::load_default();
        driver.init();

        Device            device = driver.device();
        ContextGuard      context(driver, device);
        const std::string module_path = resolve_module_path(argv[1], argv[0]);
        ModuleGuard       module(driver, module_path.c_str());

        FusedResizeConfig config = query_fused_resize_config(driver, module);
        if (config.shared_memory == 0 || config.out_tile_x == 0 || config.out_tile_y == 0) {
            std::cerr << "invalid fused resize launch config\n";
            return 5;
        }

        DeviceBuffer device_input(driver, rgb_size);
        DeviceBuffer device_output(driver, yuv_size);
        device_input.copy_from_host(host_input.data(), rgb_size);
        device_output.copy_from_host(host_output.data(), yuv_size);

        KernelLauncher fused_kernel(driver, module.function("nppdx_clang_fused_resize"));
        fused_kernel.set_max_dynamic_shared_memory(config.shared_memory);

        const unsigned int grid_x = (output_width + config.out_tile_x - 1) / config.out_tile_x;
        const unsigned int grid_y = (output_height + config.out_tile_y - 1) / config.out_tile_y;

        DevicePtr    input_pointer  = device_input.get();
        DevicePtr    output_pointer = device_output.get();
        unsigned int input_w        = input_width;
        unsigned int input_h        = input_height;
        unsigned int output_w       = output_width;
        unsigned int output_h       = output_height;
        void*        kernel_args[]  = {&input_pointer, &output_pointer, &input_w, &input_h, &output_w, &output_h};

        LaunchConfig launch_config {
            grid_x, grid_y, 1, config.block_x, config.block_y, config.block_z, config.shared_memory, nullptr};
        std::cout << "\nRunning NPPDx fused (blur + resize + convert)...\n";
        fused_kernel.launch(launch_config, kernel_args);
        driver.synchronize();
        device_output.copy_to_host(host_output.data(), yuv_size);

        bool has_data = false;
        for (std::uint8_t value : host_output) {
            if (value != 0) {
                has_data = true;
                break;
            }
        }
        if (!has_data) {
            std::cerr << "kernel produced all-zero output\n";
            return 6;
        }

        common::write_output_file("cow_blur_resize_convert_nppdx.yuv422", host_output.data(), yuv_size);
        std::cout << "PASS " << module_path << " grid=" << grid_x << "x" << grid_y << " block=" << config.block_x << "x"
                  << config.block_y << "x" << config.block_z << " smem=" << config.shared_memory
                  << " checksum=" << checksum(host_output) << '\n';
        std::cout << "\nResult: PASSED\n";
    } catch (const Error& error) {
        std::cerr << error.what() << '\n';
        return error.code();
    }

    return 0;
}
