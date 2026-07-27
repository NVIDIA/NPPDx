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
#include <vector>

#include "../common.hpp"

template<typename Ingest, typename ColorConvert, typename Exgest>
__global__ void color_convert_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
    float internal_data[Ingest::elements_per_thread];

    Ingest().execute(input, internal_data, width, height);
    ColorConvert().execute(internal_data, width, height);
    Exgest().execute(internal_data, output, width, height);
}

template<int Arch>
int image_conversion_example() {
    std::cout << "=== NPPDx Color Conversion Example (SM" << Arch << ") ===" << std::endl;

    constexpr int width    = 256;
    constexpr int height   = 256;
    constexpr int rgb_size = width * height * 3;
    constexpr int y_size   = width * height;
    constexpr int uv_size  = (width / 2) * (height / 2);
    constexpr int yuv_size = y_size + 2 * uv_size;

    std::cout << "Image: " << width << "x" << height << " | RGB24: " << rgb_size << " bytes | YUV420P: " << yuv_size
              << " bytes\n"
              << std::endl;

    // Define NPPDx operations
    using RGBIngest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                               nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<48, 48>() +
                               nppdx::SM<Arch>() + nppdx::Block());

    using YUVIngest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                               nppdx::InputFormat<nppdx::packing_format::yuv420p>() + nppdx::TileSize<48, 48>() +
                               nppdx::SM<Arch>() + nppdx::Block());

    using RGBToYUVConvert = decltype(nppdx::Function<nppdx::function::color_convert>() +
                                     nppdx::ColorConvert<nppdx::color_space::rgb, nppdx::color_space::yuv_bt601,
                                                         nppdx::bit_depth::bpp_8u, nppdx::bit_depth::bpp_8u>() +
                                     nppdx::TileSize<48, 48>() + nppdx::SM<Arch>() + nppdx::Block());

    using YUVToRGBConvert = decltype(nppdx::Function<nppdx::function::color_convert>() +
                                     nppdx::ColorConvert<nppdx::color_space::yuv_bt601, nppdx::color_space::rgb,
                                                         nppdx::bit_depth::bpp_8u, nppdx::bit_depth::bpp_8u>() +
                                     nppdx::TileSize<48, 48>() + nppdx::SM<Arch>() + nppdx::Block());

    using YUVExgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                               nppdx::OutputFormat<nppdx::packing_format::yuv420p>() + nppdx::TileSize<48, 48>() +
                               nppdx::SM<Arch>() + nppdx::Block());

    using RGBExgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                               nppdx::OutputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<48, 48>() +
                               nppdx::SM<Arch>() + nppdx::Block());

    static_assert(nppdx::is_supported_v<RGBIngest, Arch>);
    static_assert(nppdx::is_supported_v<YUVIngest, Arch>);
    static_assert(nppdx::is_supported_v<RGBToYUVConvert, Arch>);
    static_assert(nppdx::is_supported_v<YUVToRGBConvert, Arch>);
    static_assert(nppdx::is_supported_v<YUVExgest, Arch>);
    static_assert(nppdx::is_supported_v<RGBExgest, Arch>);

    std::cout << "Tile: " << RGBIngest::tile_size_x << "x" << RGBIngest::tile_size_y << "\n" << std::endl;

    std::vector<uint8_t> h_rgb_input = common::generate_rgb_test_image(width, height);
    std::vector<uint8_t> h_rgb_output(rgb_size);
    std::vector<uint8_t> h_yuv_intermediate(yuv_size);

    common::DevBuf d_rgb_input(rgb_size);
    common::DevBuf d_rgb_output(rgb_size);
    common::DevBuf d_yuv_intermediate(yuv_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_rgb_input.d_buf, h_rgb_input.data(), rgb_size, cudaMemcpyHostToDevice));

    constexpr dim3 block_dim = RGBIngest::block_dim;
    const dim3     grid_dim  = RGBIngest::calculate_grid_dim(width, height);

    std::cout << "Grid: " << grid_dim.x << "x" << grid_dim.y << " | Block: " << block_dim.x << "x" << block_dim.y
              << "\n"
              << std::endl;

    std::cout << "RGB -> YUV conversion..." << std::endl;
    color_convert_kernel<RGBIngest, RGBToYUVConvert, YUVExgest>
        <<<grid_dim, block_dim>>>(d_rgb_input.d_buf, d_yuv_intermediate.d_buf, width, height);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::cout << "YUV -> RGB conversion..." << std::endl;
    color_convert_kernel<YUVIngest, YUVToRGBConvert, RGBExgest>
        <<<grid_dim, block_dim>>>(d_yuv_intermediate.d_buf, d_rgb_output.d_buf, width, height);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_rgb_output.data(), d_rgb_output.d_buf, rgb_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(h_yuv_intermediate.data(), d_yuv_intermediate.d_buf, yuv_size, cudaMemcpyDeviceToHost));

    const uint8_t* y_plane = h_yuv_intermediate.data();
    const uint8_t* u_plane = y_plane + y_size;
    const uint8_t* v_plane = u_plane + uv_size;

    std::cout << "\nSample conversions:" << std::endl;
    const int   test_pixels[] = {0, 65 * 3, 130 * 3, 195 * 3};
    const char* color_names[] = {"Black", "White", "Red", "Green"};

    for (int t = 0; t < 4; ++t) {
        int pix_idx = test_pixels[t];
        int pix_num = pix_idx / 3;
        int y_idx   = pix_num;
        int uv_idx  = (pix_num / 2) + ((pix_num / width) / 2) * (width / 2);

        std::cout << "  " << color_names[t] << " RGB(" << (int)h_rgb_input[pix_idx] << ","
                  << (int)h_rgb_input[pix_idx + 1] << "," << (int)h_rgb_input[pix_idx + 2] << ") -> YUV("
                  << (int)y_plane[y_idx] << "," << (int)u_plane[uv_idx] << "," << (int)v_plane[uv_idx] << ") -> RGB("
                  << (int)h_rgb_output[pix_idx] << "," << (int)h_rgb_output[pix_idx + 1] << ","
                  << (int)h_rgb_output[pix_idx + 2] << ")" << std::endl;
    }

    auto stats = common::compute_conversion_stats(h_rgb_input.data(), h_rgb_output.data(), rgb_size, 2, 40, 3.0, true,
                                                  "Round-trip");

    std::cout << "\nExample complete: " << (stats.passed ? "SUCCESS" : "FAILED") << std::endl;
    return stats.passed ? 0 : 1;
}

template<int Arch>
struct image_conversion_example_functor {
    int operator()() const { return image_conversion_example<Arch>(); }
};

int main() {
    return common::run_example_with_sm<image_conversion_example_functor>();
}
