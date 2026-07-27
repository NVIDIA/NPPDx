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

#ifndef NPPDX_EXAMPLE_COMMON_TEST_DATA_HPP
#define NPPDX_EXAMPLE_COMMON_TEST_DATA_HPP

#include "test_data_host.hpp"

#include <cuda_runtime_api.h>
#include <cstdint>
#include <vector>
#include "macros.hpp"
#include "nppdx/operators/formats.hpp"
#include "example_sm_runner.hpp"

namespace common {

    template<typename T>
    struct ExpectedYUV {
        const char* name;
        T           y, u, v;
    };

    // Helper to scale YUV values from 8-bit to target bit depth
    template<typename T>
    inline constexpr T scale_yuv_value(uint8_t value_8bit, int target_bit_depth) {
        if (target_bit_depth == 8) {
            return static_cast<T>(value_8bit);
        }

        int max_8bit   = 255;
        int max_target = (1 << target_bit_depth) - 1;
        return static_cast<T>((value_8bit * max_target + max_8bit / 2) / max_8bit);
    }

    template<typename T>
    inline std::vector<ExpectedYUV<T>> get_expected_yuv_bt601(int bit_depth = 8) {
        static const struct {
            const char* name;
            uint8_t     y, u, v;
        } base_yuv[16] = {
            {"Black", 0, 128, 128},        // from RGB8(0,0,0)
            {"White", 255, 128, 128},      // from RGB8(255,255,255)
            {"Red", 76, 90, 255},          // from RGB8(255,0,0)
            {"Green", 150, 54, 0},         // from RGB8(0,255,0)
            {"Blue", 29, 239, 102},        // from RGB8(0,0,255)
            {"Yellow", 226, 17, 154},      // from RGB8(255,255,0)
            {"Magenta", 105, 202, 255},    // from RGB8(255,0,255)
            {"Cyan", 179, 166, 0},         // from RGB8(0,255,255)
            {"Gray", 128, 128, 128},       // from RGB8(128,128,128)
            {"Light Gray", 192, 128, 128}, // from RGB8(192,192,192)
            {"Dark Gray", 64, 128, 128},   // from RGB8(64,64,64)
            {"Dark Red", 38, 109, 207},    // from RGB8(128,0,0)
            {"Dark Green", 75, 91, 62},    // from RGB8(0,128,0)
            {"Dark Blue", 15, 184, 115},   // from RGB8(0,0,128)
            {"Olive", 113, 72, 141},       // from RGB8(128,128,0)
            {"Purple", 53, 165, 194}       // from RGB8(128,0,128)
        };

        std::vector<ExpectedYUV<T>> yuv_values;
        yuv_values.reserve(16);

        for (int i = 0; i < 16; ++i) {
            yuv_values.push_back({
                base_yuv[i].name, scale_yuv_value<T>(base_yuv[i].y, bit_depth), // Y (luma)
                scale_yuv_value<T>(base_yuv[i].u, bit_depth),                   // U (chroma)
                scale_yuv_value<T>(base_yuv[i].v, bit_depth)                    // V (chroma)
            });
        }

        return yuv_values;
    }

    namespace detail {

        // Shared kernel used by test-image generation helpers.
        template<typename Ingest, typename ColorConvert, typename Exgest>
        __global__ void color_convert_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
            float internal_data[Ingest::elements_per_thread];

            Ingest().execute(input, internal_data, width, height);
            ColorConvert().execute(internal_data, width, height);
            Exgest().execute(internal_data, output, width, height);
        }

        template<nppdx::packing_format OutFormat, int Arch>
        std::vector<uint8_t> generate_test_image_impl(nppdx::CVal<OutFormat>, nppdx::CVal<Arch>, int width, int height,
                                                      int block_size = 64, int bit_depth = 8) {

            constexpr nppdx::packing_format_props out_format_traits = nppdx::get_packing_format_props(OutFormat);

            using BaseOp = decltype(nppdx::TileSize<48, 48> {} + nppdx::Block {} + nppdx::SM<Arch> {});

            using IngestOp = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest> {} +
                                      nppdx::InputFormat<nppdx::packing_format::rgb24> {} + BaseOp {});

            using CCOp           = nppdx::ColorConvert<nppdx::color_space::rgb, out_format_traits.default_color_space,
                                                       nppdx::bit_depth::bpp_8u,
                                                       nppdx::bit_depth_from_int(out_format_traits.bits_per_channel)>;
            using ColorConvertOp = decltype(nppdx::Function<nppdx::function::color_convert> {} + CCOp {} + BaseOp {});

            using ExgestOp = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest> {} +
                                      nppdx::OutputFormat<OutFormat> {} + BaseOp {});


            std::vector<uint8_t> input_h = generate_rgb_test_image(width, height, block_size, bit_depth);
            std::vector<uint8_t> output_h(out_format_traits.minimum_buffer_byte_size(size_t(width), size_t(height)));
            uint8_t*             input_d;
            uint8_t*             output_d;
            CUDA_CHECK_AND_EXIT(cudaMalloc(&input_d, input_h.size()));
            CUDA_CHECK_AND_EXIT(cudaMalloc(&output_d, output_h.size()));

            CUDA_CHECK_AND_EXIT(cudaMemcpy(input_d, input_h.data(), input_h.size(), cudaMemcpyHostToDevice));

            color_convert_kernel<IngestOp, ColorConvertOp, ExgestOp>
                <<<IngestOp::calculate_grid_dim(width, height), IngestOp::block_dim>>>(input_d, output_d, width,
                                                                                       height);

            CUDA_CHECK_AND_EXIT(cudaMemcpy(output_h.data(), output_d, output_h.size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK_AND_EXIT(cudaFree(input_d));
            CUDA_CHECK_AND_EXIT(cudaFree(output_d));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            return output_h;
        }


    } // namespace detail

    // Generate a test image in an arbitrary packing format.
    // This first generates an RGB image and then uses NPPDx itself to convert
    // it to a different output format. It is defined for convenience and is not
    // optimized for performance. Notably, the output image gets copied back to
    // the host, which may not be desired.
    // Since it uses NPPDx, its results should not be used as a reference for
    // NPPDx tests, since it will reproduce any errors in the implementation.
    // This is less of an issue, though not entirely eliminated, if using this
    // function to generate an arbitrary image in the given format, e.g., to use
    // as a testing input.
    template<typename T, typename U>
    inline std::vector<uint8_t> generate_test_image(const T out_format, const U arch, int width, int height,
                                                    int block_size = 64, int bit_depth = 8) {

        static_assert(nppdx::has_carried_type<nppdx::packing_format, T>,
                      "format must have runtime type nppdx::packing_format.");
        static_assert(nppdx::has_carried_type<int, U>, "format must have runtime type int.");

        return nppdx::dispatch_format(out_format, [&](auto cFormat) {
            return common::dispatch_sm_arch(
                [&](auto cArch) {
                    return detail::generate_test_image_impl(cFormat, cArch, width, height, block_size, bit_depth);
                },
                arch);
        });
    }

    template<typename T>
    inline bool verify_yuv_test_colors(const T* y_plane, const T* u_plane, const T* v_plane, int width,
                                       int block_size = 64, int tolerance = 5, bool print_errors = true,
                                       int bit_depth = 8) {
        auto      expected       = get_expected_yuv_bt601<T>(bit_depth);
        const int blocks_per_row = width / block_size;
        bool      all_passed     = true;

        for (int i = 0; i < 16; ++i) {
            // Calculate pixel index in the center of each color block
            int block_y = (i / blocks_per_row) * block_size + block_size / 2;
            int block_x = (i % blocks_per_row) * block_size + block_size / 2;
            int pix_idx = block_y * width + block_x;
            int uv_idx  = (block_y / 2) * (width / 2) + (block_x / 2);

            // Check Y, U, V values with tolerance
            bool y_ok = (std::abs(static_cast<int>(y_plane[pix_idx]) - static_cast<int>(expected[i].y)) <= tolerance);
            bool u_ok = (std::abs(static_cast<int>(u_plane[uv_idx]) - static_cast<int>(expected[i].u)) <= tolerance);
            bool v_ok = (std::abs(static_cast<int>(v_plane[uv_idx]) - static_cast<int>(expected[i].v)) <= tolerance);

            if (!y_ok || !u_ok || !v_ok) {
                if (print_errors) {
                    std::cout << "  " << expected[i].name << " YUV check FAILED: " << "Expected ("
                              << static_cast<int>(expected[i].y) << "," << static_cast<int>(expected[i].u) << ","
                              << static_cast<int>(expected[i].v) << "), " << "Got ("
                              << static_cast<int>(y_plane[pix_idx]) << "," << static_cast<int>(u_plane[uv_idx]) << ","
                              << static_cast<int>(v_plane[uv_idx]) << ")" << std::endl;
                }
                all_passed = false;
            }
        }

        return all_passed;
    }

} // namespace common

#endif // NPPDX_EXAMPLE_COMMON_TEST_DATA_HPP
