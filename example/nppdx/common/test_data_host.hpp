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

#ifndef NPPDX_EXAMPLE_COMMON_TEST_DATA_HOST_HPP
#define NPPDX_EXAMPLE_COMMON_TEST_DATA_HOST_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace common {

    template<typename T>
    struct TestColor {
        const char* name;
        T           r, g, b;
    };

    template<typename T>
    inline constexpr T scale_color_value(uint8_t value_8bit, int target_bit_depth) {
        if (target_bit_depth == 8) {
            return static_cast<T>(value_8bit);
        }

        int max_target = (1 << target_bit_depth) - 1;
        return static_cast<T>((value_8bit * max_target + 127) / 255);
    }

    template<typename T>
    inline std::vector<TestColor<T>> get_test_colors(int bit_depth = 8) {
        static const struct {
            const char* name;
            uint8_t     r, g, b;
        } base_colors[16] = {{"Black", 0, 0, 0},        {"White", 255, 255, 255},
                             {"Red", 255, 0, 0},        {"Green", 0, 255, 0},
                             {"Blue", 0, 0, 255},       {"Yellow", 255, 255, 0},
                             {"Magenta", 255, 0, 255},  {"Cyan", 0, 255, 255},
                             {"Gray", 128, 128, 128},   {"Light Gray", 192, 192, 192},
                             {"Dark Gray", 64, 64, 64}, {"Dark Red", 128, 0, 0},
                             {"Dark Green", 0, 128, 0}, {"Dark Blue", 0, 0, 128},
                             {"Olive", 128, 128, 0},    {"Purple", 128, 0, 128}};

        std::vector<TestColor<T>> colors;
        colors.reserve(16);

        for (int i = 0; i < 16; ++i) {
            colors.push_back({base_colors[i].name, scale_color_value<T>(base_colors[i].r, bit_depth),
                              scale_color_value<T>(base_colors[i].g, bit_depth),
                              scale_color_value<T>(base_colors[i].b, bit_depth)});
        }

        return colors;
    }

    template<typename T = uint8_t>
    inline void generate_rgb_test_image(T* output, int width, int height, int block_size = 64, int bit_depth = 8) {
        auto      colors         = get_test_colors<T>(bit_depth);
        const int blocks_per_row = width / block_size;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int block_x   = x / block_size;
                int block_y   = y / block_size;
                int block_idx = (block_y * blocks_per_row + block_x) % 16;

                int idx         = (y * width + x) * 3;
                output[idx + 0] = colors[block_idx].r;
                output[idx + 1] = colors[block_idx].g;
                output[idx + 2] = colors[block_idx].b;
            }
        }
    }

    template<typename T = uint8_t>
    inline std::vector<T> generate_rgb_test_image(int width, int height, int block_size = 64, int bit_depth = 8) {
        std::vector<T> output(width * height * 3);
        generate_rgb_test_image<T>(output.data(), width, height, block_size, bit_depth);
        return output;
    }

    inline std::string get_rgb_test_data_path(const std::string& custom_path) {
        if (!custom_path.empty()) {
            return custom_path;
        }

        const char* data_dir = std::getenv("NPPDX_EXAMPLE_DATA_DIR");
        if (data_dir && data_dir[0] != '\0') {
            std::string path(data_dir);
            if (path.back() != '/' && path.back() != '\\') {
                path += '/';
            }
            return path + "cow_312x376.rgb24";
        }

        return "common/images/cow_312x376.rgb24";
    }

    inline std::vector<uint8_t> load_or_generate_rgb_test_data(int width, int height,
                                                               const std::string& custom_path = "") {
        const int            rgb_size = width * height * 3;
        std::vector<uint8_t> rgb_data(rgb_size);

        const std::string path = get_rgb_test_data_path(custom_path);
        std::ifstream     file(path, std::ios::binary);
        if (file && file.read(reinterpret_cast<char*>(rgb_data.data()), rgb_size) && file.gcount() == rgb_size) {
            std::cout << "Loaded RGB file: " << path << std::endl;
            return rgb_data;
        }

        std::cout << "Could not load RGB file from " << path << "; generating synthetic RGB test data" << std::endl;
        return generate_rgb_test_image<uint8_t>(width, height);
    }

    inline void write_output_file(const char* filename, const uint8_t* data, std::size_t size, bool verbose = true) {
        std::ofstream file(filename, std::ios::binary);
        if (file) {
            file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
            if (verbose)
                std::cout << "Wrote: " << filename << " (" << size << " bytes)" << std::endl;
        } else {
            std::cerr << "ERROR: could not open for write: " << filename << std::endl;
        }
    }

} // namespace common

#endif // NPPDX_EXAMPLE_COMMON_TEST_DATA_HOST_HPP
