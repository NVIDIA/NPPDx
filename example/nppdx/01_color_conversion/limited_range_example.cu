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

#include <nppdx.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "../common.hpp"

namespace {

    constexpr int tile_width  = 32;
    constexpr int tile_height = 32;

    template<typename IngestOp, typename Map0, typename Map1, typename ExgestOp>
    __global__ void affine_range_kernel(const uint8_t* input, uint8_t* output, int width, int height) {
        float data[IngestOp::elements_per_thread];

        // The range conversion is represented as ordinary NPPDx operators:
        // ingest samples, apply one or two affine maps, then exgest.
        IngestOp().execute(input, data, width, height);
        Map0().execute(data, width, height);
        Map1().execute(data, width, height);
        ExgestOp().execute(data, output, width, height);
    }

    template<nppdx::packing_format Format, bool ToLimited>
    std::vector<uint8_t> make_source(int width, int height);

    template<nppdx::packing_format Format, bool ToLimited>
    bool output_is_in_target_range(const std::vector<uint8_t>& data, int width, int height);

    template<nppdx::packing_format Format>
    void print_sample(const std::vector<uint8_t>& input, const std::vector<uint8_t>& output, int width, int x, int y);

    template<nppdx::packing_format Format, typename Map0, typename Map1, int Arch, bool ToLimited>
    bool run_range_case(const char* label, int width, int height) {
        using IngestOp =
            decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() + nppdx::InputFormat<Format>() +
                     nppdx::TileSize<tile_width, tile_height>() + nppdx::SM<Arch>() + nppdx::Block());

        using ExgestOp =
            decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() + nppdx::OutputFormat<Format>() +
                     nppdx::TileSize<tile_width, tile_height>() + nppdx::SM<Arch>() + nppdx::Block());

        static_assert(nppdx::is_supported_v<IngestOp, Arch>);
        static_assert(nppdx::is_supported_v<Map0, Arch>);
        static_assert(nppdx::is_supported_v<Map1, Arch>);
        static_assert(nppdx::is_supported_v<ExgestOp, Arch>);

        const std::vector<uint8_t> h_input = make_source<Format, ToLimited>(width, height);
        std::vector<uint8_t>       h_output(h_input.size(), 0);

        common::DevBuf d_input(h_input.size());
        common::DevBuf d_output(h_output.size());

        CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_input.data(), h_input.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaMemset(d_output.d_buf, 0, h_output.size()));

        constexpr dim3 block_dim = IngestOp::block_dim;
        const dim3     grid_dim  = IngestOp::calculate_grid_dim(width, height);

        affine_range_kernel<IngestOp, Map0, Map1, ExgestOp>
            <<<grid_dim, block_dim>>>(d_input.d_buf, d_output.d_buf, width, height);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        CUDA_CHECK_AND_EXIT(cudaMemcpy(h_output.data(), d_output.d_buf, h_output.size(), cudaMemcpyDeviceToHost));

        const bool in_range = output_is_in_target_range<Format, ToLimited>(h_output, width, height);

        std::cout << label << " (" << (in_range ? "range ok" : "range failed") << ")" << std::endl;
        print_sample<Format>(h_input, h_output, width, 0, 0);
        print_sample<Format>(h_input, h_output, width, 1, 0);
        print_sample<Format>(h_input, h_output, width, 0, 1);
        print_sample<Format>(h_input, h_output, width, 1, 1);

        return in_range;
    }

    template<nppdx::bit_depth Depth, int Arch>
    bool run_rgb_cases(const char* depth_label, int width, int height) {
        constexpr auto format =
            (Depth == nppdx::bit_depth::bpp_8u) ? nppdx::packing_format::rgb24 : nppdx::packing_format::rgb10;

        using ToLimited = decltype(nppdx::Function<nppdx::function::affine_channel_map>() +
                                   nppdx::ToLimitedLuma<Depth, nppdx::affine_channel_mask::channels_rgb>() +
                                   nppdx::TileSize<tile_width, tile_height>() + nppdx::SM<Arch>() + nppdx::Block());

        using FromLimited = decltype(nppdx::Function<nppdx::function::affine_channel_map>() +
                                     nppdx::FromLimitedLuma<Depth, nppdx::affine_channel_mask::channels_rgb>() +
                                     nppdx::TileSize<tile_width, tile_height>() + nppdx::SM<Arch>() + nppdx::Block());

        using Identity = decltype(nppdx::Function<nppdx::function::affine_channel_map>() +
                                  nppdx::AffineChannelMap<0, 1, 1, 0, NPPDX_AFFINE_NO_CLIP, NPPDX_AFFINE_NO_CLIP,
                                                          nppdx::affine_channel_mask::channels_all>() +
                                  nppdx::TileSize<tile_width, tile_height>() + nppdx::SM<Arch>() + nppdx::Block());

        std::cout << "\nRGB " << depth_label << std::endl;
        bool verified = true;
        verified &= run_range_case<format, ToLimited, Identity, Arch, true>("  full -> limited", width, height);
        verified &= run_range_case<format, FromLimited, Identity, Arch, false>("  limited -> full", width, height);
        return verified;
    }

    template<nppdx::bit_depth Depth, int Arch>
    bool run_yuv_bt709_cases(const char* depth_label, int width, int height) {
        // Once samples are already in YUV, full/limited range mapping is independent of the YUV matrix.
        constexpr auto format =
            (Depth == nppdx::bit_depth::bpp_8u) ? nppdx::packing_format::yuv444p : nppdx::packing_format::yuv444p10;

        using ToLimitedY =
            decltype(nppdx::Function<nppdx::function::affine_channel_map>() + nppdx::ToLimitedLuma<Depth>() +
                     nppdx::TileSize<tile_width, tile_height>() + nppdx::SM<Arch>() + nppdx::Block());

        // Chroma is centered around the neutral code value, so U/V use a
        // different affine map from Y.
        using ToLimitedUV =
            decltype(nppdx::Function<nppdx::function::affine_channel_map>() + nppdx::ToLimitedChroma<Depth>() +
                     nppdx::TileSize<tile_width, tile_height>() + nppdx::SM<Arch>() + nppdx::Block());

        using FromLimitedY =
            decltype(nppdx::Function<nppdx::function::affine_channel_map>() + nppdx::FromLimitedLuma<Depth>() +
                     nppdx::TileSize<tile_width, tile_height>() + nppdx::SM<Arch>() + nppdx::Block());

        using FromLimitedUV =
            decltype(nppdx::Function<nppdx::function::affine_channel_map>() + nppdx::FromLimitedChroma<Depth>() +
                     nppdx::TileSize<tile_width, tile_height>() + nppdx::SM<Arch>() + nppdx::Block());

        std::cout << "\nYUV BT.709 4:4:4 " << depth_label << std::endl;
        // Y and UV are separate affine ops, which is the intended channel-mask usage.
        bool verified = true;
        verified &= run_range_case<format, ToLimitedY, ToLimitedUV, Arch, true>("  full -> limited", width, height);
        verified &=
            run_range_case<format, FromLimitedY, FromLimitedUV, Arch, false>("  limited -> full", width, height);
        return verified;
    }

    template<nppdx::packing_format Format>
    struct format_access;

    template<>
    struct format_access<nppdx::packing_format::rgb24> {
        static constexpr nppdx::bit_depth depth  = nppdx::bit_depth::bpp_8u;
        static constexpr bool             is_yuv = false;

        static size_t buffer_bytes(int width, int height) {
            return nppdx::get_packing_format_props(nppdx::packing_format::rgb24)
                .minimum_buffer_byte_size(static_cast<size_t>(width), static_cast<size_t>(height));
        }

        static int get_channel(const std::vector<uint8_t>& data, int width, int x, int y, int channel) {
            return data[(static_cast<size_t>(y) * width + x) * 3 + channel];
        }

        static void set_channel(std::vector<uint8_t>& data, int width, int x, int y, int channel, int value) {
            data[(static_cast<size_t>(y) * width + x) * 3 + channel] = static_cast<uint8_t>(value);
        }
    };

    template<>
    struct format_access<nppdx::packing_format::rgb10> {
        static constexpr nppdx::bit_depth depth  = nppdx::bit_depth::bpp_10u;
        static constexpr bool             is_yuv = false;

        static size_t buffer_bytes(int width, int height) {
            return nppdx::get_packing_format_props(nppdx::packing_format::rgb10)
                .minimum_buffer_byte_size(static_cast<size_t>(width), static_cast<size_t>(height));
        }

        static uint32_t get_pixel(const std::vector<uint8_t>& data, int width, int x, int y) {
            uint32_t pixel = 0;
            std::memcpy(&pixel, data.data() + (static_cast<size_t>(y) * width + x) * sizeof(uint32_t), sizeof(pixel));
            return pixel;
        }

        static void set_pixel(std::vector<uint8_t>& data, int width, int x, int y, uint32_t pixel) {
            std::memcpy(data.data() + (static_cast<size_t>(y) * width + x) * sizeof(uint32_t), &pixel, sizeof(pixel));
        }

        static int get_channel(const std::vector<uint8_t>& data, int width, int x, int y, int channel) {
            const uint32_t pixel = get_pixel(data, width, x, y);
            if (channel == 0) {
                return static_cast<int>((pixel >> 20) & 0x3FFu);
            }
            if (channel == 1) {
                return static_cast<int>((pixel >> 10) & 0x3FFu);
            }
            return static_cast<int>(pixel & 0x3FFu);
        }

        static void set_channel(std::vector<uint8_t>& data, int width, int x, int y, int channel, int value) {
            uint32_t       pixel  = get_pixel(data, width, x, y);
            const uint32_t sample = static_cast<uint32_t>(value) & 0x3FFu;
            if (channel == 0) {
                pixel = (pixel & 0x000FFFFFu) | (sample << 20);
            } else if (channel == 1) {
                pixel = (pixel & 0x3FF003FFu) | (sample << 10);
            } else {
                pixel = (pixel & 0x3FFFFC00u) | sample;
            }
            set_pixel(data, width, x, y, pixel);
        }
    };

    template<>
    struct format_access<nppdx::packing_format::yuv444p> {
        static constexpr nppdx::bit_depth depth  = nppdx::bit_depth::bpp_8u;
        static constexpr bool             is_yuv = true;

        static size_t buffer_bytes(int width, int height) {
            return nppdx::get_packing_format_props(nppdx::packing_format::yuv444p)
                .minimum_buffer_byte_size(static_cast<size_t>(width), static_cast<size_t>(height));
        }

        static int get_channel(const std::vector<uint8_t>& data, int width, int x, int y, int channel) {
            const size_t plane_size = data.size() / 3u;
            const size_t pos        = static_cast<size_t>(y) * width + x;
            return data[static_cast<size_t>(channel) * plane_size + pos];
        }

        static void set_channel(std::vector<uint8_t>& data, int width, int x, int y, int channel, int value) {
            const size_t plane_size                               = data.size() / 3u;
            const size_t pos                                      = static_cast<size_t>(y) * width + x;
            data[static_cast<size_t>(channel) * plane_size + pos] = static_cast<uint8_t>(value);
        }
    };

    template<>
    struct format_access<nppdx::packing_format::yuv444p10> {
        static constexpr nppdx::bit_depth depth  = nppdx::bit_depth::bpp_10u;
        static constexpr bool             is_yuv = true;

        static size_t buffer_bytes(int width, int height) {
            return nppdx::get_packing_format_props(nppdx::packing_format::yuv444p10)
                .minimum_buffer_byte_size(static_cast<size_t>(width), static_cast<size_t>(height));
        }

        static size_t sample_offset(int width, int height, int x, int y, int channel) {
            const size_t plane_size = static_cast<size_t>(width) * static_cast<size_t>(height);
            return (static_cast<size_t>(channel) * plane_size + static_cast<size_t>(y) * width + x) * sizeof(uint16_t);
        }

        static int get_channel(const std::vector<uint8_t>& data, int width, int x, int y, int channel) {
            const int height = static_cast<int>(data.size() / (static_cast<size_t>(width) * 3u * sizeof(uint16_t)));
            uint16_t  sample = 0;
            std::memcpy(&sample, data.data() + sample_offset(width, height, x, y, channel), sizeof(sample));
            return static_cast<int>(sample & 0x3FFu);
        }

        static void set_channel(std::vector<uint8_t>& data, int width, int x, int y, int channel, int value) {
            const int height = static_cast<int>(data.size() / (static_cast<size_t>(width) * 3u * sizeof(uint16_t)));
            const uint16_t sample = static_cast<uint16_t>(value & 0x3FF);
            std::memcpy(data.data() + sample_offset(width, height, x, y, channel), &sample, sizeof(sample));
        }
    };

    template<nppdx::bit_depth Depth, bool ToLimited, bool IsYuv>
    int source_value(int channel, int x, int y) {
        constexpr auto values = nppdx::range_values<Depth>;
        const bool     high   = ((x + y + channel) % 2) != 0;

        // Use only range endpoints so the example checks the important cases:
        // black/white for luma and RGB, plus low/high legal chroma.
        if constexpr (ToLimited) {
            return high ? values.full_max() : 0;
        } else if constexpr (!IsYuv) {
            return high ? values.y_max() : values.y_min();
        } else if (channel == 0) {
            return high ? values.y_max() : values.y_min();
        } else {
            return high ? values.chroma_max() : values.chroma_min();
        }
    }

    template<nppdx::packing_format Format, bool ToLimited>
    std::vector<uint8_t> make_source(int width, int height) {
        using access = format_access<Format>;

        std::vector<uint8_t> data(access::buffer_bytes(width, height), 0);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int channel = 0; channel < 3; ++channel) {
                    access::set_channel(data, width, x, y, channel,
                                        source_value<access::depth, ToLimited, access::is_yuv>(channel, x, y));
                }
            }
        }
        return data;
    }

    template<nppdx::bit_depth Depth, bool ToLimited, bool IsYuv>
    bool value_is_in_target_range(int channel, int value) {
        constexpr auto values = nppdx::range_values<Depth>;
        if constexpr (ToLimited) {
            if constexpr (IsYuv) {
                if (channel == 0) {
                    return value >= values.y_min() && value <= values.y_max();
                }
                return value >= values.chroma_min() && value <= values.chroma_max();
            }
            return value >= values.y_min() && value <= values.y_max();
        }

        return value >= 0 && value <= values.full_max();
    }

    template<nppdx::packing_format Format, bool ToLimited>
    bool output_is_in_target_range(const std::vector<uint8_t>& data, int width, int height) {
        using access = format_access<Format>;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int channel = 0; channel < 3; ++channel) {
                    if (!value_is_in_target_range<access::depth, ToLimited, access::is_yuv>(
                            channel, access::get_channel(data, width, x, y, channel))) {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    template<nppdx::packing_format Format>
    void print_sample(const std::vector<uint8_t>& input, const std::vector<uint8_t>& output, int width, int x, int y) {
        using access     = format_access<Format>;
        const char* name = access::is_yuv ? "YUV" : "RGB";

        std::cout << "    (" << x << "," << y << ") " << name << "(" << access::get_channel(input, width, x, y, 0)
                  << "," << access::get_channel(input, width, x, y, 1) << ","
                  << access::get_channel(input, width, x, y, 2) << ") -> " << name << "("
                  << access::get_channel(output, width, x, y, 0) << "," << access::get_channel(output, width, x, y, 1)
                  << "," << access::get_channel(output, width, x, y, 2) << ")" << std::endl;
    }

} // namespace

template<int Arch>
int limited_range_example() {
    std::cout << "=== NPPDx Limited Range Example (SM" << Arch << ") ===" << std::endl;

    constexpr int width  = 64;
    constexpr int height = 32;

    std::cout << "Image: " << width << "x" << height << " | Tile: " << tile_width << "x" << tile_height << "\n"
              << std::endl;

    bool verified = true;
    verified &= run_rgb_cases<nppdx::bit_depth::bpp_8u, Arch>("8-bit", width, height);
    verified &= run_rgb_cases<nppdx::bit_depth::bpp_10u, Arch>("10-bit", width, height);
    verified &= run_yuv_bt709_cases<nppdx::bit_depth::bpp_8u, Arch>("8-bit", width, height);
    verified &= run_yuv_bt709_cases<nppdx::bit_depth::bpp_10u, Arch>("10-bit", width, height);

    std::cout << "\nExample complete: " << (verified ? "SUCCESS" : "FAILED") << std::endl;
    return verified ? 0 : 1;
}

template<int Arch>
struct limited_range_example_functor {
    int operator()() const { return limited_range_example<Arch>(); }
};

int main() {
    return common::run_example_with_sm<limited_range_example_functor>();
}
