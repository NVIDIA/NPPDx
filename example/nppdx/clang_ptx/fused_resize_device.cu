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

#include <cstdint>
#include <tuple>

#include <nppdx.hpp>

#ifndef NPPDX_CLANG_CUDA_COMPAT
#    error "This example is compiled by the Clang CUDA compatibility path only"
#endif

#ifndef NPPDX_CLANG_DEVICE_ARCH
#    error "NPPDX_CLANG_DEVICE_ARCH must be set by the Clang PTX example CMake target"
#endif

namespace nppdx_clang_fused_resize_detail {

    constexpr unsigned int tile_x        = 16;
    constexpr unsigned int tile_y        = 16;
    constexpr unsigned int kernel_w      = 5;
    constexpr unsigned int kernel_h      = 5;
    constexpr unsigned int scale_j       = 1;
    constexpr unsigned int scale_k       = 2;
    constexpr auto         resize_method = nppdx::interpolation_method::nearest;
    constexpr int          arch          = NPPDX_CLANG_DEVICE_ARCH;

    using ResizePre =
        decltype(nppdx::Function<nppdx::function::resize>() + nppdx::Resize<scale_j, scale_k, resize_method>() +
                 nppdx::TileSize<tile_x, tile_y>() + nppdx::SM<arch>() + nppdx::Block());
    constexpr auto resize_halo = ResizePre::local_halo;

    using BoxBlurPre = decltype(nppdx::Function<nppdx::function::box_blur>() + nppdx::BoxBlur<kernel_w, kernel_h>() +
                                nppdx::TileSize<tile_x, tile_y>() + nppdx::SM<arch>() + nppdx::Block());
    constexpr auto box_blur_halo = BoxBlurPre::local_halo;

    constexpr auto total_halo = box_blur_halo + resize_halo;

    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                            NPPDX_MAKE_HALO(nppdx::MemoryHalo, total_halo)() +
                            NPPDX_MAKE_HALO(nppdx::CumulativeHalo, total_halo)() + nppdx::SM<arch>() + nppdx::Block());

    using BoxBlur = decltype(BoxBlurPre() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, total_halo)() +
                             NPPDX_MAKE_HALO(nppdx::CumulativeHalo, total_halo)());

    using Resize = decltype(ResizePre() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, total_halo)() +
                            NPPDX_MAKE_HALO(nppdx::CumulativeHalo, resize_halo)());

    using InputTileStorageType        = nppdx::input_storage_of_t<Ingest>;
    using BlurredTileStorageType      = nppdx::output_storage_of_t<BoxBlur>;
    using IntermediateTileStorageType = nppdx::temp_storage_of_t<Resize>;
    using ResizedTileStorageType      = nppdx::output_storage_of_t<Resize>;

    constexpr unsigned int out_tile_x = ResizedTileStorageType::ChannelSliceType::width;
    constexpr unsigned int out_tile_y = ResizedTileStorageType::ChannelSliceType::height;

    constexpr auto empty_halo = nppdx::Halo4::make_empty();

    using ColorConvert =
        decltype(nppdx::Function<nppdx::function::color_convert>() +
                 nppdx::ColorConvert<nppdx::color_space::rgb, nppdx::color_space::yuv_bt601, nppdx::bit_depth::bpp_8u,
                                     nppdx::bit_depth::bpp_8u>() +
                 nppdx::TileSize<out_tile_x, out_tile_y>() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, empty_halo)() +
                 NPPDX_MAKE_HALO(nppdx::CumulativeHalo, empty_halo)() + nppdx::SM<arch>() + nppdx::Block());

    using Exgest =
        decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                 nppdx::OutputFormat<nppdx::packing_format::yuv422p>() + nppdx::TileSize<out_tile_x, out_tile_y>() +
                 NPPDX_MAKE_HALO(nppdx::MemoryHalo, empty_halo)() +
                 NPPDX_MAKE_HALO(nppdx::CumulativeHalo, empty_halo)() + nppdx::SM<arch>() + nppdx::Block());

    constexpr unsigned int block_x     = (scale_j > scale_k) ? Ingest::block_dim.x : Exgest::block_dim.x;
    constexpr unsigned int block_y     = (scale_j > scale_k) ? Ingest::block_dim.y : Exgest::block_dim.y;
    constexpr unsigned int block_z     = (scale_j > scale_k) ? Ingest::block_dim.z : Exgest::block_dim.z;
    constexpr unsigned int num_threads = block_x * block_y * block_z;

    constexpr unsigned int smem_size =
        nppdx::shared_memory::compute_total_tile_storage<InputTileStorageType, BlurredTileStorageType,
                                                         IntermediateTileStorageType, ResizedTileStorageType>();

} // namespace nppdx_clang_fused_resize_detail

extern "C" __global__ void nppdx_clang_fused_resize_query(unsigned int* config) {
    using namespace nppdx_clang_fused_resize_detail;
    config[0] = smem_size;
    config[1] = block_x;
    config[2] = block_y;
    config[3] = block_z;
    config[4] = out_tile_x;
    config[5] = out_tile_y;
}

extern "C" __global__ void nppdx_clang_fused_resize(const uint8_t* input, uint8_t* output, unsigned int input_width,
                                                    unsigned int input_height, unsigned int output_width,
                                                    unsigned int output_height) {

    using namespace nppdx_clang_fused_resize_detail;

    extern __shared__ unsigned char smem[];

    auto tile_storage =
        nppdx::shared_memory::slice_into_tile_storage<InputTileStorageType, BlurredTileStorageType,
                                                      IntermediateTileStorageType, ResizedTileStorageType>(smem);
    auto input_tile        = std::get<0>(tile_storage);
    auto blurred_tile      = std::get<1>(tile_storage);
    auto intermediate_tile = std::get<2>(tile_storage);
    auto resized_tile      = std::get<3>(tile_storage);

    Ingest().execute(input, input_tile, input_width, input_height);
    BoxBlur().execute(input_tile, blurred_tile, input_width, input_height);
    Resize().execute(blurred_tile, intermediate_tile, resized_tile, input_width, input_height);
    ColorConvert().execute(resized_tile, output_width, output_height);
    Exgest().execute(resized_tile, output, output_width, output_height);
}
