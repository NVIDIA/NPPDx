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

// NPPDx fused GPU path: ingest -> Gaussian blur -> RGB->YUV422P -> exgest (example-local; not shared "common" kernels).
#ifndef NPPDX_EXAMPLE_03_GAUSSIAN_BLUR_FUSED_GPU_CUH
#define NPPDX_EXAMPLE_03_GAUSSIAN_BLUR_FUSED_GPU_CUH

#include <nppdx.hpp>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../common.hpp"

namespace common {
    namespace gaussian_blur_npp_reference {

        template<typename Ingest, typename GaussianBlur, typename ColorConvert, typename Exgest,
                 typename InputTileStorageType, typename BlurredTileStorageType>
        // __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) is a bound for the compiler/ptxas, not the launch itself.
        // maxThreadsPerBlock=256: fused examples cap launches at 256 threads/block where we use that grid; <<<..., dim>>> must be <= this.
        // minBlocksPerSM=1: large dynamic SMEM per block - we are not designing for several co-resident blocks sharing SM resources.
        __launch_bounds__(256, 1) __global__
            void fused_gaussian_blur_convert_kernel(const uint8_t* input, uint8_t* output, const unsigned int width,
                                                    const unsigned int height) {
            extern __shared__ unsigned char smem[];

            auto tile_storage =
                nppdx::shared_memory::slice_into_tile_storage<InputTileStorageType, BlurredTileStorageType>(smem);
            auto input_channels   = std::get<0>(tile_storage);
            auto blurred_channels = std::get<1>(tile_storage);

            Ingest().execute(input, input_channels, width, height);

            GaussianBlur().execute(input_channels, blurred_channels, width, height);

            ColorConvert().execute(blurred_channels, width, height);

            Exgest().execute(blurred_channels, output, width, height);
        }

        // Radius 0.1-10.0 encoded as RadiusTenths (1-100); wide is limited to RadiusTenths <= 50.
        template<int Arch, unsigned int NumChannels, int RadiusTenths,
                 nppdx::gaussian_tail_width TailWidth = nppdx::gaussian_tail_width::standard>
        nppdx_results<uint8_t> run_nppdx_fused(uint8_t* d_input, uint8_t* d_output, const unsigned int width,
                                               const unsigned int height, const size_t yuv_size,
                                               unsigned int warm_up_runs, unsigned int runs, bool verbose = true) {
            constexpr unsigned int tile_x = 32;
            constexpr unsigned int tile_y = 32;
            constexpr nppdx::Halo4 halo   = nppdx::GaussianBlur<RadiusTenths, TailWidth>::value.halo();
            constexpr int          halo_x = halo.left_top.x;
            constexpr int          halo_y = halo.left_top.y;

            using Ingest =
                decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                         nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<tile_x, tile_y>() +
                         NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() + nppdx::SM<Arch>() + nppdx::Block());

            using GaussianBlur =
                decltype(nppdx::Function<nppdx::function::gaussian_blur>() +
                         nppdx::GaussianBlur<RadiusTenths, TailWidth>() +
                         nppdx::TileSize<tile_x, tile_y>() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() +
                         NPPDX_MAKE_HALO(nppdx::CumulativeHalo, halo)() + nppdx::SM<Arch>() + nppdx::Block());

            using ColorConvert = decltype(nppdx::Function<nppdx::function::color_convert>() +
                                          nppdx::ColorConvert<nppdx::color_space::rgb, nppdx::color_space::yuv_bt601,
                                                              nppdx::bit_depth::bpp_8u, nppdx::bit_depth::bpp_8u>() +
                                          NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() + nppdx::EmptyCumulativeHalo() +
                                          nppdx::TileSize<tile_x, tile_y>() + nppdx::SM<Arch>() + nppdx::Block());

            using Exgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                                    nppdx::OutputFormat<nppdx::packing_format::yuv422p>() +
                                    nppdx::TileSize<tile_x, tile_y>() + NPPDX_MAKE_HALO(nppdx::MemoryHalo, halo)() +
                                    nppdx::EmptyCumulativeHalo() + nppdx::SM<Arch>() + nppdx::Block());

            static_assert(nppdx::is_supported_v<Ingest, Arch>, "Ingest not supported");
            static_assert(nppdx::is_supported_v<GaussianBlur, Arch>, "GaussianBlur not supported");
            static_assert(nppdx::is_supported_v<ColorConvert, Arch>, "ColorConvert not supported");
            static_assert(nppdx::is_supported_v<Exgest, Arch>, "Exgest not supported");

            using InputTileStorageType   = nppdx::input_storage_of_t<Ingest>;
            using BlurredTileStorageType = nppdx::output_storage_of_t<GaussianBlur>;

            if (verbose) {
                std::cout << "Tile: " << tile_x << "x" << tile_y << " with halo=" << halo_x << "," << halo_y << ","
                          << halo_x << "," << halo_y << std::endl;
                std::cout << "Input storage tile: " << Ingest::input_storage.size.x << "x"
                          << Ingest::input_storage.size.y << std::endl;
                std::cout << "Blurred storage tile: " << GaussianBlur::output_storage.size.x << "x"
                          << GaussianBlur::output_storage.size.y << std::endl;
            }

            constexpr size_t smem_size =
                nppdx::shared_memory::compute_total_tile_storage<InputTileStorageType, BlurredTileStorageType>();
            if (verbose)
                std::cout << "Shared memory: " << smem_size << " bytes per block" << std::endl;

            constexpr dim3 block_dim = Ingest::block_dim;
            const dim3     grid_dim  = Ingest::calculate_grid_dim(width, height);

            if (verbose)
                std::cout << "Grid: " << grid_dim.x << "x" << grid_dim.y << " | Block: " << block_dim.x << "x"
                          << block_dim.y << std::endl;

            auto kernel_ptr = fused_gaussian_blur_convert_kernel<Ingest, GaussianBlur, ColorConvert, Exgest,
                                                                 InputTileStorageType, BlurredTileStorageType>;
            CUDA_CHECK_AND_EXIT(
                cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

            if (verbose)
                std::cout << "\nRunning NPPDx fused (Gaussian blur + convert)..." << std::endl;

            auto nppdx_execution = [&](cudaStream_t stream) {
                kernel_ptr<<<grid_dim, block_dim, smem_size, stream>>>(d_input, d_output, width, height);
            };

            nppdx_execution(0);
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            std::vector<uint8_t> h_nppdx_output(yuv_size);
            CUDA_CHECK_AND_EXIT(cudaMemcpy(h_nppdx_output.data(), d_output, yuv_size, cudaMemcpyDeviceToHost));

            auto ms = measure_execution_ms(nppdx_execution, warm_up_runs, runs, 0);
            if (verbose)
                std::cout << "NPPDx fused time: " << std::fixed << std::setprecision(6) << (ms / runs) << " ms"
                          << std::endl;

            return nppdx_results<uint8_t> {h_nppdx_output, ms / runs};
        }

    } // namespace gaussian_blur_npp_reference
} // namespace common

#endif // NPPDX_EXAMPLE_03_GAUSSIAN_BLUR_FUSED_GPU_CUH
