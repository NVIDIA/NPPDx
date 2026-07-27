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

// Texture ingest and surface exgest for CUDA arrays in NPPDx.
//
// Most NPPDx examples operate on "pitch-linear" memory: a plain device pointer whose rows follow
// one another at a fixed byte stride. CUDA arrays instead use an opaque tiled layout that the
// hardware optimizes for 2D spatial locality. You don't dereference a
// pointer into them; you read through a texture object (tex2D) and write through a surface object
// (surf2Dwrite), and the texture/surface units do the address swizzling for you.
//
// Why this matters: graphics and video pipelines hand their images to CUDA as CUDA arrays.
// OpenGL / Vulkan / Direct3D textures and NVENC/NVDEC frame buffers are all exposed this way. By
// ingesting from a texture and exgesting to a surface, an NPPDx pipeline can consume and produce
// those buffers directly, with no extra copy or conversion to pitch-linear in between.
//
// This example demonstrates RGB24 in both directions:
//   Pass 1 - texture ingest (tex2D read from a CUDA array) -> pitch-linear pointer exgest.
//   Pass 2 - pitch-linear pointer ingest -> surface exgest (surf2Dwrite to a CUDA array).

#include <iostream>
#include <vector>

#include <nppdx.hpp>
#include "../common.hpp"
#include "../common/example_sm_runner.hpp"


using namespace nppdx;

// ============================================================================
// Kernels
// ============================================================================

// Read a CUDA array through a texture, then write the result to a pitch-linear pointer.
template<typename IngestOp, typename ExgestOp>
__global__ void tex_ingest_ptr_exgest_kernel(nppdx::TexObjT texInput, uint8_t* ptrOutput, int width, int height) {
    using processing_t                   = nppdx::processing_type_of_t<IngestOp>;
    constexpr size_t elements_per_thread = nppdx::elements_per_thread_of_v<IngestOp>;

    processing_t intermediate_data[elements_per_thread];

    IngestOp().execute(texInput, intermediate_data, width, height);
    ExgestOp().execute(intermediate_data, ptrOutput, width, height);
}

// Read a pitch-linear pointer, then write the result to a CUDA array through a surface.
template<typename IngestOp, typename ExgestOp>
__global__ void ptr_ingest_surf_exgest_kernel(const uint8_t* ptrInput, nppdx::SurfObjT surfOutput, int width,
                                              int height) {
    using processing_t                   = nppdx::processing_type_of_t<IngestOp>;
    constexpr size_t elements_per_thread = nppdx::elements_per_thread_of_v<IngestOp>;

    processing_t intermediate_data[elements_per_thread];

    IngestOp().execute(ptrInput, intermediate_data, width, height);
    ExgestOp().execute(intermediate_data, surfOutput, width, height);
}

// ============================================================================
// Helpers
// ============================================================================

// Generate a deterministic RGB24 test pattern.
static std::vector<uint8_t> generate_rgb24_test_data(int width, int height) {
    const int            size = width * height * 3;
    std::vector<uint8_t> data(size);
    for (int i = 0; i < size; i += 3) {
        data[i + 0] = static_cast<uint8_t>((i / 3) % 256);       // R
        data[i + 1] = static_cast<uint8_t>(128 + (i / 3) % 256); // G
        data[i + 2] = static_cast<uint8_t>(64 + (i / 3) % 256);  // B
    }
    return data;
}

// Pack RGB24 -> uchar4 (pad alpha = 255).
static std::vector<uchar4> pack_rgb24_to_uchar4(const std::vector<uint8_t>& rgb, int width, int height) {
    std::vector<uchar4> rgba(width * height);
    for (int i = 0; i < width * height; ++i) {
        rgba[i] = make_uchar4(rgb[i * 3 + 0], rgb[i * 3 + 1], rgb[i * 3 + 2], 255);
    }
    return rgba;
}

// Texture object over a CUDA array, configured the way texture ingest expects:
// point sampling (no interpolation, exact texels) and clamp addressing (out-of-bounds reads
// replicate the edge, which supplies the halo for tiled/area operations).
static nppdx::TexObjT create_point_clamp_texture(cudaArray_t cuArray) {
    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = cuArray;

    cudaTextureDesc texDesc  = {};
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK_AND_EXIT(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    return static_cast<nppdx::TexObjT>(texObj);
}

// Surface object over a CUDA array. The array must be allocated with cudaArraySurfaceLoadStore
// for surf2Dwrite to be able to write into it.
static nppdx::SurfObjT create_surface(cudaArray_t cuArray) {
    cudaResourceDesc resDesc = {};
    resDesc.resType          = cudaResourceTypeArray;
    resDesc.res.array.array  = cuArray;

    cudaSurfaceObject_t surfObj = 0;
    CUDA_CHECK_AND_EXIT(cudaCreateSurfaceObject(&surfObj, &resDesc));
    return static_cast<nppdx::SurfObjT>(surfObj);
}

// ============================================================================
// Example
// ============================================================================

template<int SM>
struct texture_example {
    int operator()() {
        std::cout << "Running Texture PoC with SM<" << SM << ">" << std::endl;

        const int width    = 640;
        const int height   = 480;
        const int rgb_size = width * height * 3;

        auto h_input_rgb  = generate_rgb24_test_data(width, height);
        auto h_input_rgba = pack_rgb24_to_uchar4(h_input_rgb, width, height);

        using IngestOp = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                                  nppdx::InputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<48, 48>() +
                                  nppdx::Block() + nppdx::SM<SM>());
        using ExgestOp = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                                  nppdx::OutputFormat<nppdx::packing_format::rgb24>() + nppdx::TileSize<48, 48>() +
                                  nppdx::Block() + nppdx::SM<SM>());

        constexpr auto tile_size = IngestOp::suggested_tile_size;
        constexpr auto block_dim = IngestOp::block_dim;

        const unsigned int num_blocks_x = (width + tile_size.x - 1) / tile_size.x;
        const unsigned int num_blocks_y = (height + tile_size.y - 1) / tile_size.y;
        const dim3         grid_dim(num_blocks_x, num_blocks_y, 1);

        std::cout << "Image size: " << width << "x" << height << std::endl;

        bool all_passed = true;

        // ====================================================================
        // Pass 1: Texture ingest -> pointer exgest
        // ====================================================================
        {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
            cudaArray_t           cuArray     = nullptr;
            CUDA_CHECK_AND_EXIT(cudaMallocArray(&cuArray, &channelDesc, width, height));

            CUDA_CHECK_AND_EXIT(cudaMemcpy2DToArray(cuArray, 0, 0, h_input_rgba.data(), width * sizeof(uchar4),
                                                    width * sizeof(uchar4), height, cudaMemcpyHostToDevice));

            nppdx::TexObjT texObj = create_point_clamp_texture(cuArray);

            common::DevBuf d_output(rgb_size);

            tex_ingest_ptr_exgest_kernel<IngestOp, ExgestOp>
                <<<grid_dim, block_dim>>>(texObj, d_output.d_buf, width, height);
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            std::vector<uint8_t> h_output(rgb_size, 0);
            CUDA_CHECK_AND_EXIT(cudaMemcpy(h_output.data(), d_output.d_buf, rgb_size, cudaMemcpyDeviceToHost));

            auto stats = common::compute_conversion_stats(h_input_rgb.data(), h_output.data(), rgb_size,
                                                          /*error_threshold=*/0, /*max_error_limit=*/0,
                                                          /*avg_error_limit=*/0.0, /*print=*/true,
                                                          /*test_name=*/"Texture ingest -> pointer exgest");
            all_passed &= stats.passed;

            CUDA_CHECK_AND_EXIT(cudaDestroyTextureObject(static_cast<cudaTextureObject_t>(texObj)));
            CUDA_CHECK_AND_EXIT(cudaFreeArray(cuArray));
        }

        // ====================================================================
        // Pass 2: Pointer ingest -> surface exgest
        // ====================================================================
        {
            common::DevBuf d_input(rgb_size);
            CUDA_CHECK_AND_EXIT(cudaMemcpy(d_input.d_buf, h_input_rgb.data(), rgb_size, cudaMemcpyHostToDevice));

            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
            cudaArray_t           cuArray     = nullptr;
            CUDA_CHECK_AND_EXIT(cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore));

            nppdx::SurfObjT surfObj = create_surface(cuArray);

            ptr_ingest_surf_exgest_kernel<IngestOp, ExgestOp>
                <<<grid_dim, block_dim>>>(d_input.d_buf, surfObj, width, height);
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            std::vector<uchar4> h_output_rgba(width * height);
            CUDA_CHECK_AND_EXIT(cudaMemcpy2DFromArray(h_output_rgba.data(), width * sizeof(uchar4), cuArray, 0, 0,
                                                      width * sizeof(uchar4), height, cudaMemcpyDeviceToHost));

            // Unpack uchar4 -> RGB24 for comparison.
            std::vector<uint8_t> h_output(rgb_size);
            for (int i = 0; i < width * height; ++i) {
                h_output[i * 3 + 0] = h_output_rgba[i].x;
                h_output[i * 3 + 1] = h_output_rgba[i].y;
                h_output[i * 3 + 2] = h_output_rgba[i].z;
            }

            auto stats = common::compute_conversion_stats(h_input_rgb.data(), h_output.data(), rgb_size,
                                                          /*error_threshold=*/0, /*max_error_limit=*/0,
                                                          /*avg_error_limit=*/0.0, /*print=*/true,
                                                          /*test_name=*/"Pointer ingest -> surface exgest");
            all_passed &= stats.passed;

            CUDA_CHECK_AND_EXIT(cudaDestroySurfaceObject(static_cast<cudaSurfaceObject_t>(surfObj)));
            CUDA_CHECK_AND_EXIT(cudaFreeArray(cuArray));
        }

        std::cout << (all_passed ? "All tests PASSED" : "Some tests FAILED") << std::endl;
        return all_passed ? 0 : 1;
    }
};

int main() {
    std::cout << "NPPDx Texture Example" << std::endl;
    std::cout << "RGB24 texture ingest (tex2D) and surface exgest (surf2Dwrite)" << std::endl;

    return common::run_example_with_sm<texture_example>();
}
