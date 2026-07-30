# NPPDx

[![Version](https://img.shields.io/badge/version-v0.1.1-blue)](#)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE.txt)
[![Platforms](https://img.shields.io/badge/platform-linux--x86__64%20%7C%20windows--x64-lightgrey)](#prerequisites)
[![CUDA](https://img.shields.io/badge/CUDA-%3E%3D13.0-76B900)](#prerequisites)
[![GCC](https://img.shields.io/badge/GCC-%3E%3D10-blue)](#prerequisites)
[![Clang Host](https://img.shields.io/badge/Clang%20host-%3E%3D15-blue)](#prerequisites)
[![MSVC](https://img.shields.io/badge/MSVC-%3E%3D1944%20%28VS%202022%29-blue)](#prerequisites)
[![Clang PTX](https://img.shields.io/badge/Clang%20PTX-%3E%3D21-blue)](#clang-device-compilation)
[![CMake](https://img.shields.io/badge/CMake-%3E%3D3.30-blue)](#prerequisites)

NPPDx is a header-only CUDA C++ library for building image-processing kernels with
the NVIDIA Dx programming model. NPPDx lets applications describe image I/O,
format conversion, pointwise transforms, area operations, and resize operations
as composable C++ types, then execute those operations directly inside user CUDA
kernels.

NPPDx operations are configured at compile time by composing operators such as
`InputOutput`, `InputFormat`, `OutputFormat`, `Function`, `TileSize`, `SM`, and
`Block`. The resulting operation type exposes launch traits, such as block
dimensions and shared-memory requirements, and provides an `execute()` member
function for use in CUDA device code.

## Features

- Header-only CUDA C++ API with C++17 support.
- Dx-style operator composition for image-processing descriptions.
- Ingest and exgest operations for reading image data into registers/shared memory and
  writing processed data back to image buffers.
- Register-backed and shared-memory-backed tile processing.
- Texture ingest and surface exgest for CUDA array-based workflows.
- Color conversion between RGB and YUV color spaces.
- Pointwise operations including gamma transforms and affine channel mapping.
- Area operations including box blur, Gaussian blur, sharpen, and median.
- Rational resize with nearest, bilinear, bicubic, and Lanczos3 interpolation.
- CMake package integration through `find_package(nppdx CONFIG)`.

## Documentation

Detailed documentation is available at https://nvidia.github.io/NPPDx/.

## Supported Formats

NPPDx supports the following packing formats:

- Packed RGB: `rgb24`, `rgb10`, `rgb16`
- Packed YUV: `y210`, `uyvp`, `v210`, `yuv2`
- Semi-planar YUV: `nv12`, `p010`, `nv16`, `p216`
- Planar RGB/BGR: `rgbp`, `bgrp`
- Planar YUV: `yuv420p`, `yuv420p10`, `yuv422p`, `yuv422p10`, `yuv444p`,
  `yuv444p10`

## Prerequisites

- Linux x86_64 or Windows x64.
- NVIDIA GPU architecture `sm_75` or newer.
- CUDA Toolkit 13.0 or newer.
- CMake 3.30 or newer.
- C++17 and CUDA C++17 capable host compiler:
  - GCC 10 or newer on Linux.
  - Clang 15 or newer when using Clang as host compiler.
  - MSVC 1944 or newer with Visual Studio 2022 on Windows.
- Clang 21 or newer for Clang device compilation to PTX.

## Repository Layout

The NPPDx repository contains a ready-to-use layout, so there is no top-level
NPPDx configure, build, or install step. After cloning or unpacking it, use the
repository root directly:

```bash
git clone https://github.com/NVIDIA/nppdx.git
cd nppdx
export NPPDX_ROOT="$PWD"
```

The repository contains:

- `README.md`: this README.
- `LICENSE.txt`: license text.
- `CONTRIBUTING.md`: contribution guidelines.
- `include/`: NPPDx headers and bundled commonDx headers.
- `lib/cmake/nppdx/`: NPPDx CMake package configuration files.
- `lib/cmake/commondx/`: commonDx CMake package configuration files.
- `example/nppdx/`: CUDA and Clang PTX examples.

On Windows, set `NPPDX_ROOT` to the repository root:

```cmd
git clone https://github.com/NVIDIA/nppdx.git
cd nppdx
set NPPDX_ROOT=%CD%
```

## Build Examples

The example CMake project supports these NPPDx options:

- `NPPDX_BUILD_CUDA_EXAMPLES`: build CUDA examples, default `ON`.
- `NPPDX_BUILD_CLANG_PTX_EXAMPLES`: build Clang device-only PTX examples,
  default `OFF`.
- `NPPDX_CUDA_ARCHITECTURES`: CUDA architectures for example builds.
- `NPPDX_CLANG_PTX_COMPAT_INCLUDE_DIR`: compatibility include directory that
  provides `cuda_runtime.h` for Clang PTX examples.

```bash
cmake -S "${NPPDX_ROOT}/example/nppdx" -B build-nppdx-examples \
      -DNPPDX_BUILD_CUDA_EXAMPLES=ON \
      -DNPPDX_BUILD_CLANG_PTX_EXAMPLES=OFF \
      -DNPPDX_CUDA_ARCHITECTURES=80-real

cmake --build build-nppdx-examples --target nppdx_examples
ctest --test-dir build-nppdx-examples --output-on-failure
```

Use `NPPDX_CUDA_ARCHITECTURES` to select the GPU architectures for the examples,
for example `80-real`, `90-real`, or `80-real;90-real`.

On Windows, configure from a Visual Studio Developer Command Prompt or another
CUDA-capable CMake environment:

```cmd
cmake -S "%NPPDX_ROOT%\example\nppdx" -B build-nppdx-examples ^
      -DNPPDX_BUILD_CUDA_EXAMPLES=ON ^
      -DNPPDX_BUILD_CLANG_PTX_EXAMPLES=OFF ^
      -DNPPDX_CUDA_ARCHITECTURES=90-real

cmake --build build-nppdx-examples --target nppdx_examples --config Release
ctest --test-dir build-nppdx-examples -C Release --output-on-failure
```

## NPPDx in a CMake project

Add NPPDx to a CUDA target with the package target `nppdx::nppdx`:

```cmake
cmake_minimum_required(VERSION 3.30)

project(the_nppdx_app LANGUAGES CXX CUDA)

find_package(nppdx REQUIRED CONFIG)

add_executable(the_nppdx_app main.cu)
target_link_libraries(the_nppdx_app nppdx::nppdx)
set_target_properties(the_nppdx_app PROPERTIES CUDA_ARCHITECTURES "80-real")
```

Configure the project with `CMAKE_PREFIX_PATH` pointing at the NPPDx root:

```bash
cmake -S /path/to/the_nppdx_app -B build-the-nppdx-app \
      -DCMAKE_PREFIX_PATH="${NPPDX_ROOT}"
```

The configuration provides the imported INTERFACE target `nppdx::nppdx` and the
variables `nppdx_INCLUDE_DIRS` and `nppdx_VERSION`.

A CUDA kernel can compose and execute NPPDx operations directly:

```cuda
#include <nppdx.hpp>

template<typename Ingest, typename Convert, typename Exgest>
__global__ void convert_kernel(const uint8_t* input, uint8_t* output,
                               int width, int height) {
    float tile_data[Ingest::elements_per_thread];

    Ingest().execute(input, tile_data, width, height);
    Convert().execute(tile_data, width, height);
    Exgest().execute(tile_data, output, width, height);
}

template<int SM>
void launch_rgb_to_yuv(const uint8_t* input, uint8_t* output,
                       int width, int height) {
    using Ingest = decltype(nppdx::InputOutput<nppdx::input_output_direction::ingest>() +
                            nppdx::InputFormat<nppdx::packing_format::rgb24>() +
                            nppdx::TileSize<48, 48>() + nppdx::SM<SM>() + nppdx::Block());

    using Convert = decltype(nppdx::Function<nppdx::function::color_convert>() +
                             nppdx::ColorConvert<nppdx::color_space::rgb,
                                                 nppdx::color_space::yuv_bt601,
                                                 nppdx::bit_depth::bpp_8u,
                                                 nppdx::bit_depth::bpp_8u>() +
                             nppdx::TileSize<48, 48>() + nppdx::SM<SM>() + nppdx::Block());

    using Exgest = decltype(nppdx::InputOutput<nppdx::input_output_direction::exgest>() +
                            nppdx::OutputFormat<nppdx::packing_format::yuv420p>() +
                            nppdx::TileSize<48, 48>() + nppdx::SM<SM>() + nppdx::Block());

    dim3 grid = Ingest::calculate_grid_dim(width, height);
    convert_kernel<Ingest, Convert, Exgest>
        <<<grid, Ingest::block_dim>>>(input, output, width, height);
}
```

## Clang Device Compilation

The examples include a Clang PTX path that compiles CUDA device code to PTX with
Clang and runs a C++ host executable through the CUDA Driver API.

To build the Clang PTX examples, configure the example directory with Clang as
the C++ compiler:

```bash
cmake -S "${NPPDX_ROOT}/example/nppdx" -B build-nppdx-clang-ptx \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DNPPDX_BUILD_CUDA_EXAMPLES=OFF \
      -DNPPDX_BUILD_CLANG_PTX_EXAMPLES=ON \
      -DNPPDX_CLANG_PTX_COMPAT_INCLUDE_DIR=/path/to/cuda-compat/include \
      -DNPPDX_CUDA_ARCHITECTURES=90-real

cmake --build build-nppdx-clang-ptx --target nppdx_examples
ctest --test-dir build-nppdx-clang-ptx --output-on-failure
```

`NPPDX_CLANG_PTX_COMPAT_INCLUDE_DIR` must point to a directory containing a
`cuda_runtime.h` entrypoint. That header can come from the CUDA Toolkit include
directory. Projects with specialized Clang PTX needs can instead provide a
custom `cuda_runtime.h` with the required definitions. The CMake build
force-includes that header for device-only PTX compilation.

### Compatibility Entrypoint Requirements

For Clang PTX example builds, `cuda_runtime.h` must make available the CUDA
vocabulary that NPPDx headers and device examples use. When using a custom
`cuda_runtime.h`, make sure it provides these definitions, either directly or
through headers included by it:

- CUDA qualifiers and attributes: `__global__`, `__device__`, `__host__`,
  `__forceinline__`, `__shared__`, `__device_builtin__`, `__align__`, and
  `__inline__`.
- Launch built-ins: `blockIdx`, `blockDim`, `threadIdx`, and a `dim3`
  compatible type or constructor.
- Vector and handle types: `cudaTextureObject_t`, `cudaSurfaceObject_t`,
  `uchar2`, `ushort2`, `float2`, `int2`, `uint3`, `uchar4`, `ushort4`, `int4`,
  and `float4`.
- Constructors for vector values used by NPPDx code: `make_int2`,
  `make_uchar2`, `make_ushort2`, `make_float2`, `make_int4`, `make_uchar4`,
  `make_ushort4`, and `make_float4`.
- Texture and surface device APIs: `tex2D<T>(cudaTextureObject_t, float, float)`,
  `cudaSurfaceBoundaryMode`, and `surf2Dwrite` overloads for `unsigned char`,
  `unsigned short`, `uchar2`, `ushort2`, and `uchar4`.
- Atomic and control helpers: `atomicAdd`, `atomicCAS` for 16-bit, 32-bit, and
  64-bit unsigned integer storage, plus `__trap` for NPPDx device-side fatal
  errors.
- Math and utility device functions used by the headers: `floor`, `floorf`,
  `ceil`, `ceilf`, `trunc`, `truncf`, `fabs`, `fabsf`, `sqrtf`, `rintf`,
  `fminf`, `fmaxf`, `__saturatef`, `__sinf`, `__cosf`, `__expf`, `__powf`,
  `__logf`, and device `printf`.

## License

NPPDx is licensed under the Apache License, Version 2.0. See [LICENSE.txt](LICENSE.txt) for the full text.
