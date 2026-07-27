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

#ifndef CUDA_DRIVER_DYNLINK_HPP
#define CUDA_DRIVER_DYNLINK_HPP

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <dlfcn.h>

namespace common {
    namespace cuda_driver {
        using Device         = int;
        using DevicePtr      = unsigned long long;
        using Result         = int;
        using Context        = struct CUctx_st*;
        using Module         = struct CUmod_st*;
        using Function       = struct CUfunc_st*;
        using StreamT        = struct CUstream_st*;
        using Array          = struct CUarray_st*;
        using MipmappedArray = struct CUmipmappedArray_st*;
        using TexObject      = unsigned long long;
        using SurfObject     = unsigned long long;

        constexpr Result success = 0;

        enum ArrayFormat
        {
            array_format_unsigned_int8  = 0x01,
            array_format_unsigned_int16 = 0x02
        };

        enum MemoryType
        {
            memory_type_host   = 0x01,
            memory_type_device = 0x02,
            memory_type_array  = 0x03
        };

        enum ResourceType
        {
            resource_type_array = 0x00
        };

        enum AddressMode
        {
            address_mode_clamp = 1
        };

        enum FilterMode
        {
            filter_mode_point = 0
        };

        constexpr unsigned int texture_read_as_integer  = 0x01;
        constexpr unsigned int array_surface_load_store = 0x02;

        struct Memcpy2D {
            std::size_t srcXInBytes = 0;
            std::size_t srcY        = 0;

            MemoryType  srcMemoryType = memory_type_host;
            const void* srcHost       = nullptr;
            DevicePtr   srcDevice     = 0;
            Array       srcArray      = nullptr;
            std::size_t srcPitch      = 0;

            std::size_t dstXInBytes = 0;
            std::size_t dstY        = 0;

            MemoryType  dstMemoryType = memory_type_host;
            void*       dstHost       = nullptr;
            DevicePtr   dstDevice     = 0;
            Array       dstArray      = nullptr;
            std::size_t dstPitch      = 0;

            std::size_t WidthInBytes = 0;
            std::size_t Height       = 0;
        };

        struct Array3DDescriptor {
            std::size_t  Width       = 0;
            std::size_t  Height      = 0;
            std::size_t  Depth       = 0;
            ArrayFormat  Format      = array_format_unsigned_int8;
            unsigned int NumChannels = 0;
            unsigned int Flags       = 0;
        };

        struct ResourceDesc {
            ResourceType resType = resource_type_array;

            union {
                struct {
                    Array hArray;
                } array;
                struct {
                    MipmappedArray hMipmappedArray;
                } mipmap;
                struct {
                    DevicePtr    devPtr;
                    ArrayFormat  format;
                    unsigned int numChannels;
                    std::size_t  sizeInBytes;
                } linear;
                struct {
                    DevicePtr    devPtr;
                    ArrayFormat  format;
                    unsigned int numChannels;
                    std::size_t  width;
                    std::size_t  height;
                    std::size_t  pitchInBytes;
                } pitch2D;
                struct {
                    int reserved[32];
                } reserved;
            } res {};

            unsigned int flags = 0;
        };

        struct TextureDesc {
            AddressMode  addressMode[3]      = {address_mode_clamp, address_mode_clamp, address_mode_clamp};
            FilterMode   filterMode          = filter_mode_point;
            unsigned int flags               = texture_read_as_integer;
            unsigned int maxAnisotropy       = 0;
            FilterMode   mipmapFilterMode    = filter_mode_point;
            float        mipmapLevelBias     = 0.0f;
            float        minMipmapLevelClamp = 0.0f;
            float        maxMipmapLevelClamp = 0.0f;
            float        borderColor[4]      = {0.0f, 0.0f, 0.0f, 0.0f};
            int          reserved[12]        = {};
        };

        using tcuInit               = Result (*)(unsigned int);
        using tcuDeviceGet          = Result (*)(Device*, int);
        using tcuDeviceGetAttribute = Result (*)(int*, int, Device);
        using tcuCtxCreate          = Result (*)(Context*, unsigned int, Device);
        using tcuCtxDestroy         = Result (*)(Context);
        using tcuModuleLoad         = Result (*)(Module*, const char*);
        using tcuModuleUnload       = Result (*)(Module);
        using tcuModuleGetFunction  = Result (*)(Function*, Module, const char*);
        using tcuMemAlloc           = Result (*)(DevicePtr*, std::size_t);
        using tcuMemFree            = Result (*)(DevicePtr);
        using tcuMemcpyHtoD         = Result (*)(DevicePtr, const void*, std::size_t);
        using tcuMemcpyDtoH         = Result (*)(void*, DevicePtr, std::size_t);
        using tcuMemcpy2D           = Result (*)(const Memcpy2D*);
        using tcuArray3DCreate      = Result (*)(Array*, const Array3DDescriptor*);
        using tcuArrayDestroy       = Result (*)(Array);
        using tcuTexObjectCreate    = Result (*)(TexObject*, const ResourceDesc*, const TextureDesc*, const void*);
        using tcuTexObjectDestroy   = Result (*)(TexObject);
        using tcuSurfObjectCreate   = Result (*)(SurfObject*, const ResourceDesc*);
        using tcuSurfObjectDestroy  = Result (*)(SurfObject);
        using tcuLaunchKernel       = Result (*)(Function, unsigned int, unsigned int, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int, StreamT, void**, void**);
        using tcuFuncSetAttribute   = Result (*)(Function, int, int);
        using tcuCtxSynchronize     = Result (*)();
        using tcuStreamCreate       = Result (*)(StreamT*, unsigned int);
        using tcuStreamDestroy      = Result (*)(StreamT);
        using tcuStreamSynchronize  = Result (*)(StreamT);
        using tcuGetErrorName       = Result (*)(Result, const char**);
        using tcuGetErrorString     = Result (*)(Result, const char**);

#define NPPDX_FOR_CUDA_DRIVER_FUNCTIONS(X)                                 \
    X(tcuInit, cuInit, "cuInit")                                           \
    X(tcuDeviceGet, cuDeviceGet, "cuDeviceGet")                            \
    X(tcuDeviceGetAttribute, cuDeviceGetAttribute, "cuDeviceGetAttribute") \
    X(tcuCtxCreate, cuCtxCreate, "cuCtxCreate_v2")                         \
    X(tcuCtxDestroy, cuCtxDestroy, "cuCtxDestroy_v2")                      \
    X(tcuModuleLoad, cuModuleLoad, "cuModuleLoad")                         \
    X(tcuModuleUnload, cuModuleUnload, "cuModuleUnload")                   \
    X(tcuModuleGetFunction, cuModuleGetFunction, "cuModuleGetFunction")    \
    X(tcuMemAlloc, cuMemAlloc, "cuMemAlloc_v2")                            \
    X(tcuMemFree, cuMemFree, "cuMemFree_v2")                               \
    X(tcuMemcpyHtoD, cuMemcpyHtoD, "cuMemcpyHtoD_v2")                      \
    X(tcuMemcpyDtoH, cuMemcpyDtoH, "cuMemcpyDtoH_v2")                      \
    X(tcuMemcpy2D, cuMemcpy2D, "cuMemcpy2D_v2")                            \
    X(tcuArray3DCreate, cuArray3DCreate, "cuArray3DCreate_v2")             \
    X(tcuArrayDestroy, cuArrayDestroy, "cuArrayDestroy")                   \
    X(tcuTexObjectCreate, cuTexObjectCreate, "cuTexObjectCreate")          \
    X(tcuTexObjectDestroy, cuTexObjectDestroy, "cuTexObjectDestroy")       \
    X(tcuSurfObjectCreate, cuSurfObjectCreate, "cuSurfObjectCreate")       \
    X(tcuSurfObjectDestroy, cuSurfObjectDestroy, "cuSurfObjectDestroy")    \
    X(tcuLaunchKernel, cuLaunchKernel, "cuLaunchKernel")                   \
    X(tcuFuncSetAttribute, cuFuncSetAttribute, "cuFuncSetAttribute")       \
    X(tcuCtxSynchronize, cuCtxSynchronize, "cuCtxSynchronize")             \
    X(tcuStreamCreate, cuStreamCreate, "cuStreamCreate")                   \
    X(tcuStreamDestroy, cuStreamDestroy, "cuStreamDestroy_v2")             \
    X(tcuStreamSynchronize, cuStreamSynchronize, "cuStreamSynchronize")    \
    X(tcuGetErrorName, cuGetErrorName, "cuGetErrorName")                   \
    X(tcuGetErrorString, cuGetErrorString, "cuGetErrorString")

        struct Functions {
#define NPPDX_CUDA_DRIVER_DECLARE_FUNCTION(type, member, symbol) type member = nullptr;
            NPPDX_FOR_CUDA_DRIVER_FUNCTIONS(NPPDX_CUDA_DRIVER_DECLARE_FUNCTION)
#undef NPPDX_CUDA_DRIVER_DECLARE_FUNCTION
        };

        class Error: public std::runtime_error
        {
        public:
            Error(std::string message, int code): std::runtime_error(std::move(message)), code_(code) {}

            int code() const noexcept { return code_; }

        private:
            int code_;
        };

        class Driver
        {
        public:
            Driver()                         = default;
            Driver(const Driver&)            = delete;
            Driver& operator=(const Driver&) = delete;

            Driver(Driver&& other) noexcept { move_from(other); }

            Driver& operator=(Driver&& other) noexcept {
                if (this != &other) {
                    close();
                    move_from(other);
                }
                return *this;
            }

            ~Driver() { close(); }

            static Driver load_default() {
                Driver            driver;
                std::string       errors;
                const char* const library_names[] = {"libcuda.so.1", "libcuda.so"};

                for (const char* library_name : library_names) {
                    if (driver.try_load(library_name, errors)) {
                        return driver;
                    }
                }

                throw Error("unable to load libcuda.so: " + errors, 4);
            }

            const Functions& fn() const { return functions_; }

            void check(Result result, const char* call) const {
                if (result == success) {
                    return;
                }

                throw Error(describe_error(result, call), 3);
            }

            void init() const { check(functions_.cuInit(0), "cuInit"); }

            Device device(int ordinal = 0) const {
                Device result = 0;
                check(functions_.cuDeviceGet(&result, ordinal), "cuDeviceGet");
                return result;
            }

            int device_attribute(Device device, int attribute) const {
                int result = 0;
                check(functions_.cuDeviceGetAttribute(&result, attribute, device), "cuDeviceGetAttribute");
                return result;
            }

            unsigned int device_arch(Device device) const {
                constexpr int compute_capability_major = 75;
                constexpr int compute_capability_minor = 76;
                const int     major                    = device_attribute(device, compute_capability_major);
                const int     minor                    = device_attribute(device, compute_capability_minor);
                return static_cast<unsigned int>(major) * 100u + static_cast<unsigned int>(minor) * 10u;
            }

            void synchronize() const { check(functions_.cuCtxSynchronize(), "cuCtxSynchronize"); }

            void set_max_dynamic_shared_memory(Function function, unsigned int bytes) const {
                constexpr int max_dynamic_shared_size_bytes = 8;
                check(functions_.cuFuncSetAttribute(function, max_dynamic_shared_size_bytes, int(bytes)),
                      "cuFuncSetAttribute(max dynamic shared memory)");
            }

        private:
            template<typename Fn>
            Fn load_symbol(const char* name) const {
                dlerror();
                void*       symbol = dlsym(library_, name);
                const char* error  = dlerror();
                if (error || !symbol) {
                    throw Error(std::string("missing Driver API symbol: ") + name, 4);
                }
                return reinterpret_cast<Fn>(symbol);
            }

            bool try_load(const char* library_name, std::string& errors) {
                library_ = dlopen(library_name, RTLD_NOW | RTLD_LOCAL);
                if (!library_) {
                    if (!errors.empty()) {
                        errors += "; ";
                    }
                    const char* error = dlerror();
                    errors += library_name;
                    errors += ": ";
                    errors += error ? error : "unknown error";
                    return false;
                }

#define NPPDX_CUDA_DRIVER_LOAD_FUNCTION(type, member, symbol) functions_.member = load_symbol<type>(symbol);
                NPPDX_FOR_CUDA_DRIVER_FUNCTIONS(NPPDX_CUDA_DRIVER_LOAD_FUNCTION)
#undef NPPDX_CUDA_DRIVER_LOAD_FUNCTION

                return true;
            }

            std::string describe_error(Result result, const char* call) const {
                const char* name = nullptr;
                const char* text = nullptr;
                if (functions_.cuGetErrorName) {
                    functions_.cuGetErrorName(result, &name);
                }
                if (functions_.cuGetErrorString) {
                    functions_.cuGetErrorString(result, &text);
                }

                std::ostringstream message;
                message << call << " failed";
                if (name) {
                    message << " (" << name << ')';
                }
                if (text) {
                    message << ": " << text;
                }
                return message.str();
            }

            void close() noexcept {
                if (library_) {
                    dlclose(library_);
                }
                library_   = nullptr;
                functions_ = {};
            }

            void move_from(Driver& other) noexcept {
                library_         = std::exchange(other.library_, nullptr);
                functions_       = other.functions_;
                other.functions_ = {};
            }

            void*     library_ = nullptr;
            Functions functions_ {};
        };
    } // namespace cuda_driver
} // namespace common

#endif // CUDA_DRIVER_DYNLINK_HPP
