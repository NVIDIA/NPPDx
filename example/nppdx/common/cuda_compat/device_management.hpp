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

#ifndef NPPDX_EXAMPLE_COMMON_CUDA_COMPAT_DEVICE_MANAGEMENT_HPP
#define NPPDX_EXAMPLE_COMMON_CUDA_COMPAT_DEVICE_MANAGEMENT_HPP

#ifndef NPPDX_CLANG_CUDA_COMPAT
#    error "cuda_compat/device_management.hpp is only for the clang CUDA compatibility path"
#endif

#include "cuda_driver_dynlink.hpp"

#include <cstddef>
#include <fstream>
#include <string>
#include <utility>

namespace common {
    namespace cuda_driver {
        inline bool file_exists(const std::string& path) {
            std::ifstream file(path, std::ios::binary);
            return file.good();
        }

        inline std::string path_basename(const std::string& path) {
            const std::string::size_type pos = path.find_last_of("/\\");
            return pos == std::string::npos ? path : path.substr(pos + 1);
        }

        inline std::string path_dirname(const std::string& path) {
            const std::string::size_type pos = path.find_last_of("/\\");
            return pos == std::string::npos ? std::string(".") : path.substr(0, pos);
        }

        inline std::string resolve_module_path(const char* requested_path, const char* executable_path) {
            std::string requested = requested_path ? requested_path : "";
            if (file_exists(requested)) {
                return requested;
            }

            const std::string sibling =
                path_dirname(executable_path ? executable_path : "") + "/" + path_basename(requested);
            if (sibling != requested && file_exists(sibling)) {
                return sibling;
            }

            throw Error("PTX module file not found: " + requested + " (also tried " + sibling + ")", 3);
        }

        class ContextGuard
        {
        public:
            ContextGuard(const Driver& driver, Device device, unsigned int flags = 0): driver_(&driver) {
                driver_->check(driver_->fn().cuCtxCreate(&context_, flags, device), "cuCtxCreate_v2");
            }

            ContextGuard(const ContextGuard&)            = delete;
            ContextGuard& operator=(const ContextGuard&) = delete;

            ContextGuard(ContextGuard&& other) noexcept { move_from(other); }

            ContextGuard& operator=(ContextGuard&& other) noexcept {
                if (this != &other) {
                    reset();
                    move_from(other);
                }
                return *this;
            }

            ~ContextGuard() { reset(); }

            // Raw driver context handle, for interop with APIs (e.g. NVENC/NVDEC) that need it.
            Context get() const { return context_; }

        private:
            void reset() noexcept {
                if (context_) {
                    driver_->fn().cuCtxDestroy(context_);
                }
                context_ = nullptr;
            }

            void move_from(ContextGuard& other) noexcept {
                driver_  = other.driver_;
                context_ = std::exchange(other.context_, nullptr);
            }

            const Driver* driver_  = nullptr;
            Context       context_ = nullptr;
        };

        class ModuleGuard
        {
        public:
            ModuleGuard(const Driver& driver, const char* path): driver_(&driver) {
                driver_->check(driver_->fn().cuModuleLoad(&module_, path), "cuModuleLoad");
            }

            ModuleGuard(const ModuleGuard&)            = delete;
            ModuleGuard& operator=(const ModuleGuard&) = delete;

            ModuleGuard(ModuleGuard&& other) noexcept { move_from(other); }

            ModuleGuard& operator=(ModuleGuard&& other) noexcept {
                if (this != &other) {
                    reset();
                    move_from(other);
                }
                return *this;
            }

            ~ModuleGuard() { reset(); }

            Function function(const char* name) const {
                Function result = nullptr;
                driver_->check(driver_->fn().cuModuleGetFunction(&result, module_, name), "cuModuleGetFunction");
                return result;
            }

        private:
            void reset() noexcept {
                if (module_) {
                    driver_->fn().cuModuleUnload(module_);
                }
                module_ = nullptr;
            }

            void move_from(ModuleGuard& other) noexcept {
                driver_ = other.driver_;
                module_ = std::exchange(other.module_, nullptr);
            }

            const Driver* driver_ = nullptr;
            Module        module_ = nullptr;
        };

        class Stream
        {
        public:
            explicit Stream(const Driver& driver, unsigned int flags = 0): driver_(&driver) {
                driver_->check(driver_->fn().cuStreamCreate(&stream_, flags), "cuStreamCreate");
            }

            Stream(const Stream&)            = delete;
            Stream& operator=(const Stream&) = delete;

            Stream(Stream&& other) noexcept { move_from(other); }

            Stream& operator=(Stream&& other) noexcept {
                if (this != &other) {
                    reset();
                    move_from(other);
                }
                return *this;
            }

            ~Stream() { reset(); }

            operator StreamT() const { return stream_; }

            void synchronize() const {
                driver_->check(driver_->fn().cuStreamSynchronize(stream_), "cuStreamSynchronize");
            }

        private:
            void reset() noexcept {
                if (stream_) {
                    driver_->fn().cuStreamDestroy(stream_);
                }
                stream_ = nullptr;
            }

            void move_from(Stream& other) noexcept {
                driver_ = other.driver_;
                stream_ = std::exchange(other.stream_, nullptr);
            }

            const Driver* driver_ = nullptr;
            StreamT       stream_ = nullptr;
        };

        class DeviceBuffer
        {
        public:
            DeviceBuffer(const Driver& driver, std::size_t bytes): driver_(&driver), bytes_(bytes) {
                driver_->check(driver_->fn().cuMemAlloc(&d_buf, bytes_), "cuMemAlloc");
                allocated_ = true;
            }

            DeviceBuffer(const DeviceBuffer&)            = delete;
            DeviceBuffer& operator=(const DeviceBuffer&) = delete;

            DeviceBuffer(DeviceBuffer&& other) noexcept { move_from(other); }

            DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
                if (this != &other) {
                    reset();
                    move_from(other);
                }
                return *this;
            }

            ~DeviceBuffer() { reset(); }

            DevicePtr get() const { return d_buf; }

            void copy_from_host(const void* source, std::size_t bytes) const {
                check_size(bytes, "cuMemcpyHtoD");
                driver_->check(driver_->fn().cuMemcpyHtoD(d_buf, source, bytes), "cuMemcpyHtoD");
            }

            void copy_to_host(void* destination, std::size_t bytes) const {
                check_size(bytes, "cuMemcpyDtoH");
                driver_->check(driver_->fn().cuMemcpyDtoH(destination, d_buf, bytes), "cuMemcpyDtoH");
            }

        private:
            DevicePtr d_buf = 0;

            void check_size(std::size_t bytes, const char* call) const {
                if (bytes > bytes_) {
                    throw Error(std::string(call) + " exceeds device allocation size", 3);
                }
            }

            void reset() noexcept {
                if (allocated_) {
                    driver_->fn().cuMemFree(d_buf);
                }
                d_buf      = 0;
                bytes_     = 0;
                allocated_ = false;
            }

            void move_from(DeviceBuffer& other) noexcept {
                driver_    = other.driver_;
                d_buf      = std::exchange(other.d_buf, 0);
                bytes_     = std::exchange(other.bytes_, 0);
                allocated_ = std::exchange(other.allocated_, false);
            }

            const Driver* driver_    = nullptr;
            std::size_t   bytes_     = 0;
            bool          allocated_ = false;
        };

        struct LaunchConfig {
            unsigned int grid_x        = 1;
            unsigned int grid_y        = 1;
            unsigned int grid_z        = 1;
            unsigned int block_x       = 1;
            unsigned int block_y       = 1;
            unsigned int block_z       = 1;
            unsigned int shared_memory = 0;
            StreamT      stream        = nullptr;
        };

        inline void launch_kernel(const Driver& driver, Function function, const LaunchConfig& config, void** args) {
            driver.check(driver.fn().cuLaunchKernel(function, config.grid_x, config.grid_y, config.grid_z,
                                                    config.block_x, config.block_y, config.block_z,
                                                    config.shared_memory, config.stream, args, nullptr),
                         "cuLaunchKernel");
        }

        class KernelLauncher
        {
        public:
            KernelLauncher(const Driver& driver, Function function): driver_(&driver), function_(function) {}

            void set_max_dynamic_shared_memory(unsigned int bytes) const {
                driver_->set_max_dynamic_shared_memory(function_, bytes);
            }

            void launch(const LaunchConfig& config, void** args) const {
                launch_kernel(*driver_, function_, config, args);
            }

            Function get() const { return function_; }

        private:
            const Driver* driver_   = nullptr;
            Function      function_ = nullptr;
        };
    } // namespace cuda_driver

    inline unsigned int get_cuda_device_arch() {
        auto driver = cuda_driver::Driver::load_default();
        driver.init();
        return driver.device_arch(driver.device());
    }
} // namespace common

#endif // NPPDX_EXAMPLE_COMMON_CUDA_COMPAT_DEVICE_MANAGEMENT_HPP
