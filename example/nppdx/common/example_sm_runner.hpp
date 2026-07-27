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

#ifndef NPPDX_EXAMPLE_COMMON_EXAMPLE_SM_RUNNER_HPP
#define NPPDX_EXAMPLE_COMMON_EXAMPLE_SM_RUNNER_HPP

#include <sstream>

#include <nppdx.hpp>
#include <nppdx/utils.hpp>
#include <stdexcept>

#include "cudart.hpp"


namespace common {
    struct nppdx_enable_example_sm {
#if defined(NPPDX_EXAMPLE_ENABLE_SM_70)
        static constexpr bool sm_70 = true;
#else
        static constexpr bool sm_70  = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_72)
        static constexpr bool sm_72 = true;
#else
        static constexpr bool sm_72  = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_75)
        static constexpr bool sm_75 = true;
#else
        static constexpr bool sm_75  = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_80)
        static constexpr bool sm_80 = true;
#else
        static constexpr bool sm_80  = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_86)
        static constexpr bool sm_86 = true;
#else
        static constexpr bool sm_86  = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_87)
        static constexpr bool sm_87 = true;
#else
        static constexpr bool sm_87  = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_89)
        static constexpr bool sm_89 = true;
#else
        static constexpr bool sm_89  = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_90)
        static constexpr bool sm_90 = true;
#else
        static constexpr bool sm_90  = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_100)
        static constexpr bool sm_100 = true;
#else
        static constexpr bool sm_100 = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_101)
        static constexpr bool sm_101 = true;
#else
        static constexpr bool sm_101 = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_103)
        static constexpr bool sm_103 = true;
#else
        static constexpr bool sm_103 = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_110)
        static constexpr bool sm_110 = true;
#else
        static constexpr bool sm_110 = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_120)
        static constexpr bool sm_120 = true;
#else
        static constexpr bool sm_120 = false;
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_121)
        static constexpr bool sm_121 = true;
#else
        static constexpr bool sm_121 = false;
#endif
    };

    template<class EnableSM>
    void print_supported_sm(const unsigned int cuda_device_arch) {
        auto stream = std::stringstream();

        if constexpr (EnableSM::sm_70) {
            stream << "- SM 700" << std::endl;
        }
        if constexpr (EnableSM::sm_72) {
            stream << "- SM 720" << std::endl;
        }
        if constexpr (EnableSM::sm_75) {
            stream << "- SM 750" << std::endl;
        }
        if constexpr (EnableSM::sm_80) {
            stream << "- SM 800" << std::endl;
        }
        if constexpr (EnableSM::sm_86) {
            stream << "- SM 860" << std::endl;
        }
        if constexpr (EnableSM::sm_87) {
            stream << "- SM 870" << std::endl;
        }
        if constexpr (EnableSM::sm_89) {
            stream << "- SM 890" << std::endl;
        }
        if constexpr (EnableSM::sm_90) {
            stream << "- SM 900" << std::endl;
        }
        if constexpr (EnableSM::sm_100) {
            stream << "- SM 1000" << std::endl;
        }
        if constexpr (EnableSM::sm_101) {
            stream << "- SM 1010" << std::endl;
        }
        if constexpr (EnableSM::sm_103) {
            stream << "- SM 1030" << std::endl;
        }
        if constexpr (EnableSM::sm_110) {
            stream << "- SM 1100" << std::endl;
        }
        if constexpr (EnableSM::sm_120) {
            stream << "- SM 1200" << std::endl;
        }
        if constexpr (EnableSM::sm_121) {
            stream << "- SM 1210" << std::endl;
        }

        std::cerr << "Functor failed to run on any supported SM, supported SMs: \n"
                  << stream.str() << std::endl
                  << "this device architecture: " << cuda_device_arch << std::endl;
    }
    // This function enables creating architecture agnostic examples
    // and functions while avoid compilation overhead, by compiling
    // only the enabled branches and then based on runtime CUDA compute
    // capability dispatching with appropriate argument.
    //
    // Functor is example function which takes static integer type as
    // its argument. Then the example can read this value and use it
    // for its SM<Val>() operator.
    template<template<int> class Functor, typename... Args>
    inline int run_example_with_sm(Args&&... args) {
        // Get CUDA device compute capability
        const auto cuda_device_arch = common::get_cuda_device_arch();

        switch (cuda_device_arch) {
// All SM supported by NPPDx
#ifdef NPPDX_EXAMPLE_ENABLE_SM_70
            case 700: return Functor<700>()(std::forward<Args>(args)...);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_72
            case 720: return Functor<720>()(std::forward<Args>(args)...);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_75
            case 750: return Functor<750>()(std::forward<Args>(args)...);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_80
            case 800: return Functor<800>()(std::forward<Args>(args)...);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_86
            case 860: return Functor<860>()(std::forward<Args>(args)...);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_87
            case 870: return Functor<870>()(std::forward<Args>(args)...);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_89
            case 890: return Functor<890>()(std::forward<Args>(args)...);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_90
            case 900: return Functor<900>()(std::forward<Args>(args)...);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_100
            case 1000: return Functor<1000>()(std::forward<Args>(args)...);
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_101) && (CUDA_VERSION < 13000)
            case 1010: return Functor<1010>()(std::forward<Args>(args)...);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_103
            case 1030: return Functor<1030>()(std::forward<Args>(args)...);
#endif
#if defined(NPPDX_EXAMPLE_ENABLE_SM_110) && (CUDA_VERSION >= 13000)
            case 1100: return Functor<1100>()(std::forward<Args>(args)...);
#endif
#if defined(NPPDX_EXAMPLE_ENABLE_SM_120) && (CUDA_VERSION >= 13000)
            case 1200: return Functor<1200>()(std::forward<Args>(args)...);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_121
            case 1210: return Functor<1210>()(std::forward<Args>(args)...);
#endif

            default: {
                print_supported_sm<nppdx_enable_example_sm>(cuda_device_arch);
                // Fail
                return 1;
            }
        }
    }

    // Variant of the above using CVal machinery.
    template<typename FunctorT, typename ArchCarrierT, typename... ArgsT>
    inline auto dispatch_sm_arch(FunctorT&& functor, ArchCarrierT archCarrier, ArgsT&&... args) {
        static_assert(nppdx::has_carried_type<int, ArchCarrierT>);

#define NPPDX_PRIVATE_CASE_BODY(arch) std::forward<FunctorT>(functor)(arch, std::forward<ArgsT>(args)...)

        if constexpr (nppdx::is_cval<ArchCarrierT>) {
            return NPPDX_PRIVATE_CASE_BODY(archCarrier);
        } else {
#define NPPDX_PRIVATE_CASE(arch) \
    case arch: return NPPDX_PRIVATE_CASE_BODY(nppdx::C<arch>)
            switch (archCarrier) {
// All SM supported by NPPDx
#ifdef NPPDX_EXAMPLE_ENABLE_SM_70
                NPPDX_PRIVATE_CASE(700);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_72
                NPPDX_PRIVATE_CASE(720);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_75
                NPPDX_PRIVATE_CASE(750);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_80
                NPPDX_PRIVATE_CASE(800);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_86
                NPPDX_PRIVATE_CASE(860);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_87
                NPPDX_PRIVATE_CASE(870);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_89
                NPPDX_PRIVATE_CASE(890);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_90
                NPPDX_PRIVATE_CASE(900);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_100
                NPPDX_PRIVATE_CASE(1000);
#endif

#if defined(NPPDX_EXAMPLE_ENABLE_SM_101) && (CUDA_VERSION < 13000)
                NPPDX_PRIVATE_CASE(1010);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_103
                NPPDX_PRIVATE_CASE(1030);
#endif
#if defined(NPPDX_EXAMPLE_ENABLE_SM_110) && (CUDA_VERSION >= 13000)
                NPPDX_PRIVATE_CASE(1100);
#endif
#if defined(NPPDX_EXAMPLE_ENABLE_SM_120) && (CUDA_VERSION >= 13000)
                NPPDX_PRIVATE_CASE(1200);
#endif
#ifdef NPPDX_EXAMPLE_ENABLE_SM_121
                NPPDX_PRIVATE_CASE(1210);
#endif
                default: {
                    print_supported_sm<nppdx_enable_example_sm>(archCarrier);
                    // Fail
                    throw std::runtime_error("Functor failed to run on any supported SM");
                }
#undef NPPDX_PRIVATE_CASE
            }
        }
#undef NPPDX_PRIVATE_CASE_BODY
    }
} // namespace common

#endif // NPPDX_EXAMPLE_COMMON_EXAMPLE_SM_RUNNER_HPP
