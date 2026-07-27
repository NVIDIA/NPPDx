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

#ifndef NPPDX_EXAMPLE_COMMON_MEASURE_HPP
#define NPPDX_EXAMPLE_COMMON_MEASURE_HPP

#include "macros.hpp"
#include <chrono>
#include <algorithm>
#include <limits>

namespace common {

    double get_gflops(unsigned int m, unsigned int n, unsigned int k) {
        // We assume elementwise single flop per element
        return m + n + k;
    }

    struct measure {
        template<typename Kernel>
        static void run_warmup(Kernel&& kernel, const unsigned int warm_up_runs, cudaStream_t stream) {
            for (unsigned int i = 0; i < warm_up_runs; i++) {
                kernel(stream);
            }
            CUDA_CHECK_AND_EXIT(cudaGetLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        }

        // Total ms for `runs` consecutive kernel(stream) invocations between CUDA events (no warmup).
        template<typename Kernel>
        static float timed_sequence_ms(Kernel&& kernel, const unsigned int runs, cudaStream_t stream) {
            cudaEvent_t startEvent, stopEvent;
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));

            CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent, stream));
            for (unsigned int i = 0; i < runs; i++) {
                kernel(stream);
            }
            CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent, stream));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            float time;
            CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
            return time;
        }

        // Returns execution time in ms (total for all `runs` kernels in one timed span).
        template<typename Kernel>
        static float execution(Kernel&& kernel, const unsigned int warm_up_runs, const unsigned int runs,
                               cudaStream_t stream) {
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
            run_warmup(kernel, warm_up_runs, stream);
            return timed_sequence_ms(kernel, runs, stream);
        }

        // After optional warmup (often 0 when caller takes min over trials): run `trials` independent CUDA timings.
        // Each trial times `runs_per_trial` consecutive launches and divides by runs_per_trial → ms per launch.
        // Returns the minimum (reduces scheduler / multi-strip / host jitter vs a single batched sample mean).
        template<typename Kernel>
        static float execution_min_per_launch_ms(Kernel&& kernel, const unsigned int warm_up_runs,
                                                 const unsigned int trials, const unsigned int runs_per_trial,
                                                 cudaStream_t stream) {
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
            run_warmup(kernel, warm_up_runs, stream);

            float min_per_launch_ms = std::numeric_limits<float>::max();
            for (unsigned int t = 0; t < trials; ++t) {
                const float batch_ms = timed_sequence_ms(kernel, runs_per_trial, stream);
                const float per_launch_ms =
                    batch_ms / static_cast<float>(runs_per_trial > 0u ? runs_per_trial : 1u);
                min_per_launch_ms = std::min(min_per_launch_ms, per_launch_ms);
            }
            return min_per_launch_ms;
        }
    };

    // Returns execution time in ms
    template<typename Kernel>
    float measure_execution_ms(Kernel&& kernel, const unsigned int warm_up_runs, const unsigned int runs,
                               cudaStream_t stream) {
        cudaEvent_t startEvent, stopEvent;
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        for (size_t i = 0; i < warm_up_runs; i++) {
            kernel(stream);
        }
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent, stream));
        for (size_t i = 0; i < runs; i++) {
            kernel(stream);
        }
        CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent, stream));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        float time;
        CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
        CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
        CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
        return time;
    }

    template<typename Function>
    float measure_host_ms(Function&& kernel) {
        auto t1 = std::chrono::high_resolution_clock::now();
        kernel();
        auto                                     t2       = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> ms_float = t2 - t1;
        return ms_float.count();
    }

    template<class T>
    struct nppdx_results {
        std::vector<T> output;
        float          avg_time_in_ms;
    };
} // namespace common

#endif
