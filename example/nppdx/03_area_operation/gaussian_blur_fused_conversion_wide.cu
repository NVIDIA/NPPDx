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

// Wide-tail counterpart to gaussian_blur_fused_conversion.cu.
// Both examples use sigma=2.0. This one uses the 9-tap wide kernel (halo 4)
// and passes those same normalized coefficients to NPP's separable reference.

#include "gaussian_blur_fused_example.cuh"

namespace {

    constexpr int k_demo_radius_tenths = 20;

    struct gaussian_wide_fir_kernel_provider {
        static constexpr nppdx::gaussian_tail_width tail_width = nppdx::gaussian_tail_width::wide;

        template<int RadiusTenths>
        static void set(float (&kernel)[21]) {
            static_assert(RadiusTenths == 20, "The public wide NPP reference provides taps for sigma=2.0");
            // Independently precomputed exp(-x^2 / (2*sigma^2)), x=-4..4, sigma=2.0, then L1-normalized.
            constexpr float weights[] = {0.027630551f, 0.066282245f, 0.123831537f,
                                         0.180173823f, 0.204163689f, 0.180173823f,
                                         0.123831537f, 0.066282245f, 0.027630551f};
            for (unsigned int i = 0; i < 9; ++i) {
                kernel[i] = weights[i];
            }
        }
    };

} // namespace

template<int Arch>
struct gaussian_blur_fused_conversion_wide_functor {
    int operator()() const {
        return common::gaussian_blur_example::run_single_radius<
            Arch, k_demo_radius_tenths, /*Tabular=*/false, k_demo_radius_tenths, gaussian_wide_fir_kernel_provider,
            nppdx::gaussian_tail_width::wide>();
    }
};

int main() {
    return common::run_example_with_sm<gaussian_blur_fused_conversion_wide_functor>();
}
