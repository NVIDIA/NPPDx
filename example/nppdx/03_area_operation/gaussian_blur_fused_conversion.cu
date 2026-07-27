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

// Fused Gaussian Blur + RGB to YUV422P Example - NPPDx vs NPP FilterGaussAdvanced comparison.
//
// Public single-radius demo. Runs the fused (ingest -> Gaussian blur -> RGB->YUV422P -> exgest) pipeline
// at one radius (default sigma = 2.0, kernel = 5x5) and compares against an NPP separable reference when
// NPP is available in the build.
//
// See gaussian_blur_fused_example.cuh for the shared driver and a description of the radius encoding,
// halo behavior, and output filenames.

#include "gaussian_blur_fused_example.cuh"

// Sigma the public demo runs at. RadiusTenths is exact: 20 => sigma = 2.0 => 5x5 separable FIR.
static constexpr int k_demo_radius_tenths = 20;

template<int Arch>
struct gaussian_blur_fused_conversion_functor {
    int operator()() const {
        return common::gaussian_blur_example::run_single_radius<Arch, k_demo_radius_tenths, /*Tabular=*/false>();
    }
};

int main() {
    return common::run_example_with_sm<gaussian_blur_fused_conversion_functor>();
}
