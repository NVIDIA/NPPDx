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

// CPU golden resize used by public examples (e.g. fused_resize). Logic matches test/common/cpu/test_cpu_reference.hpp
// separable resize<J,K,Method,...> for pass/fail vs NPPDx without pulling test targets into example CMake.
#ifndef NPPDX_EXAMPLE_COMMON_CPU_RESIZE_REFERENCE_HPP
#define NPPDX_EXAMPLE_COMMON_CPU_RESIZE_REFERENCE_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <nppdx/operators/function.hpp>

namespace nppdx::example::cpu_reference {

inline constexpr float kPi = 3.14159265358979323846f;

template<typename T>
inline T round_to(float val) {
    if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        return static_cast<T>(val + 0.5f);
    }
}

inline float lanczos_kernel(float x, int a) {
    if (x == 0.0f)
        return 1.0f;
    if (std::abs(x) >= static_cast<float>(a))
        return 0.0f;
    float pi_x = kPi * x;
    return (std::sin(pi_x) / pi_x) * (std::sin(pi_x / a) / (pi_x / a));
}

inline float bicubic_kernel(float x) {
    x = std::abs(x);
    if (x < 1.0f) {
        return (1.5f * x - 2.5f) * x * x + 1.0f;
    } else if (x < 2.0f) {
        return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    }
    return 0.0f;
}

template<unsigned int J, unsigned int K, nppdx::interpolation_method Method, unsigned int Channels = 3,
         bool UseRadialLanczos2D = false, typename T = float>
inline std::vector<T> resize(const std::vector<T>& input, int width, int height) {
    const unsigned int out_w = width * K / J;
    const unsigned int out_h = height * K / J;

    std::vector<T> output(out_w * out_h * Channels);

    constexpr unsigned int mu_num = []() constexpr {
        if constexpr (Method == nppdx::interpolation_method::nearest) {
            return 1U;
        } else if constexpr (Method == nppdx::interpolation_method::bilinear) {
            return 1U;
        } else if constexpr (Method == nppdx::interpolation_method::bicubic) {
            return 2U;
        } else {
            return 3U;
        }
    }
    ();
    constexpr unsigned int mu_den          = (Method == nppdx::interpolation_method::nearest) ? 2U : 1U;
    constexpr unsigned int radius_num      = (J > K) ? (mu_num * J) : mu_num;
    constexpr unsigned int radius_den      = (J > K) ? (mu_den * K) : mu_den;
    constexpr unsigned int kernel_span_raw = (2 * radius_num + radius_den - 1) / radius_den;
    constexpr unsigned int kernel_span =
        (Method == nppdx::interpolation_method::nearest) ? 1U : (kernel_span_raw + (kernel_span_raw % 2));

    auto compute_input_start_index = [](int n_o, float theta, unsigned int span) -> int {
        int numerator = 2 * n_o * static_cast<int>(J) + static_cast<int>(J) - static_cast<int>(K);
        int n_c       = (numerator >= 0) ? (numerator / (2 * static_cast<int>(K)))
                                         : ((numerator - 2 * static_cast<int>(K) + 1) / (2 * static_cast<int>(K)));
        int delta = (span % 2 == 0) ? 2 : ((theta < 0.5f) ? 1 : 3);
        return n_c - (static_cast<int>(span) - delta) / 2;
    };

    auto compute_kernel_position = [](int n_o, int j, unsigned int span) -> float {
        float val   = (static_cast<float>(n_o) + 0.5f) * (static_cast<float>(J) / static_cast<float>(K)) - 0.5f;
        float theta = val - std::floor(val);
        int   delta = (span % 2 == 0) ? 2 : ((theta < 0.5f) ? 1 : 3);
        float L     = static_cast<float>(static_cast<int>(span) - delta) / 2.0f;
        return theta + L - static_cast<float>(j);
    };

    [[maybe_unused]] auto evaluate_kernel = [](float x) -> float {
        if constexpr (Method == nppdx::interpolation_method::nearest) {
            return (std::abs(x) <= 0.5f) ? 1.0f : 0.0f;
        } else if constexpr (Method == nppdx::interpolation_method::bilinear) {
            float ax = std::abs(x);
            return (ax < 1.0f) ? (1.0f - ax) : 0.0f;
        } else if constexpr (Method == nppdx::interpolation_method::bicubic) {
            return bicubic_kernel(x);
        } else {
            return lanczos_kernel(x, 3);
        }
    };

    for (unsigned int out_y = 0; out_y < out_h; ++out_y) {
        for (unsigned int out_x = 0; out_x < out_w; ++out_x) {
            for (unsigned int c = 0; c < Channels; ++c) {
                float result     = 0.0f;
                float weight_sum = 0.0f;

                float val_x =
                    (static_cast<float>(out_x) + 0.5f) * (static_cast<float>(J) / static_cast<float>(K)) - 0.5f;
                float val_y =
                    (static_cast<float>(out_y) + 0.5f) * (static_cast<float>(J) / static_cast<float>(K)) - 0.5f;
                float theta_x = val_x - std::floor(val_x);
                float theta_y = val_y - std::floor(val_y);

                const int n_ix_start = compute_input_start_index(static_cast<int>(out_x), theta_x, kernel_span);
                const int n_iy_start = compute_input_start_index(static_cast<int>(out_y), theta_y, kernel_span);

                for (unsigned int jy = 0; jy < kernel_span; ++jy) {
                    const float y_pos = compute_kernel_position(static_cast<int>(out_y), static_cast<int>(jy), kernel_span);
                    int         sy    = std::clamp(n_iy_start + static_cast<int>(jy), 0, static_cast<int>(height) - 1);

                    for (unsigned int jx = 0; jx < kernel_span; ++jx) {
                        const float x_pos = compute_kernel_position(static_cast<int>(out_x), static_cast<int>(jx), kernel_span);
                        int         sx    = std::clamp(n_ix_start + static_cast<int>(jx), 0, static_cast<int>(width) - 1);
                        float       w     = 0.0f;
                        if constexpr (Method == nppdx::interpolation_method::lanczos3 && UseRadialLanczos2D) {
                            w = lanczos_kernel(std::sqrt(x_pos * x_pos + y_pos * y_pos), 3);
                        } else {
                            float wx = evaluate_kernel(x_pos);
                            float wy = evaluate_kernel(y_pos);
                            w        = wx * wy;
                        }
                        if (w == 0.0f) {
                            continue;
                        }
                        result += w * static_cast<float>(input[(sy * width + sx) * Channels + c]);
                        weight_sum += w;
                    }
                }

                if (weight_sum > 1e-6f) {
                    result /= weight_sum;
                }

                output[(out_y * out_w + out_x) * Channels + c] = round_to<T>(result);
            }
        }
    }
    return output;
}

} // namespace nppdx::example::cpu_reference

#endif // NPPDX_EXAMPLE_COMMON_CPU_RESIZE_REFERENCE_HPP
