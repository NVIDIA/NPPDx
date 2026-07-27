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

#ifndef NPPDX_DETAIL_BACKEND_CONSTEXPR_MATH_HPP
#define NPPDX_DETAIL_BACKEND_CONSTEXPR_MATH_HPP

// =============================================================================
// Constexpr math for compile-time Q tensor computation
// =============================================================================
namespace nppdx {
    namespace detail {
        namespace backend {
            namespace constexpr_math {

                constexpr double PI = 3.14159265358979323846264338327950288;

                constexpr float const_abs(float x) {
                    return x < 0 ? -x : x;
                }

                // Constexpr power (for integer exponents)
                constexpr float pow(float base, int exp) {
                    if (exp == 0) {
                        return 1.0f;
                    }
                    if (exp < 0) {
                        return 1.0f / pow(base, -exp);
                    }
                    float result = 1.0f;
                    for (int i = 0; i < exp; ++i) {
                        result *= base;
                    }
                    return result;
                }

                constexpr float factorial(int n) {
                    float result = 1.0f;
                    for (int i = 2; i <= n; ++i) {
                        result *= i;
                    }
                    return result;
                }

                // Constexpr sin using Taylor series on [-pi/2, pi/2]
                constexpr float sin(float x) {
                    // Use double precision internally to reduce accumulated rounding error.
                    constexpr double HALF_PI = 0.5 * PI;
                    constexpr double TWO_PI  = 2.0 * PI;
                    double           xd      = static_cast<double>(x);

                    // Reduce to [-pi, pi]
                    while (xd > PI) {
                        xd -= TWO_PI;
                    }
                    while (xd < -PI) {
                        xd += TWO_PI;
                    }

                    // Reduce relative error using the symmetry of the sine function.
                    if (xd > HALF_PI) {
                        xd = PI - xd;
                    } else if (xd < -HALF_PI) {
                        xd = -PI - xd;
                    }
                    double term   = xd;
                    double result = term;
                    for (int n = 1; n < 10; ++n) {
                        double denom = static_cast<double>((2 * n) * (2 * n + 1));
                        term *= -(xd * xd) / denom;
                        result += term;
                    }
                    return static_cast<float>(result);
                }

                constexpr float sinc(float x) {
                    constexpr float SINC_FLOAT_ONE_THRESHOLD = 5.98019978497e-4f;
                    if (const_abs(x) < SINC_FLOAT_ONE_THRESHOLD) {
                        return 1.0f;
                    }
                    float pi_x = static_cast<float>(PI * static_cast<double>(x));
                    return sin(pi_x) / pi_x;
                }

                constexpr float lanczos3(float x) {
                    if (const_abs(x) >= 3.0f) {
                        return 0.0f;
                    }
                    return sinc(x) * sinc(x / 3.0f);
                }

                constexpr float bilinear(float x) {
                    float ax = const_abs(x);
                    return (ax < 1.0f) ? (1.0f - ax) : 0.0f;
                }

                // Constexpr bicubic kernel (Catmull-Rom)
                constexpr float bicubic(float x) {
                    float ax = const_abs(x);
                    if (ax < 1.0f) {
                        return (1.5f * ax - 2.5f) * ax * ax + 1.0f;
                    } else if (ax < 2.0f) {
                        return ((-0.5f * ax + 2.5f) * ax - 4.0f) * ax + 2.0f;
                    }
                    return 0.0f;
                }

            } // namespace constexpr_math

        } // namespace backend
    } // namespace detail
} // namespace nppdx
#endif // NPPDX_DETAIL_BACKEND_CONSTEXPR_MATH_HPP
