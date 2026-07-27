/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NPPDX_EXAMPLE_COMMON_RANDOM_HPP
#define NPPDX_EXAMPLE_COMMON_RANDOM_HPP

#include <nppdx.hpp>
#include <vector>
#include <memory>


namespace common {

    template<class T>
    std::tuple<float, float> get_random_range() {
        // For uint8_t (image data), use full 0-255 range
        if constexpr (std::is_same_v<T, uint8_t>) {
            return std::make_tuple(0.0f, 255.0f);
        } else if constexpr ((sizeof(T) <= 2)) {
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ == 7)
            return std::make_tuple(-0.01f,
                                   0.01f); // essential to avoid overflow with (e5m2, e4m3, fp32) gcc-7 + ctk 12.1
#else
            return std::make_tuple(-0.5f, 0.5f);
#endif
        } else if constexpr (sizeof(T) < 4) {
            return std::make_tuple(-0.5f, 0.5f);
        } else {
            return std::make_tuple(-10.f, 10.f);
        }
    }

    template<typename T, bool Host = true>
    std::vector<T> generate_random_data(const float min, const float max, const size_t size) {
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        std::vector<T> data(size);

        std::generate(data.begin(), data.end(), [&]() { return convert<T>(dist(gen)); });

        return data;
    }

    template<typename T, bool Host = true>
    std::vector<T> generate_random_data(const size_t size) {
        const auto [min, max] = get_random_range<T>();
        return generate_random_data<T, Host>(min, max, size);
    }
} // namespace common

#endif // NPPDX_EXAMPLE_COMMON_RANDOM_HPP
