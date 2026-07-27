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

#ifndef NPPDX_EXAMPLE_COMMON_NUMERIC_HPP
#define NPPDX_EXAMPLE_COMMON_NUMERIC_HPP

#ifndef NPPDX_EXAMPLE_NVRTC

#    include <vector>
#    include <algorithm>

#    include <nppdx.hpp>

namespace common {

    // Scalar conversion
    template<typename Tout, typename Tin>
    Tout convert(const Tin& input) {
        return static_cast<Tout>(input);
    }

    // Vector conversion
    template<typename Tin, typename Tout>
    std::vector<Tout> convert(const std::vector<Tin>& input) {
        std::vector<Tout> output(input.size());
        std::transform(input.begin(), input.end(), output.begin(),
                       [](const Tin& val) { return static_cast<Tout>(val); });
        return output;
    }

} // namespace common

#endif // NPPDX_EXAMPLE_NVRTC

#endif // NPPDX_EXAMPLE_COMMON_NUMERIC_HPP
