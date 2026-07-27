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

#ifndef NPPDX_DETAIL_BACKEND_RESIZE_GEOMETRY_HPP
#define NPPDX_DETAIL_BACKEND_RESIZE_GEOMETRY_HPP

#include "nppdx/detail/decl.hpp"
#include "nppdx/types.hpp"
#include "nppdx/utils.hpp"

namespace nppdx::detail::backend::resize {

    NPPDX_DECL_CIHD unsigned int compute_resize_output_nominal_dim(unsigned int input_nominal,
                                                                   unsigned int input_samples,
                                                                   unsigned int output_samples) {
        const long long numerator =
            2ll * static_cast<long long>(input_nominal) * static_cast<long long>(output_samples) -
            static_cast<long long>(input_samples);
        const long long denominator = 2ll * static_cast<long long>(input_samples);
        const long long result      = dir_quot<-1>(numerator, denominator);
        return static_cast<unsigned int>(result > 0 ? result : 0);
    }

    NPPDX_DECL_CIHD uint2 compute_resize_output_nominal_tile(uint2 input_nominal, unsigned int input_samples,
                                                             unsigned int output_samples) {
        return {compute_resize_output_nominal_dim(input_nominal.x, input_samples, output_samples),
                compute_resize_output_nominal_dim(input_nominal.y, input_samples, output_samples)};
    }

} // namespace nppdx::detail::backend::resize

#endif // NPPDX_DETAIL_BACKEND_RESIZE_GEOMETRY_HPP
