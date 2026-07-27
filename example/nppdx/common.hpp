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

#ifndef NPPDX_EXAMPLE_COMMON_HPP_
#define NPPDX_EXAMPLE_COMMON_HPP_

#include <algorithm>
#include <type_traits>
#include <vector>
#include <random>
#include <complex>

#include <cuda_runtime_api.h>


#ifndef NPPDX_EXAMPLE_NVRTC
#    include <nppdx.hpp>
#endif

#include "common/macros.hpp"
#include "common/cudart.hpp"
#include "common/error_checking.hpp"
#include "common/measure.hpp"
#include "common/numeric.hpp"
#include "common/random.hpp"
#include "common/example_sm_runner.hpp"
#include "common/device_io.hpp"
#include "common/print.hpp"
#include "common/test_data.hpp"
#include "common/device_management.hpp"


namespace example {
    using namespace common;

} // namespace example

#endif
