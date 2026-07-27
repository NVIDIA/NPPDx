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

#ifndef NPPDX_DETAIL_CLANG_CUDA_COMPAT_CSTDLIB_HPP
#define NPPDX_DETAIL_CLANG_CUDA_COMPAT_CSTDLIB_HPP

#pragma push_macro("min")
#pragma push_macro("max")
#pragma push_macro("abs")
#undef min
#undef max
#undef abs

#include <cstdlib>

#pragma pop_macro("abs")
#pragma pop_macro("max")
#pragma pop_macro("min")

#endif // NPPDX_DETAIL_CLANG_CUDA_COMPAT_CSTDLIB_HPP
