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

#ifndef NPPDX_DETAIL_BACKEND_TEXTURE_OPERATIONS_HPP
#define NPPDX_DETAIL_BACKEND_TEXTURE_OPERATIONS_HPP

// Umbrella header for CUDA-array texture/surface support.
// Each format provides a texture_format_backend<Format> specialization; remove an
// include below to drop that format (its execute() calls then fail an
// is_implemented static_assert).

#include "nppdx/detail/backend/texture/packed_rgb.hpp"       // rgb24
#include "nppdx/detail/backend/texture/semi_planar_42x.hpp"  // nv12, p010, nv16, p216
#include "nppdx/detail/backend/texture/planar.hpp"           // yuv444p, yuv444p10, rgbp, bgrp

#endif // NPPDX_DETAIL_BACKEND_TEXTURE_OPERATIONS_HPP
