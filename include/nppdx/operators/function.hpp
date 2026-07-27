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

#ifndef NPPDX_OPERATORS_FUNCTION_HPP
#define NPPDX_OPERATORS_FUNCTION_HPP


#include "nppdx/detail/config.hpp"
#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"
#include "nppdx/operators/operator_type.hpp"
#include "nppdx/operators/color_space.hpp"
#include "nppdx/operators/operation_operator_traits.hpp"
#include "nppdx/detail/backend/resize_geometry.hpp"
#include "nppdx/detail/database/region.hpp"
#include "nppdx/detail/database/storage_config.hpp"
#include "nppdx/detail/decl.hpp"
#include "nppdx/detail/backend/affine_channel_map_operations.hpp"

#include NPPDX_STD_INCLUDE_CSTDINT
#include NPPDX_STD_INCLUDE_CASSERT
#include NPPDX_STD_INCLUDE_NUMERIC
#include NPPDX_STD_INCLUDE_ARRAY

namespace nppdx {
    // Function types for image processing operations
    enum class function
    {
        none,
        color_convert,
        gamma,
        affine_channel_map, //affine means linear mapping, so 0..255 -> 16..240 or other transformations
        sharpen,
        box_blur,      // NxM average filter - requires halo {(N-1)/2, (N-1)/2, (M-1)/2, (M-1)/2}
        gaussian_blur, // Discrete separable Gaussian FIR; odd tap count 3..21 by radius band (see gaussian_blur_params)
        median,        // Median filter - shape/radius from Median<median_radius> operator
        resize         //Rational resize
    };

    template<function TagValue>
    struct function_tag {
        using type                      = function;
        static constexpr function value = TagValue;
    };

    // Median filter shape by effective radius (rounds up to halo size).
    // r0_5 = 5-element plus; r1_0 = 3x3; r1_5 = 5x5 no corners; r2_0 = full 5x5.
    enum class median_radius
    {
        r0_5, // 5-element plus, halo 1
        r1_0, // 3x3, halo 1 (NPP equivalent)
        r1_5, // 5x5 without corners, halo 2
        r2_0  // full 5x5, halo 2
    };


    // Bit depth types for color conversion
    enum class bit_depth : NPPDX_STD::int32_t
    {
        bpp_8u  = 8,  // 8-bit unsigned (0-255)
        bpp_10u = 10, // 10-bit unsigned (0-1023) - default internal format
        bpp_16u = 16  // 16-bit unsigned (0-65535)
    };

    // Utility for converting an integer to a bit_depth.
    NPPDX_DECL_NCIHD bit_depth bit_depth_from_int(NPPDX_STD::int32_t int_value) {
        switch (int_value) {
            case 8: return bit_depth::bpp_8u;
            case 10: return bit_depth::bpp_10u;
            case 16: return bit_depth::bpp_16u;
            default:
                assert(false);
                return bit_depth::bpp_8u;
        }
    }

    // Direction of gamma function
    enum class gamma_dir
    {
        forward,
        inverse
    };

    // Type of a gamma curve
    enum class gamma_transfer_function
    {
        SDR, // Standard Dynamic Range
        HLG, // Hybrid log-gamma
        PQ   // Perceptual quantizer
    };

// Apply a custom macro to all interpolation methods.
#define NPPDX_FOR_INTERPOLATION_METHODS(MACRO, ...) \
    MACRO(nearest, __VA_ARGS__)                     \
    MACRO(bilinear, __VA_ARGS__)                    \
    MACRO(bicubic, __VA_ARGS__)                     \
    MACRO(lanczos3, __VA_ARGS__)

    // Interpolation methods
    enum class interpolation_method
    {
#define NPPDX_INTERPOLATION_METHOD_ENUM_ENTRY(name, ...) name,
        NPPDX_FOR_INTERPOLATION_METHODS(NPPDX_INTERPOLATION_METHOD_ENUM_ENTRY)
#undef NPPDX_INTERPOLATION_METHOD_ENUM_ENTRY
    };

    // Function operator template
    template<function FunctionType>
    struct Function: public commondx::detail::operator_expression {
        static_assert(FunctionType == function::none || (FunctionType == function::color_convert) ||
                          (FunctionType == function::gamma) || (FunctionType == function::affine_channel_map) ||
                          (FunctionType == function::sharpen) || (FunctionType == function::box_blur) ||
                          (FunctionType == function::gaussian_blur) || (FunctionType == function::median) ||
                          (FunctionType == function::resize),
                      "Unsupported function type");
        static constexpr function value = FunctionType;
    };

    // BoxBlur parameter operator with configurable kernel size
    // Width and Height must be odd numbers >= 1
    // Use with: Function<function::box_blur>() + BoxBlur<3, 3>()
    template<unsigned int Width, unsigned int Height>
    struct BoxBlur: public commondx::detail::operator_expression {
        static_assert(Width >= 1 && (Width % 2) == 1, "BoxBlur width must be odd and >= 1");
        static_assert(Height >= 1 && (Height % 2) == 1, "BoxBlur height must be odd and >= 1");

        static constexpr unsigned int width  = Width;
        static constexpr unsigned int height = Height;

        // Local halo is half the kernel size (floor division for odd kernels)
        static constexpr unsigned int halo_x = Width / 2;
        static constexpr unsigned int halo_y = Height / 2;

        static constexpr uint2 value = {width, height};

        using tag_type = function_tag<function::box_blur>;
    };

    NPPDX_DECL_NCFHD Halo4 compute_local_halo(const uint2& val) {
        const int2 uni_halo = int2(val) / int2::uniform(2);
        return {uni_halo, uni_halo};
    }

    NPPDX_DECL_NCHD detail::storage_config compute_storage_config(const uint2& /*box_blur_kernel*/,
                                                                  const layout_props& layout) {
        return detail::make_unary_storage_config(layout);
    }

    template<>
    struct operation_operator_traits<function_tag<function::box_blur>> {
        using value_type                        = uint2;
        NPPDX_DECL_SC operator_type op_type_val = operator_type::box_blur;
        NPPDX_DECL_NSCHD Halo4      get_local_halo(const value_type& val) { return compute_local_halo(val); }
        NPPDX_DECL_NSCHD detail::storage_config get_storage_config(const value_type& val, const layout_props& layout) {
            return compute_storage_config(val, layout);
        }
        NPPDX_DECL_NSCHD uint2 get_output_nominal_tile(const value_type&, uint2 input_nominal) { return input_nominal; }
        using default_op = BoxBlur<3, 3>;
    };

    // Gaussian blur

    // Discrete Gaussian tap tables: standard vs wide (same sigma = R; wide uses ~2x taps / support ~2*sigma).
    enum class gaussian_tail_width : unsigned char
    {
        standard,
        wide
    };

    struct tap_config {
        int min_radius_tenths;
        int max_radius_tenths;
        int standard_taps;

        NPPDX_DECL_NCFHD int tap_count(gaussian_tail_width width) const {
            switch (width) {
                case gaussian_tail_width::standard:
                    return standard_taps;
                case gaussian_tail_width::wide:
                    return 2 * standard_taps - 1;
                default:
                    fatal_error("Invalid gaussian_tail_width enumerator.");
                    return 0;
            }
        }
    };

    NPPDX_DECL_CI Array<tap_config, 10> tap_configs {{
        {/*min_radius_tenths=*/1, /*max_radius_tenths=*/9, /*standard_taps=*/3},
        {/*min_radius_tenths=*/10, /*max_radius_tenths=*/20, /*standard_taps=*/5},
        {/*min_radius_tenths=*/21, /*max_radius_tenths=*/30, /*standard_taps=*/7},
        {/*min_radius_tenths=*/31, /*max_radius_tenths=*/40, /*standard_taps=*/9},
        {/*min_radius_tenths=*/41, /*max_radius_tenths=*/50, /*standard_taps=*/11},
        {/*min_radius_tenths=*/51, /*max_radius_tenths=*/60, /*standard_taps=*/13},
        {/*min_radius_tenths=*/61, /*max_radius_tenths=*/70, /*standard_taps=*/15},
        {/*min_radius_tenths=*/71, /*max_radius_tenths=*/80, /*standard_taps=*/17},
        {/*min_radius_tenths=*/81, /*max_radius_tenths=*/90, /*standard_taps=*/19},
        {/*min_radius_tenths=*/91, /*max_radius_tenths=*/100, /*standard_taps=*/21},
    }};

    NPPDX_DECL_NCFHD tap_config get_tap_config(int radius_tenths) {
        bool gcc9_war = false;
        for (const auto& tap_config : tap_configs) {
            if (radius_tenths <= tap_config.max_radius_tenths) {
                return tap_config;
            }
            gcc9_war = true;
        }
        if (gcc9_war) {
            // Workaround for GCC 9 bug where it interprets this non-constexpr fatal_error() as always being called in
            // get_tap_config and consequently complains that get_tap_config() cannot be constexpr, unless we wrap it in
            // a dummy conditional.
            fatal_error("get_tap_config: radius_tenths out of range");
        }
        return {};
    }

    NPPDX_DECL_NCFHD int halo_for_fir_kernel(int kw) {
        return kw / 2;
    }

    struct gaussian_blur_params {
        int                 radius_tenths;
        gaussian_tail_width tail_width;

        NPPDX_DECL_NCFHD float radius() const { return static_cast<float>(radius_tenths) / 10.f; }
        NPPDX_DECL_NCFHD int   taps() const { return get_tap_config(radius_tenths).tap_count(tail_width); }
        NPPDX_DECL_NCFHD uint2 kernel_size() const { return uint2::uniform(taps()); }
        NPPDX_DECL_NCFHD Halo4 halo() const { return Halo4::make_uniform(halo_for_fir_kernel(taps())); }
    };

    // GaussianBlur parameter operator: radius 0.1 to 10.0 (RadiusTenths 1..100).
    // FIR bands: RT<=9 -> 3-tap ... <=100 -> 21-tap (see tap_configs). Wide: same sigma=R, taps = 2*taps_std-1
    // (support ~2*sigma); requires RadiusTenths <= 50 so wide taps stay within the 21-tap FIR max
    // (e.g. GaussianBlur<35, wide> -> 17-tap, halo 8).
    template<int RadiusTenths, gaussian_tail_width TailWidth = gaussian_tail_width::standard>
    struct GaussianBlur: public commondx::detail::operator_expression {
        static_assert(RadiusTenths >= 1 && RadiusTenths <= 100, "GaussianBlur radius 0.1-10.0 (RadiusTenths 1-100)");
        static_assert(TailWidth != gaussian_tail_width::wide || RadiusTenths <= 50,
                      "GaussianBlur wide requires RadiusTenths <= 50 (max 21-tap FIR)");

        static constexpr gaussian_blur_params value = {RadiusTenths, TailWidth};

        static constexpr int                 radius_tenths = RadiusTenths;
        static constexpr gaussian_tail_width tail_width    = TailWidth;

        using tag_type = function_tag<function::gaussian_blur>;
    };

    NPPDX_DECL_NCFHD Halo4 compute_local_halo(const gaussian_blur_params& params) {
        return params.halo();
    }

    NPPDX_DECL_NCHD detail::storage_config compute_storage_config(const gaussian_blur_params&,
                                                                  const layout_props& layout) {
        return detail::make_unary_storage_config(layout);
    }

    template<>
    struct operation_operator_traits<function_tag<function::gaussian_blur>> {
        using value_type                        = gaussian_blur_params;
        NPPDX_DECL_SC operator_type op_type_val = operator_type::gaussian_blur;
        NPPDX_DECL_NSCHD Halo4      get_local_halo(const value_type& val) { return compute_local_halo(val); }
        NPPDX_DECL_NSCHD detail::storage_config get_storage_config(const value_type& val, const layout_props& layout) {
            return compute_storage_config(val, layout);
        }
        NPPDX_DECL_NSCHD uint2 get_output_nominal_tile(const value_type&, uint2 input_nominal) { return input_nominal; }
        using default_op = GaussianBlur<20>;
    };

    NPPDX_DECL_NCFHD Halo4 compute_local_halo(const median_radius& val) {
        // Halo: r0_5 and r1_0 -> 1; r1_5 and r2_0 -> 2
        return Halo4::make_uniform((val == median_radius::r0_5 || val == median_radius::r1_0) ? 1u : 2u);
    }

    NPPDX_DECL_NCHD detail::storage_config compute_storage_config(const median_radius&, const layout_props& layout) {
        return detail::make_unary_storage_config(layout);
    }

    // Median parameter operator: shape by effective radius. Use with: Function<function::median>() + Median<median_radius::r1_0>().
    // Only r1_0 (3x3) is implemented initially; others reserved.
    template<median_radius Radius>
    struct Median: public commondx::detail::operator_expression {
        static constexpr median_radius value = Radius;

        static constexpr median_radius radius = Radius;

        using tag_type = function_tag<function::median>;

        static constexpr Halo4 halo = compute_local_halo(Radius);
    };

    template<>
    struct operation_operator_traits<function_tag<function::median>> {
        using value_type                        = median_radius;
        NPPDX_DECL_SC operator_type op_type_val = operator_type::median;
        NPPDX_DECL_NSCHD Halo4      get_local_halo(const value_type& val) { return compute_local_halo(val); }
        NPPDX_DECL_NSCHD detail::storage_config get_storage_config(const value_type& val, const layout_props& layout) {
            return compute_storage_config(val, layout);
        }
        NPPDX_DECL_NSCHD uint2 get_output_nominal_tile(const value_type&, uint2 input_nominal) { return input_nominal; }
        using default_op = Median<median_radius::r1_0>;
    };

    struct sharpen_data {
        float corner_w;
        float side_w;
        float center_w;
    };


    // Sharpen parameter operator: 3-float Weights (corner_w, side_w, center_w). Use with: Function<function::sharpen>() + Sharpen<Weights>().
    // For NPP verification, scale Weights to integer kernel + divisor via detail::backend::sharpen::npp_kernel_from_weights<Weights, Divisor>.
    template<typename Weights>
    struct Sharpen: public commondx::detail::operator_expression {
        static constexpr sharpen_data value = {Weights::corner_w, Weights::side_w, Weights::center_w};

        using weights_type = Weights;

        using tag_type = function_tag<function::sharpen>;
    };

    NPPDX_DECL_NCFHD Halo4 compute_local_halo(const sharpen_data&) {
        return Halo4::make_uniform(1);
    }

    NPPDX_DECL_NCHD detail::storage_config compute_storage_config(const sharpen_data&, const layout_props& layout) {
        return detail::make_unary_storage_config(layout);
    }

    // Preset 3x3 sharpen weight patterns (corner_w, side_w, center_w). Used with Sharpen<...> + function::sharpen.
    namespace sharpen_weights {

        // Generalized additive Rosenfeld Laplacian: WeightPercent (0-100) is additive sharpen strength.
        // At 0%: identity. At 50%: (-1/24, -2/24, 36/24). Coefficients scale so 4*corner_w + 4*side_w + center_w = 1.
        template<int WeightPercent>
        struct rosenfeld_generalized_weights {
            static constexpr float _weight  = static_cast<float>(WeightPercent) / 100.f;
            static constexpr float corner_w = -_weight / 12.f;
            static constexpr float side_w   = -_weight / 6.f;
            static constexpr float center_w = 1.f + _weight;
        };

        using rosenfeld_50_weights = rosenfeld_generalized_weights<50>;

        struct identity_weights {
            static constexpr float corner_w = 0.f;
            static constexpr float side_w   = 0.f;
            static constexpr float center_w = 1.f;
        };

        struct binomial_3x3_weights {
            static constexpr float corner_w = 0.0625f;
            static constexpr float side_w   = 0.1250f;
            static constexpr float center_w = 0.2500f;
        };
    } // namespace sharpen_weights

    template<>
    struct operation_operator_traits<function_tag<function::sharpen>> {
        using value_type                        = sharpen_data;
        NPPDX_DECL_SC operator_type op_type_val = operator_type::sharpen;
        NPPDX_DECL_NSCHD Halo4      get_local_halo(const value_type& val) { return compute_local_halo(val); }
        NPPDX_DECL_NSCHD detail::storage_config get_storage_config(const value_type& val, const layout_props& layout) {
            return compute_storage_config(val, layout);
        }
        NPPDX_DECL_NSCHD uint2 get_output_nominal_tile(const value_type&, uint2 input_nominal) { return input_nominal; }
        using default_op = Sharpen<sharpen_weights::rosenfeld_50_weights>;
    };

    struct ColorConvertData {
        color_space input_color_space;
        color_space output_color_space;
        bit_depth   input_bit_depth;
        bit_depth   output_bit_depth;
    };

    // ColorConvert function with input and output color spaces and bit depths
    template<color_space InputColorSpace, color_space OutputColorSpace, bit_depth InputBitDepth,
             bit_depth OutputBitDepth>
    struct ColorConvert: public commondx::detail::operator_expression {
        static constexpr ColorConvertData value = {InputColorSpace, OutputColorSpace, InputBitDepth, OutputBitDepth};

        static constexpr color_space input_color_space  = InputColorSpace;
        static constexpr color_space output_color_space = OutputColorSpace;
        static constexpr bit_depth   input_bit_depth    = InputBitDepth;
        static constexpr bit_depth   output_bit_depth   = OutputBitDepth;

        using tag_type = function_tag<function::color_convert>;
    };

    NPPDX_DECL_NCFHD Halo4 compute_local_halo(const ColorConvertData&) {
        return Halo4::make_empty();
    }

    NPPDX_DECL_NCHD detail::storage_config compute_storage_config(const ColorConvertData&, const layout_props& layout) {
        return detail::make_inplace_storage_config(layout);
    }

    template<>
    struct operation_operator_traits<function_tag<function::color_convert>> {
        using value_type                        = ColorConvertData;
        NPPDX_DECL_SC operator_type op_type_val = operator_type::color_convert;
        NPPDX_DECL_NSCHD Halo4      get_local_halo(const value_type& val) { return compute_local_halo(val); }
        NPPDX_DECL_NSCHD detail::storage_config get_storage_config(const value_type& val, const layout_props& layout) {
            return compute_storage_config(val, layout);
        }
        NPPDX_DECL_NSCHD uint2 get_output_nominal_tile(const value_type&, uint2 input_nominal) { return input_nominal; }
        using default_op = ColorConvert<color_space::rgb, color_space::rgb, bit_depth::bpp_10u, bit_depth::bpp_10u>;
    };

    struct GammaTransformData {
        bit_depth               input_output_bit_depth;
        gamma_dir               transform_direction;
        gamma_transfer_function transfer_function;
    };

    template<bit_depth InputOutputBitDepth, gamma_dir TransformDirection, gamma_transfer_function TransferFunction>
    struct GammaTransform: public commondx::detail::operator_expression {
        static constexpr GammaTransformData value = {InputOutputBitDepth, TransformDirection, TransferFunction};

        static constexpr bit_depth               input_output_bit_depth = InputOutputBitDepth;
        static constexpr gamma_dir               transform_direction    = TransformDirection;
        static constexpr gamma_transfer_function transfer_function      = TransferFunction;

        using tag_type = function_tag<function::gamma>;
    };

    NPPDX_DECL_NCFHD Halo4 compute_local_halo(const GammaTransformData&) {
        return Halo4::make_empty();
    }

    NPPDX_DECL_NCHD detail::storage_config compute_storage_config(const GammaTransformData&,
                                                                  const layout_props& layout) {
        return detail::make_inplace_storage_config(layout);
    }

    template<>
    struct operation_operator_traits<function_tag<function::gamma>> {
        using value_type                        = GammaTransformData;
        NPPDX_DECL_SC operator_type op_type_val = operator_type::gamma_transform;
        NPPDX_DECL_NSCHD Halo4      get_local_halo(const value_type& val) { return compute_local_halo(val); }
        NPPDX_DECL_NSCHD detail::storage_config get_storage_config(const value_type& val, const layout_props& layout) {
            return compute_storage_config(val, layout);
        }
        NPPDX_DECL_NSCHD uint2 get_output_nominal_tile(const value_type&, uint2 input_nominal) { return input_nominal; }
        using default_op = GammaTransform<bit_depth::bpp_10u, gamma_dir::forward, gamma_transfer_function::SDR>;
    };

    struct AffineChannelMapData {
        // out = (in + pre_offset) * (numerator / denominator) + post_offset, optionally clipped to [clip_min, clip_max]
        int32_t  pre_offset;
        int32_t  numerator;
        int32_t  denominator;
        int32_t  post_offset;
        int32_t  clip_min;
        int32_t  clip_max;
        uint32_t channel_mask;
    };

    template<int32_t PreOffset, int32_t Numerator, int32_t Denominator, int32_t PostOffset,
             int32_t ClipMin = NPPDX_AFFINE_NO_CLIP, int32_t ClipMax = NPPDX_AFFINE_NO_CLIP,
             uint32_t ChannelMask = 0xFFFFFFFFu>
    struct AffineChannelMap: public commondx::detail::operator_expression {
        // denominator == 0 is only valid with numerator == 0 (scale step omitted when equal).
        static_assert(Denominator != 0 || Numerator == 0, "denominator must be non-zero unless numerator is also zero");
        static_assert(ClipMin == NPPDX_AFFINE_NO_CLIP || ClipMax == NPPDX_AFFINE_NO_CLIP || ClipMin <= ClipMax,
                      "clip_min > clip_max is invalid when both clip bounds are enabled "
                      "(neither is NPPDX_AFFINE_NO_CLIP)");

        static constexpr AffineChannelMapData value = {PreOffset, Numerator, Denominator, PostOffset,
                                                       ClipMin,   ClipMax,   ChannelMask};

        static constexpr int32_t  pre_offset   = PreOffset;
        static constexpr int32_t  numerator    = Numerator;
        static constexpr int32_t  denominator  = Denominator;
        static constexpr int32_t  post_offset  = PostOffset;
        static constexpr int32_t  clip_min     = ClipMin;
        static constexpr int32_t  clip_max     = ClipMax;
        static constexpr uint32_t channel_mask = ChannelMask;

        using tag_type = function_tag<function::affine_channel_map>;
    };

    NPPDX_DECL_NCFHD Halo4 compute_local_halo(const AffineChannelMapData&) {
        return Halo4::make_empty();
    }

    NPPDX_DECL_NCHD detail::storage_config compute_storage_config(const AffineChannelMapData&,
                                                                  const layout_props& layout) {
        return detail::make_inplace_storage_config(layout);
    }

    template<>
    struct operation_operator_traits<function_tag<function::affine_channel_map>> {
        using value_type                        = AffineChannelMapData;
        NPPDX_DECL_SC operator_type op_type_val = operator_type::affine_channel_map;
        NPPDX_DECL_NSCHD Halo4      get_local_halo(const value_type& val) { return compute_local_halo(val); }
        NPPDX_DECL_NSCHD detail::storage_config get_storage_config(const value_type& val, const layout_props& layout) {
            return compute_storage_config(val, layout);
        }
        NPPDX_DECL_NSCHD uint2 get_output_nominal_tile(const value_type&, uint2 input_nominal) { return input_nominal; }
        using default_op = AffineChannelMap<0, 1, 1, 0>; //identity mapping, nop
    };

    struct ResizeData {
        unsigned int         input_samples;
        unsigned int         output_samples;
        interpolation_method method;
    };

    // Resize operator
    // J input samples -> K output samples
    // Upscale: J < K (e.g., J=1, K=2 for 2x upscale)
    // Downscale: J > K (e.g., J=2, K=1 for 2x downscale)
    template<unsigned int J, unsigned int K, interpolation_method Method>
    struct Resize: public commondx::detail::operator_expression {
        static constexpr ResizeData value = {J, K, Method};

        static_assert(J >= 1, "J (input samples) must be at least 1");
        static_assert(K >= 1, "K (output samples) must be at least 1");
        static_assert(NPPDX_STD::gcd(J, K) == 1, "J and K must be coprime");

        static constexpr unsigned int         input_samples  = J;
        static constexpr unsigned int         output_samples = K;
        static constexpr interpolation_method method         = Method;

        using tag_type = function_tag<function::resize>;
    };

    // Filter properties based on interpolation method
    struct filter_props {
        uint mu_num;
        uint mu_den;

        NPPDX_DECL_NCFHD uint radius_num(uint J, uint K) const { return mu_num * ((J > K) ? J : 1); }
        NPPDX_DECL_NCFHD uint radius_den(uint J, uint K) const { return mu_den * ((J > K) ? K : 1); }

        NPPDX_DECL_NCFHD uint radius_ceil(uint J, uint K) const {
            return (2 * radius_num(J, K) + radius_den(J, K)) / (2 * radius_den(J, K));
        }
        NPPDX_DECL_NCFHD uint taps(uint J, uint K) const {
            if (mu_num < mu_den) {
                // If the radius is less than 1, return 1 exactly. Currently true for nearest-neighbor.
                return 1u;
            }
            // Otherwise return the result rounded up to the nearest even integer to simplify expressions.
            const uint raw_taps = ceil_div(2 * radius_num(J, K), radius_den(J, K));
            return raw_taps + (raw_taps % 2);
        }
    };

    NPPDX_DECL_NCFHD filter_props get_filter_props(interpolation_method Method) {
        switch (Method) {
            case interpolation_method::nearest: return filter_props {1, 2};
            case interpolation_method::bilinear: return filter_props {1, 1};
            case interpolation_method::bicubic: return filter_props {2, 1};
            case interpolation_method::lanczos3: return filter_props {3, 1};
            default: fatal_error("Invalid interpolation method");
        }
    }

    NPPDX_DECL_NCFHD Halo4 compute_local_halo(const ResizeData& val) {
        return Halo4::make_uniform(get_filter_props(val.method).radius_ceil(val.input_samples, val.output_samples));
    }

    NPPDX_DECL_NCHD uint2 compute_output_nominal_tile(const ResizeData& val, uint2 input_nominal) {
        return detail::backend::resize::compute_resize_output_nominal_tile(input_nominal, val.input_samples,
                                                                           val.output_samples);
    }

    NPPDX_DECL_NCHD detail::storage_config compute_storage_config(const ResizeData& val, const layout_props& layout) {
        const uint2 nominal     = layout.nominal_size;
        const uint2 input_size  = layout.memory_size();
        const uint2 output_size = compute_output_nominal_tile(val, nominal);
        const uint2 inter_size  = {output_size.x, detail::make_input_region(layout).size.y};

        const unsigned int n = detail::storage_num_channels;
        return {detail::tile_shape {input_size, n}, detail::tile_shape {output_size, n},
                detail::tile_shape {inter_size, n}};
    }

    template<>
    struct operation_operator_traits<function_tag<function::resize>> {
        using value_type                        = ResizeData;
        NPPDX_DECL_SC operator_type op_type_val = operator_type::resize;
        NPPDX_DECL_NSCHD Halo4      get_local_halo(const value_type& val) { return compute_local_halo(val); }
        NPPDX_DECL_NSCHD detail::storage_config get_storage_config(const value_type& val, const layout_props& layout) {
            return compute_storage_config(val, layout);
        }
        NPPDX_DECL_NSCHD uint2 get_output_nominal_tile(const value_type& val, uint2 input_nominal) {
            return compute_output_nominal_tile(val, input_nominal);
        }
        using default_op = Resize<1, 2, interpolation_method::bilinear>;
    };

} // namespace nppdx

namespace commondx::detail {

    using default_function = nppdx::Function<nppdx::function::none>;
    // Function operator specializations
    template<nppdx::function FunctionType>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::function, nppdx::Function<FunctionType>>:
        NPPDX_STD::true_type {
    };

    template<nppdx::function FunctionType>
    struct get_operator_type<nppdx::operator_type, nppdx::Function<FunctionType>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::function;
    };

    template<nppdx::color_space InputColorSpace, nppdx::color_space OutputColorSpace, nppdx::bit_depth InputBitDepth,
             nppdx::bit_depth OutputBitDepth>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::color_convert,
                       nppdx::ColorConvert<InputColorSpace, OutputColorSpace, InputBitDepth, OutputBitDepth>>:
        NPPDX_STD::true_type {
    };

    template<nppdx::color_space InputColorSpace, nppdx::color_space OutputColorSpace, nppdx::bit_depth InputBitDepth,
             nppdx::bit_depth OutputBitDepth>
    struct get_operator_type<nppdx::operator_type,
                             nppdx::ColorConvert<InputColorSpace, OutputColorSpace, InputBitDepth, OutputBitDepth>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::color_convert;
    };

    template<nppdx::bit_depth InputOutputBitDepth, nppdx::gamma_dir TransformDirection,
             nppdx::gamma_transfer_function TransferFunction>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::gamma_transform,
                       nppdx::GammaTransform<InputOutputBitDepth, TransformDirection, TransferFunction>>:
        NPPDX_STD::true_type {
    };

    template<nppdx::bit_depth InputOutputBitDepth, nppdx::gamma_dir TransformDirection,
             nppdx::gamma_transfer_function TransferFunction>
    struct get_operator_type<nppdx::operator_type,
                             nppdx::GammaTransform<InputOutputBitDepth, TransformDirection, TransferFunction>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::gamma_transform;
    };

    template<int32_t PreOffset, int32_t Numerator, int32_t Denominator, int32_t PostOffset, int32_t ClipMin,
             int32_t ClipMax, uint32_t ChannelMask>
    struct is_operator<
        nppdx::operator_type, nppdx::operator_type::affine_channel_map,
        nppdx::AffineChannelMap<PreOffset, Numerator, Denominator, PostOffset, ClipMin, ClipMax, ChannelMask>>:
        NPPDX_STD::true_type {
    };

    template<int32_t PreOffset, int32_t Numerator, int32_t Denominator, int32_t PostOffset, int32_t ClipMin,
             int32_t ClipMax, uint32_t ChannelMask>
    struct get_operator_type<nppdx::operator_type, nppdx::AffineChannelMap<PreOffset, Numerator, Denominator,
                                                                           PostOffset, ClipMin, ClipMax, ChannelMask>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::affine_channel_map;
    };

    template<nppdx::function FunctionType>
    struct is_color_convert:
        COMMONDX_STL_NAMESPACE::integral_constant<bool, (FunctionType == nppdx::function::color_convert)> {
    };

    template<nppdx::function FunctionType>
    struct is_gamma: COMMONDX_STL_NAMESPACE::integral_constant<bool, (FunctionType == nppdx::function::gamma)> {
    };

    template<nppdx::function FunctionType>
    struct is_affine_channel_map:
        COMMONDX_STL_NAMESPACE::integral_constant<bool, (FunctionType == nppdx::function::affine_channel_map)> {
    };

    template<nppdx::function FunctionType>
    struct is_sharpen: COMMONDX_STL_NAMESPACE::integral_constant<bool, (FunctionType == nppdx::function::sharpen)> {
    };

    template<nppdx::function FunctionType>
    struct is_box_blur: COMMONDX_STL_NAMESPACE::integral_constant<bool, (FunctionType == nppdx::function::box_blur)> {
    };

    template<nppdx::function FunctionType>
    struct is_gaussian_blur:
        COMMONDX_STL_NAMESPACE::integral_constant<bool, (FunctionType == nppdx::function::gaussian_blur)> {
    };

    template<nppdx::function FunctionType>
    struct is_median: COMMONDX_STL_NAMESPACE::integral_constant<bool, (FunctionType == nppdx::function::median)> {
    };

    // BoxBlur operator specializations (parameter operator, like ColorConvert)
    template<unsigned int W, unsigned int H>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::box_blur, nppdx::BoxBlur<W, H>>:
        NPPDX_STD::true_type {
    };

    template<unsigned int W, unsigned int H>
    struct get_operator_type<nppdx::operator_type, nppdx::BoxBlur<W, H>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::box_blur;
    };

    template<int RadiusTenths, nppdx::gaussian_tail_width Tw>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::gaussian_blur,
                       nppdx::GaussianBlur<RadiusTenths, Tw>>: NPPDX_STD::true_type {
    };

    template<int RadiusTenths, nppdx::gaussian_tail_width Tw>
    struct get_operator_type<nppdx::operator_type, nppdx::GaussianBlur<RadiusTenths, Tw>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::gaussian_blur;
    };

    template<typename W>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::sharpen, nppdx::Sharpen<W>>: NPPDX_STD::true_type {
    };

    template<typename W>
    struct get_operator_type<nppdx::operator_type, nppdx::Sharpen<W>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::sharpen;
    };

    template<nppdx::median_radius R>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::median, nppdx::Median<R>>: NPPDX_STD::true_type {
    };

    template<nppdx::median_radius R>
    struct get_operator_type<nppdx::operator_type, nppdx::Median<R>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::median;
    };

    template<nppdx::function FunctionType>
    struct is_resize: COMMONDX_STL_NAMESPACE::integral_constant<bool, (FunctionType == nppdx::function::resize)> {
    };

    // Resize operator specializations.
    template<unsigned int J, unsigned int K, nppdx::interpolation_method Method>
    struct is_operator<nppdx::operator_type, nppdx::operator_type::resize, nppdx::Resize<J, K, Method>>:
        NPPDX_STD::true_type {
    };

    template<unsigned int J, unsigned int K, nppdx::interpolation_method Method>
    struct get_operator_type<nppdx::operator_type, nppdx::Resize<J, K, Method>> {
        static constexpr nppdx::operator_type value = nppdx::operator_type::resize;
    };


} // namespace commondx::detail

#endif // NPPDX_OPERATORS_FUNCTION_HPP
