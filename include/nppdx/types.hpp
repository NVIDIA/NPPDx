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

#ifndef NPPDX_TYPES_HPP
#define NPPDX_TYPES_HPP

#include <nppdx/detail/config.hpp>
#include "nppdx/detail/decl.hpp"
#include "nppdx/utils.hpp"

#include NPPDX_STD_INCLUDE_CSTDDEF
#include NPPDX_STD_INCLUDE_CSTDINT
namespace nppdx {

    // Simple array wrapper, useful for passing arrays by value.
    template<typename T, int Size>
    struct Array {
        T value[Size];

#define NPPDX_PRIVATE_ARRAY_GET_BODY(SizeRef, ValRef) \
    if constexpr (SizeRef == 1) {                     \
        return ValRef[0];                             \
    } else {                                          \
        return ValRef;                                \
    }

        constexpr auto get() { NPPDX_PRIVATE_ARRAY_GET_BODY(Size, value) }
        constexpr auto get() const {NPPDX_PRIVATE_ARRAY_GET_BODY(Size, value)}
#undef NPPDX_PRIVATE_ARRAY_GET_BODY

        NPPDX_DECL_NCFHD T& operator[](int index) {
            return value[index];
        }
        NPPDX_DECL_NCFHD const T& operator[](int index) const { return value[index]; }

        template<typename Functor>
        NPPDX_DECL_CHD auto transform(Functor&& functor) const {
            using return_type = decltype(functor(value[0]));
            Array<return_type, Size> ret_val {};
            for (int i = 0; i < Size; i++) {
                ret_val.value[i] = functor(value[i]);
            }
            return ret_val;
        }

        template<typename U>
        NPPDX_DECL_NCHD auto cast() const {
            return transform([](const T& val) { return U(val); });
        }

        NPPDX_DECL_NCHD T*       begin() { return value; }
        NPPDX_DECL_NCHD const T* begin() const { return value; }
        NPPDX_DECL_NCHD T*       end() { return value + Size; }
        NPPDX_DECL_NCHD const T* end() const { return value + Size; }
    };

    template<typename T, typename... Args>
    NPPDX_DECL_NCFHD Array<T, int(sizeof...(Args))> make_array(Args&&... args) {
        return {{args...}};
    }

    // Image layout structure. Note that it does not include the size, only
    // the base pointers and strides.
    template<typename T, int PlaneCount>
    struct ImageLayout {
        Array<T*, PlaneCount>  ptrs;
        Array<int, PlaneCount> byte_strides;

        NPPDX_DECL_NCHD Array<int, PlaneCount> element_strides() const {
            return byte_strides.transform([](int stride) { return stride / int(sizeof(T)); });
        }
    };


    // Fwd decl.
    template<typename T>
    struct vec2;

    using uint   = unsigned int;
    using uint8  = NPPDX_STD::uint8_t;
    using uint16 = NPPDX_STD::uint16_t;
    using uint32 = NPPDX_STD::uint32_t;
    using uint64 = NPPDX_STD::uint64_t;

    // Texture/surface object types — interchangeable with cudaTextureObject_t / CUtexObject
    // and cudaSurfaceObject_t / CUsurfObject respectively.
    // Defined as unsigned long long to avoid requiring <cuda_runtime.h> (FFmpeg compatibility).
    // NOTE: TexObjT and SurfObjT are the same underlying type; the ingest/exgest execute()
    // overloads disambiguate by operation direction, not by the handle type.
    //
    // Required configuration of the texture/surface objects passed to the texture/surface
    // execute() overloads. The caller creates these; the library does not inspect or
    // reconfigure them, so a mismatch silently produces wrong results.
    //   Textures: normalizedCoords = 0 (unnormalized, integer pixel coordinates),
    //             readMode    = cudaReadModeElementType,
    //             filterMode  = cudaFilterModePoint,
    //             addressMode = cudaAddressModeClamp (edge replication supplies the halo).
    //   Surfaces: backing cudaArray created with the cudaArraySurfaceLoadStore flag.
    //   Both:     the array's channel format must match the packing_format's element type
    //             per plane — rgb24 = uchar4, nv12 = uchar + uchar2,
    //             p010 = ushort + ushort2.
    using TexObjT  = unsigned long long;
    using SurfObjT = unsigned long long;

    // Bundles of texture/surface handles for planar formats, one handle per
    // plane (e.g. NV12: [0] = luma, [1] = interleaved chroma). NumPlanes equals the format's
    // plane count; single-plane packed formats (rgb24) use the bare TexObjT / SurfObjT overloads.
    template<int NumPlanes>
    using TexObjPlanes = Array<TexObjT, NumPlanes>;
    template<int NumPlanes>
    using SurfObjPlanes = Array<SurfObjT, NumPlanes>;

    using bool2 = vec2<bool>;
    using int2  = vec2<int>;
    using uint2 = vec2<uint>;
    using size2 = vec2<NPPDX_STD::size_t>;

    template<typename Fn, typename... Vec2Ts>
    NPPDX_DECL_CHD auto vec2_apply(Fn&& fn, const Vec2Ts&... vecs) {
        using R = decltype(fn(vecs.x...));
        return vec2<R>(fn(vecs.x...), fn(vecs.y...));
    }

    template<typename T>
    struct vec2 {
        using value_type = T;

        T x;
        T y;

        // Constructors and static creation functions.

        NPPDX_DECL_CFHD vec2(T x_ = 0, T y_ = 0): x(x_), y(y_) {}

        template<typename U>
        explicit NPPDX_DECL_CFHD vec2(const vec2<U>& other): x(T(other.x)), y(T(other.y)) {}

        NPPDX_DECL_NSCFHD vec2 uniform(T val) { return vec2(val, val); }

        NPPDX_DECL_NSCFHD vec2 zero() { return uniform(0); }

        // Non-static member functions.

        NPPDX_DECL_NCHD vec2 ceil_div(const vec2& other) const { return (*this + other - vec2::uniform(T(1))) / other; }

        template<int Dir>
        NPPDX_DECL_NCHD vec2 dir_quot(const vec2& other) const {
            return {nppdx::dir_quot<Dir>(x, other.x), nppdx::dir_quot<Dir>(y, other.y)};
        }

        template<int Dir>
        NPPDX_DECL_NCHD vec2 dir_rem(const vec2& other) const {
            return {nppdx::dir_rem<Dir>(x, other.x), nppdx::dir_rem<Dir>(y, other.y)};
        }

        NPPDX_DECL_NCHD bool is_zero() const { return x == 0 && y == 0; }

        // Unary operators.

        NPPDX_DECL_NCHD vec2 operator-() const { return vec2(-x, -y); }

        // Hidden-friend operator definitions.

#define NPPDX_PRIVATE_DEFINE_VEC2_BINARY_OPERATOR(Op, RetT)                                      \
    [[nodiscard]] friend NPPDX_DECL_CHD vec2<RetT> operator Op(const vec2 & a, const vec2 & b) { \
        return {RetT(a.x Op b.x), RetT(a.y Op b.y)};                                             \
    }

#define NPPDX_PRIVATE_DEFINE_VEC2_BINARY_ARITHMETIC_OPERATOR(Op) NPPDX_PRIVATE_DEFINE_VEC2_BINARY_OPERATOR(Op, T)
#define NPPDX_PRIVATE_DEFINE_VEC2_BINARY_LOGICAL_OPERATOR(Op) NPPDX_PRIVATE_DEFINE_VEC2_BINARY_OPERATOR(Op, bool)

        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_ARITHMETIC_OPERATOR(+)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_ARITHMETIC_OPERATOR(-)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_ARITHMETIC_OPERATOR(*)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_ARITHMETIC_OPERATOR(/)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_ARITHMETIC_OPERATOR(%)

        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_LOGICAL_OPERATOR(==)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_LOGICAL_OPERATOR(!=)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_LOGICAL_OPERATOR(<)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_LOGICAL_OPERATOR(<=)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_LOGICAL_OPERATOR(>)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_LOGICAL_OPERATOR(>=)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_LOGICAL_OPERATOR(&&)
        NPPDX_PRIVATE_DEFINE_VEC2_BINARY_LOGICAL_OPERATOR(||)

#undef NPPDX_PRIVATE_DEFINE_VEC2_BINARY_OPERATOR
#undef NPPDX_PRIVATE_DEFINE_VEC2_BINARY_ARITHMETIC_OPERATOR
#undef NPPDX_PRIVATE_DEFINE_VEC2_BINARY_LOGICAL_OPERATOR
    };

    // Boolean reductions.

    NPPDX_DECL_NCFHD bool all(const bool2& a) {
        return a.x && a.y;
    }
    NPPDX_DECL_NCFHD bool any(const bool2& a) {
        return a.x || a.y;
    }

    // Emulate dim3 from CUDA runtime.
    NPPDX_DECL_NCFHD uint2 make_dim2(uint x = 1, uint y = 1) {
        return uint2 {x, y};
    }

    namespace tp { // tp stands for "template parameter"
        template<typename ValT, ValT XV, ValT YV>
        struct Vec2D {
            NPPDX_DECL_SC auto value = vec2<ValT> {XV, YV};

            using value_type           = ValT;
            NPPDX_DECL_SC value_type X = XV;
            NPPDX_DECL_SC value_type Y = YV;
        };

        template<typename T, typename ValT>
        NPPDX_DECL_CI bool IsVec2D = false;
        template<typename ValT, ValT XV, ValT YV>
        NPPDX_DECL_CI bool IsVec2D<Vec2D<ValT, XV, YV>, ValT> = true;

        // copy() is used below as a workaround for a GCC bug present in versions 9.4-13.4 where the compiler wrongly
        // complains that Val should not only be constexpr but also static.
#define NPPDX_MAKE_TP_VEC2D(Class, Val) Class<::nppdx::copy((Val).x), ::nppdx::copy((Val).y)>

    } // namespace tp

    template<int XV, int YV>
    using Int2D = tp::Vec2D<int, XV, YV>;
    template<typename T>
    NPPDX_DECL_CI bool IsInt2D = tp::IsVec2D<T, int>;
#define NPPDX_MAKE_INT2D(Val) NPPDX_MAKE_TP_VEC2D(::nppdx::Int2D, Val)

    template<uint XV, uint YV>
    using UInt2D = tp::Vec2D<uint, XV, YV>;
    template<typename T>
    NPPDX_DECL_CI bool IsUInt2D = tp::IsVec2D<T, uint>;
#define NPPDX_MAKE_UINT2D(Val) NPPDX_MAKE_TP_VEC2D(::nppdx::UInt2D, Val)

} // namespace nppdx

#endif // NPPDX_TYPES_HPP
