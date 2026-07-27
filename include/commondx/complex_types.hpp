/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMONDX_COMPLEX_TYPES_HPP
#define COMMONDX_COMPLEX_TYPES_HPP

// FP16, BF16 types are available in CUDA 11.0 https://docs.nvidia.com/cuda/archive/11.0/cuda-math-api/index.html
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// FP8 types are available starting CUDA 11.8+ https://docs.nvidia.com/cuda/archive/11.8.0/cuda-math-api/index.html
#if (__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 8))
#define COMMONDX_DETAIL_CUDA_FP8_ENABLED 1
#include <cuda_fp8.h>
#endif

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/cstdint.hpp"

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        // Validation helpers type conversions
        template<typename T, typename K>
        struct is_valid_conversion_to_from {
            static constexpr bool value = true;
        };

        template<class T>
        struct complex_base {
            using value_type = T;

            complex_base()                    = default;
            complex_base(const complex_base&) = default;
            complex_base(complex_base&&)      = default;
            __device__ __forceinline__ __host__ constexpr complex_base(value_type re, value_type im): x(re), y(im) {}

            __device__ __forceinline__ __host__ constexpr value_type real() const { return x; }
            __device__ __forceinline__ __host__ constexpr value_type imag() const { return y; }
            __device__ __forceinline__ __host__ void                 real(value_type re) { x = re; }
            __device__ __forceinline__ __host__ void                 imag(value_type im) { y = im; }

            __device__ __forceinline__ __host__ complex_base& operator=(value_type re) {
                x = re;
                y = value_type(); // zero-initialized with default ctor. see https://en.cppreference.com/w/cpp/language/value_initialization.
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator+=(value_type re) {
                x += re;
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator-=(value_type re) {
                x -= re;
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator*=(value_type re) {
                x *= re;
                y *= re;
                return *this;
            }
            __device__ __forceinline__ __host__ complex_base& operator/=(value_type re) {
                x /= re;
                y /= re;
                return *this;
            }

            complex_base& operator=(const complex_base&) = default;
            complex_base& operator=(complex_base&&) = default;

            template<class K>
            __device__ __forceinline__ __host__ complex_base& operator=(const complex_base<K>& other) {
                x = other.real();
                y = other.imag();
                return *this;
            }

            template<class OtherType>
            __device__ __forceinline__ __host__ complex_base& operator+=(const OtherType& other) {
                x = x + other.x;
                y = y + other.y;
                return *this;
            }

            template<class OtherType>
            __device__ __forceinline__ __host__ complex_base& operator-=(const OtherType& other) {
                x = x - other.x;
                y = y - other.y;
                return *this;
            }

            template<class OtherType>
            __device__ __forceinline__ __host__ complex_base& operator*=(const OtherType& other) {
                auto saved_x = x;
                x            = x * other.x - y * other.y;
                y            = saved_x * other.y + y * other.x;
                return *this;
            }

            /// \internal
            value_type x, y;
        };

        template<typename T>
        struct alignas(2 * sizeof(T)) complex : complex_base<T> {
        private:
            using base_type = complex_base<T>;

        public:
            using value_type        = T;
            complex()               = default;
            complex(const complex&) = default;
            complex(complex&&)      = default;
            __device__ __forceinline__ __host__ constexpr complex(T re, T im): base_type(re, im) {}

            template<class K>
            __device__ __forceinline__ __host__ explicit constexpr complex(const complex<K>& other) : complex(T(other.real()), T(other.imag())) {
                static_assert(is_valid_conversion_to_from<T, K>::value, "Unsupported conversion from K to T");
            };

            using base_type::operator+=;
            using base_type::operator-=;
            using base_type::operator*=;
            using base_type::operator/=;
            using base_type::operator=;

            complex& operator=(const complex&) = default;
            complex& operator=(complex&&)      = default;

            template<class K>
            __device__ __forceinline__ __host__
            complex& operator=(const complex<K>& other) {
                static_assert(is_valid_conversion_to_from<T, K>::value, "Unsupported conversion from K to T");
                this->x = static_cast<T>(other.real());
                this->y = static_cast<T>(other.imag());
                return *this;
            }
        };

        template<typename T>
        struct use_default_operator {
            static constexpr bool value = COMMONDX_STL_NAMESPACE::is_same_v<T, float>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, double>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, int8_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, uint8_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, int16_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, uint16_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, int32_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, uint32_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, int64_t>
                                       || COMMONDX_STL_NAMESPACE::is_same_v<T, uint64_t>
                                       ;
        };

        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<use_default_operator<T>::value, int> = 0>
        __device__ __forceinline__ __host__ complex<T> operator*(const complex<T>& a, const complex<T>& b) {
            return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
        }

        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<use_default_operator<T>::value, int> = 0>
        __device__ __forceinline__ __host__ complex<T> operator+(const complex<T>& a, const complex<T>& b) {
            return {a.x + b.x, a.y + b.y};
        }

        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<use_default_operator<T>::value, int> = 0>
        __device__ __forceinline__ __host__ complex<T> operator-(const complex<T>& a, const complex<T>& b) {
            return {a.x - b.x, a.y - b.y};
        }

        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<use_default_operator<T>::value, int> = 0>
        __device__ __forceinline__ __host__ bool operator==(const complex<T>& a, const complex<T>& b) {
            return (a.x == b.x) && (a.y == b.y);
        }

        // FP8
#ifdef COMMONDX_DETAIL_CUDA_FP8_ENABLED
        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_fp8_e4m3> ||
                                                                 COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_fp8_e5m2>, int> = 0>
        __device__ __forceinline__ complex<T> operator*(const complex<T>& a, const complex<T>& b) {
            return complex<T>(complex<float>(a) * complex<float>(b));
        }

        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_fp8_e4m3> ||
                                                                 COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_fp8_e5m2>, int> = 0>
        __device__ __forceinline__ complex<T> operator+(const complex<T>& a, const complex<T>& b) {
            return complex<T>(complex<float>(a) + complex<float>(b));
        }

        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_fp8_e4m3> ||
                                                                 COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_fp8_e5m2>, int> = 0>
        __device__ __forceinline__ complex<T> operator-(const complex<T>& a, const complex<T>& b) {
            return complex<T>(complex<float>(a) - complex<float>(b));
        }

        template<typename T, COMMONDX_STL_NAMESPACE::enable_if_t<COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_fp8_e4m3> ||
                                                                 COMMONDX_STL_NAMESPACE::is_same_v<T, __nv_fp8_e5m2>, int> = 0>
        __device__ __forceinline__ bool operator==(const complex<T>& a, const complex<T>& b) {
            return complex<float>(a) == complex<float>(b);
        }
#endif

        // FP16
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
        template<>
        struct alignas(2 * sizeof(__half)) complex<__half> {
            using value_type        = __half;
            complex()               = default;
            complex(const complex&) = default;
            complex(complex&&)      = default;

            __device__ __forceinline__ __host__ complex(const __half2& h): xy(h) {};
            __device__ __forceinline__ __host__ complex(__half re, __half im): xy(re, im) {}
            __device__ __forceinline__ __host__ complex(float re, float im):
                xy(__float2half_rn(re), __float2half_rn(im)) {}
            __device__ __forceinline__ __host__ complex(double re, double im):
                xy(__double2half(re), __double2half(im)) {}

            template<class K>
            __device__ __forceinline__ __host__ explicit complex(const complex<K>& other) : complex(__half(other.real()), __half(other.imag())) {
                static_assert(is_valid_conversion_to_from<__half, K>::value, "Unsupported conversion from K to T");
            };

            __device__ __forceinline__ __host__ value_type real() const { return xy.x; }
            __device__ __forceinline__ __host__ value_type imag() const { return xy.y; }
            __device__ __forceinline__ __host__ void       real(value_type re) { xy.x = re; }
            __device__ __forceinline__ __host__ void       imag(value_type im) { xy.y = im; }

            complex& operator=(const complex&) = default;
            complex& operator=(complex&&) = default;

            __device__ __forceinline__ __host__ complex& operator=(value_type re) {
                xy.x = re;
                xy.y = value_type(); // zero-initialized with default ctor. see https://en.cppreference.com/w/cpp/language/value_initialization.
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(value_type re) {
                xy.x += re;
                return *this;
            }
            __device__ __forceinline__ complex& operator-=(value_type re) {
                xy.x -= re;
                return *this;
            }
            __device__ __forceinline__ complex& operator*=(value_type re) {
                xy = __hmul2(xy, __half2 {re, re});
                return *this;
            }
            __device__ __forceinline__ complex& operator/=(value_type re) {
                xy = __h2div(xy, __half2 {re, re});
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(const complex& other) {
                xy += other.xy;
                return *this;
            }

            __device__ __forceinline__ complex& operator-=(const complex& other) {
                xy -= other.xy;
                return *this;
            }

            __device__ __forceinline__ complex& operator*=(const complex& other) {
                auto saved_x = xy.x;
                xy.x         = __hfma(xy.x   , other.xy.x, -xy.y * other.xy.y);
                xy.y         = __hfma(saved_x, other.xy.y,  xy.y * other.xy.x);
                return *this;
            }

            friend __device__ __forceinline__ complex operator*(const complex& a, const complex& b) {
                auto result = a;
                result *= b;
                return result;
            }

            friend __device__ __forceinline__ complex operator+(const complex& a, const complex& b) {
                return {a.xy + b.xy};
            }

            friend __device__ __forceinline__ complex operator-(const complex& a, const complex& b) {
                return {a.xy - b.xy};
            }

            friend __device__ __forceinline__ bool operator==(const complex& a, const complex& b) {
                return (a.xy == b.xy);
            }

            /// \internal
            __half2 xy;
        };
#endif

        // BP16
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
        template<>
        struct alignas(2 * sizeof(__nv_bfloat16)) complex<__nv_bfloat16> {
            using value_type        = __nv_bfloat16;
            complex()               = default;
            complex(const complex&) = default;
            complex(complex&&)      = default;

            __device__ __forceinline__ __host__ complex(const __nv_bfloat162& h): xy(h) {};
            __device__ __forceinline__ __host__ complex(__nv_bfloat16 re, __nv_bfloat16 im): xy(re, im) {}
            __device__ __forceinline__ __host__ complex(float re, float im): xy(__float2bfloat16_rn(re), __float2bfloat16_rn(im)) {}
            __device__ __forceinline__ __host__ complex(double re, double im): xy(__double2bfloat16(re), __double2bfloat16(im)) {}

            template<class K>
            __device__ __forceinline__ __host__ explicit constexpr complex(const complex<K>& other) :
                complex(__nv_bfloat16(other.real()), __nv_bfloat16(other.imag())) {
                static_assert(is_valid_conversion_to_from<__nv_bfloat16, K>::value, "Unsupported conversion from K to T");
            };

            __device__ __forceinline__ __host__ explicit complex(const complex<__half> & other):
                complex<__nv_bfloat16>(__float2bfloat16(__half2float(other.real())), __float2bfloat16(__half2float(other.imag()))) {};

            __device__ __forceinline__ __host__ value_type real() const { return xy.x; }
            __device__ __forceinline__ __host__ value_type imag() const { return xy.y; }
            __device__ __forceinline__ __host__ void       real(value_type re) { xy.x = re; }
            __device__ __forceinline__ __host__ void       imag(value_type im) { xy.y = im; }

            complex& operator=(const complex&) = default;
            complex& operator=(complex&&) = default;

            __device__ __forceinline__ __host__ complex& operator=(value_type re) {
                xy.x = re;
                xy.y = value_type(); // zero-initialized with default ctor. see https://en.cppreference.com/w/cpp/language/value_initialization.
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(value_type re) {
                xy.x += re;
                return *this;
            }

            __device__ __forceinline__ complex& operator-=(value_type re) {
                xy.x -= re;
                return *this;
            }

            __device__ __forceinline__ complex& operator*=(value_type re) {
                xy *= __nv_bfloat162 {re, re};
                return *this;
            }

            __device__ __forceinline__ complex& operator/=(value_type re) {
                xy /= __nv_bfloat162 {re, re};
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(const complex& other) {
                xy += other.xy;
                return *this;
            }

            __device__ __forceinline__ complex& operator-=(const complex& other) {
                xy -= other.xy;
                return *this;
            }

            __device__ __forceinline__ complex& operator*=(const complex& other) {
                auto saved_x = xy.x;
                xy.x         = __hfma(xy.x   , other.xy.x, -xy.y * other.xy.y);
                xy.y         = __hfma(saved_x, other.xy.y,  xy.y * other.xy.x);
                return *this;
            }

            friend __device__ __forceinline__ complex operator*(const complex& a, const complex& b) {
                auto result = a;
                result *= b;
                return result;
            }

            friend __device__ __forceinline__ complex operator+(const complex& a, const complex& b) {
                return {a.xy + b.xy};
            }

            friend __device__ __forceinline__ complex operator-(const complex& a, const complex& b) {
                return {a.xy - b.xy};
            }

            friend __device__ __forceinline__ bool operator==(const complex& a, const complex& b) {
                return (a.xy == b.xy);
            }

            /// \internal
            __nv_bfloat162 xy;
        };
#endif

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
        // For FFT computations, complex<half2> should be in RRII layout.
        template<>
        struct alignas(2 * sizeof(__half2)) complex<__half2> {
            using value_type        = __half2;
            complex()               = default;
            complex(const complex&) = default;
            complex(complex&&)      = default;

            __device__ __forceinline__ __host__ complex(value_type re,
                                                        value_type im)
                : x(re), y(im) {}

            __device__ __forceinline__ __host__ complex(double re, double im) {
                __half hre = __double2half(re);
                x = __half2(hre, hre);
                __half him = __double2half(im);
                y = __half2(him, him);
            }

            __device__ __forceinline__ __host__ complex(float re, float im)
                : x(__float2half2_rn(re)), y(__float2half2_rn(im)) {}

            __device__ __forceinline__
                __host__ explicit complex(const complex<double> &other) {

                __half hre = __double2half(other.real());
                x = __half2(hre, hre);
                __half him = __double2half(other.imag());
                y = __half2(him, him);
            }

            __device__ __forceinline__ __host__ explicit complex(const complex<float>& other):
                x(__float2half2_rn(other.real())), y(__float2half2_rn(other.imag())) {}

            __device__ __forceinline__ __host__ value_type real() const { return x; }
            __device__ __forceinline__ __host__ value_type imag() const { return y; }
            __device__ __forceinline__ __host__ void       real(value_type re) { x = re; }
            __device__ __forceinline__ __host__ void       imag(value_type im) { y = im; }

            complex& operator=(const complex&) = default;
            complex& operator=(complex&&) = default;

            __device__ __forceinline__ __host__ complex& operator=(value_type re) {
                x = re;
                y = value_type(); // zero-initialized with default ctor. see https://en.cppreference.com/w/cpp/language/value_initialization.
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(value_type re) {
                x += re;
                return *this;
            }

            __device__ __forceinline__ complex& operator-=(value_type re) {
                x -= re;
                return *this;
            }

            __device__ __forceinline__ complex& operator*=(value_type re) {
                x *= re;
                y *= re;
                return *this;
            }

            __device__ __forceinline__ complex& operator/=(value_type re) {
                x /= re;
                y /= re;
                return *this;
            }

            __device__ __forceinline__ complex& operator+=(const complex& other) {
                x = x + other.x;
                y = y + other.y;
                return *this;
            }

            __device__ __forceinline__ complex& operator-=(const complex& other) {
                x = x - other.x;
                y = y - other.y;
                return *this;
            }

            __device__ __forceinline__ complex& operator*=(const complex& other) {
                auto saved_x = x;
                x            = __hfma2(x, other.x, -y * other.y);
                y            = __hfma2(saved_x, other.y, y * other.x);
                return *this;
            }

            /// \internal
            value_type x, y;
        };
#endif

        // build complex<T> from complex<K>
        template<>
        template<>
        __host__ __device__ __forceinline__ complex<double>::complex(const complex<__half>& other):
            complex<double>(float(other.real()), float(other.imag())) {};

        template<>
        template<>
        __host__ __device__ __forceinline__ complex<double>::complex(const complex<__nv_bfloat16>& other):
            complex<double>(float(other.real()), float(other.imag())) {};

        template<>
        template<>
        __host__ __device__ __forceinline__ constexpr complex<float>::complex(const complex<double>& other):
            complex<float>(static_cast<float>(other.real()), static_cast<float>(other.imag())) {};

        // operator=
        template<>
        template<>
        __host__ __device__ __forceinline__
        complex<double>& complex<double>::operator=(const complex<__half>& other) {
            this->x = static_cast<float>(other.real()); // half->double requires intermediary conversion to float
            this->y = static_cast<float>(other.imag());
            return *this;
        }

        template<>
        template<>
        __host__ __device__ __forceinline__
        complex<double>& complex<double>::operator=(const complex<__nv_bfloat16>& other) {
            this->x = static_cast<float>(other.real()); // bf16->double requires intermediary conversion to float
            this->y = static_cast<float>(other.imag());
            return *this;
        }

        template<>
        template<>
        __host__ __device__ __forceinline__
        constexpr complex<float>& complex<float>::operator=(const complex<double>& other) {
            this->x = static_cast<float>(other.real()); // requires implicit cast
            this->y = static_cast<float>(other.imag());
            return *this;
        }
    } // namespace detail

    template<class T>
    using complex = typename detail::complex<T>;
} // namespace commondx

// Namespace wrapper
#include "commondx/detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_COMPLEX_TYPES_HPP
