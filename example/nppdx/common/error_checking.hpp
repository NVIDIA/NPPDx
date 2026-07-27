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

#ifndef NPPDX_EXAMPLE_COMMON_ERROR_CHECKING_HPP
#define NPPDX_EXAMPLE_COMMON_ERROR_CHECKING_HPP

#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdint>

#include <type_traits>

#include <nppdx.hpp>

#include "numeric.hpp"

namespace common {

    // ============================================================================
    // Value comparison utilities
    // ============================================================================

    // Check if a value is within tolerance of expected value
    template<typename T>
    inline bool near(T value, T expected, T tolerance = T(5)) {
        return std::abs(static_cast<int>(value) - static_cast<int>(expected)) <= static_cast<int>(tolerance);
    }

    // ============================================================================
    // Extended error statistics structure
    // ============================================================================

    struct ErrorStats {
        // Relative error metrics (for floating point)
        double relative_error = 0.0; // L2 norm-based relative error

        // Absolute error metrics (for integer types)
        int    max_abs_error    = 0;   // Maximum absolute error
        double avg_abs_error    = 0.0; // Average absolute error
        int    error_count      = 0;   // Count of errors above threshold
        double error_percentage = 0.0; // Percentage of errors above threshold

        // Pass/fail
        bool passed = false;
    };

    // Extended check_error that returns detailed statistics including absolute error
    // This unifies all error checking in one place
    template<typename ResultType, typename ReferenceType>
    ErrorStats check_error_detailed(const ResultType* data, const ReferenceType* reference, std::size_t total_length,
                                    int abs_error_threshold = 2, int max_abs_error_limit = 40,
                                    double avg_abs_error_limit = 3.0, const int total_batches = 1,
                                    const int correctness_batches = 1, bool print = false, bool verbose = false);

    struct ErrorCheckConfig {
        int abs_error_threshold = 2;
        int max_abs_error_limit = 40;
        double avg_abs_error_limit = 3.0;
        int total_batches = 1;
        int correctness_batches = 1;
        bool print = false;
        bool verbose = false;
    };

    template<typename ResultType, typename ReferenceType>
    ErrorStats check_error_detailed(const ResultType* data, const ReferenceType* reference, std::size_t total_length,
                                    const ErrorCheckConfig& config)
    {
        return check_error_detailed(data, reference, total_length,
            config.abs_error_threshold, config.max_abs_error_limit, config.avg_abs_error_limit,
            config.total_batches, config.correctness_batches, config.print, config.verbose);
    }

    // Legacy function signatures (backward compatibility)
    template<typename ResultType, typename ReferenceType>
    double check_error(const ResultType* data, const ReferenceType* reference, std::size_t total_length,
                       const int total_batches = 1, const int correctness_batches = 1, bool print = false,
                       bool verbose = false);

    // Convenience typedef for color conversion
    using ColorConversionStats = ErrorStats;

    // Compute error statistics for color conversion using unified check_error_detailed
    // Optionally prints results based on the print flag
    template<typename T = uint8_t>
    inline ColorConversionStats compute_conversion_stats(const T* input, const T* output, std::size_t total_elements,
                                                         int error_threshold = 2, int max_error_limit = 40,
                                                         double avg_error_limit = 3.0, bool print = true,
                                                         const char* test_name = "Conversion") {

        auto stats =
            check_error_detailed(output, input, total_elements, error_threshold, max_error_limit, avg_error_limit);

        // Print if requested
        if (print) {
            std::cout << "Max error: " << stats.max_abs_error << std::endl;
            std::cout << "Average error: " << stats.avg_abs_error << std::endl;
            if (stats.relative_error > 0.0) {
                std::cout << "Relative error: " << stats.relative_error << std::endl;
            }
            std::cout << "Elements with error > " << error_threshold << ": " << stats.error_percentage << "%"
                      << std::endl;
            std::cout << test_name << ": " << (stats.passed ? "PASSED" : "FAILED") << std::endl;
        }

        return stats;
    }

    // ============================================================================
    // Data validation - check for unreasonable values
    // ============================================================================

    // Generic function to check if data values appear reasonable (not mostly at extremes)
    // Works with any data format - checks if too many values are at min or max of range
    template<typename T>
    inline bool is_data_reasonable(const T* data, std::size_t size, T min_value, T max_value,
                                   double max_min_percentage = 70.0, double max_max_percentage = 70.0) {
        std::size_t min_count = 0;
        std::size_t max_count = 0;

        for (std::size_t i = 0; i < size; ++i) {
            if (data[i] == min_value)
                min_count++;
            if (data[i] == max_value)
                max_count++;
        }

        double min_percentage = (100.0 * min_count) / size;
        double max_percentage = (100.0 * max_count) / size;

        return (min_percentage < max_min_percentage && max_percentage < max_max_percentage);
    }

} // namespace common


#endif // NPPDX_TEST_COMMON_ERROR_CHECKING_HPP
