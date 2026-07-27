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

#include "error_checking.hpp"

#include <iomanip>
#include <tuple>
#include <numeric>
#include <cassert>
#include <random>

namespace common {

    template <typename StreamT>
    class TempFormatStream {
        StreamT& stream;
        std::ios_base::fmtflags original_flags;
        std::streamsize original_precision;
        std::streamsize original_width;

    public:
        TempFormatStream(StreamT& stream) : stream(stream) {
            original_flags = stream.flags();
            original_precision = stream.precision();
            original_width = stream.width();
        }

        StreamT& get() const {
            return stream;
        }

        ~TempFormatStream() {
            stream.flags(original_flags);
            stream.precision(original_precision);
            stream.width(original_width);
        }
    };

    template <typename StreamT>
    TempFormatStream(StreamT& stream) -> TempFormatStream<StreamT>;

    // This function computes the total relative error between any
    // two sequences of arbitrary precisions.
    // @pre: Sequences are of the same length
    template<typename ResultType, typename ReferenceType>
    auto check_batch_error(const ResultType* data, const ReferenceType* reference, const size_t length, bool print,
                           bool verbose) {
        using std::abs;
        using std::sqrt;

        double eps = 1e-200;

        double tot_error_sq    = 0;
        double tot_norm_sq     = 0;
        double tot_ind_rel_err = 0;
        double max_ind_rel_err = 0;

        // Use double for error computation
        using error_type = double;

        if (print && verbose) {
            std::cout << "Idx, Val, RefVal, RelError" << '\n';
        }

        for (size_t i = 0; i < length; ++i) {
            double val = static_cast<double>(data[i]);
            double ref = static_cast<double>(reference[i]);

            double aref      = std::abs(ref);
            double diff      = std::abs(ref - val);
            double rel_error = diff / (aref + eps);

            // Individual relative error
            tot_ind_rel_err += rel_error;

            // Maximum relative error
            max_ind_rel_err = std::max(max_ind_rel_err, rel_error);

            // Total relative error
            tot_error_sq += diff * diff;
            tot_norm_sq += aref * aref;

            if (print && verbose) {
                TempFormatStream temp_cout(std::cout);
                temp_cout.get() << std::scientific << std::setprecision(5) << i << ":\t" << val << "\t" << ref << "\t"
                          << rel_error << "\n";
            }
        }

        if (print) {
            double ave_rel_err = tot_ind_rel_err / double(length);
            TempFormatStream temp_cout(std::cout);
            temp_cout.get() << std::scientific << std::setprecision(5)
                      << "Single batch vector reference norm: " << sqrt(tot_norm_sq) << '\n'
                      << "Average relative error: " << ave_rel_err << '\n'
                      << "Maximum relative error: " << max_ind_rel_err << '\n';
        }

        return std::make_tuple(tot_error_sq, tot_norm_sq);
    }

    // Extended error checking with both relative and absolute metrics
    template<typename ResultType, typename ReferenceType>
    ErrorStats check_error_detailed(const ResultType* data, const ReferenceType* reference, std::size_t total_length,
                                    int abs_error_threshold, int max_abs_error_limit, double avg_abs_error_limit,
                                    const int total_batches, const int correctness_batches, bool print, bool verbose) {
        using std::abs;
        using std::sqrt;

        double     eps = 1e-200;
        ErrorStats stats {};

        assert(total_batches >= correctness_batches);
        assert(total_length % total_batches == 0);
        assert(total_batches >= 1);

        const auto length = total_length / total_batches;

        // Generate a sequence of batch indices
        std::vector<int> batch_indices(total_batches);
        std::iota(batch_indices.begin(), batch_indices.end(), 0);

        // Shuffle those indices
        std::random_device rd;
        std::mt19937       g(rd());
        std::shuffle(batch_indices.begin(), batch_indices.end(), g);

        batch_indices.resize(correctness_batches);

        double tot_error_sq    = 0.;
        double tot_norm_sq     = 0.;
        double total_abs_error = 0.0;

        for (const auto batch_idx : batch_indices) {
            const auto offset = batch_idx * length;

            // Compute both relative and absolute errors
            auto [batch_error_sq, batch_norm_sq] =
                check_batch_error(data + offset, reference + offset, length, print, verbose);
            tot_error_sq += batch_error_sq;
            tot_norm_sq += batch_norm_sq;

            // Compute absolute error metrics
            for (size_t i = 0; i < length; ++i) {
                int abs_error = std::abs(static_cast<int>(data[offset + i]) - static_cast<int>(reference[offset + i]));
                stats.max_abs_error = std::max(stats.max_abs_error, abs_error);
                total_abs_error += abs_error;
                if (abs_error > abs_error_threshold) {
                    stats.error_count++;
                }
            }
        }

        // Compute final statistics
        stats.relative_error   = sqrt(tot_error_sq / (tot_norm_sq + eps));
        stats.avg_abs_error    = total_abs_error / (correctness_batches * length);
        stats.error_percentage = (100.0 * stats.error_count) / (correctness_batches * length);
        stats.passed = (stats.max_abs_error <= max_abs_error_limit && stats.avg_abs_error <= avg_abs_error_limit);

        return stats;
    }

    // Legacy function for backward compatibility
    template<typename ResultType, typename ReferenceType>
    double check_error(const ResultType* data, const ReferenceType* reference, std::size_t total_length,
                       const int total_batches, const int correctness_batches, bool print, bool verbose) {
        double eps = 1e-200;

        assert(total_batches >= correctness_batches);
        assert(total_length % total_batches == 0);

        const auto length = total_length / total_batches;
        // Generate a sequence of batch indices
        std::vector<int> batch_indices(total_batches);
        std::iota(batch_indices.begin(), batch_indices.end(), 0);

        // Shuffle those indices
        std::random_device rd;
        std::mt19937       g(rd());
        std::shuffle(batch_indices.begin(), batch_indices.end(), g);

        batch_indices.resize(correctness_batches);
        double tot_error_sq = 0.;
        double tot_norm_sq  = 0.;

        for (const auto batch_idx : batch_indices) {
            const auto offset = batch_idx * length;
            auto [batch_error_sq, batch_norm_sq] =
                check_batch_error(data + offset, reference + offset, length, print, verbose);
            tot_error_sq += batch_error_sq;
            tot_norm_sq += batch_norm_sq;
        }

        double tot_rel_err = sqrt(tot_error_sq / (tot_norm_sq + eps));

        return tot_rel_err;
    }

#define NPPDX_DETAIL_CHECK_ERROR(type1, type2)                                                                    \
    template double check_error<type1, type2>(const type1* data, const type2* ref, const std::size_t length,      \
                                              const int total_batches, const int correctness_batches, bool print, \
                                              bool verbose);

#define NPPDX_DETAIL_CHECK_ERROR_DETAILED(type1, type2)                                                              \
    template ErrorStats check_error_detailed<type1, type2>(                                                          \
        const type1* data, const type2* ref, const std::size_t length, int abs_error_threshold,                      \
        int max_abs_error_limit, double avg_abs_error_limit, const int total_batches, const int correctness_batches, \
        bool print, bool verbose);

    NPPDX_DETAIL_CHECK_ERROR(float, float)
    NPPDX_DETAIL_CHECK_ERROR(float, double)
    NPPDX_DETAIL_CHECK_ERROR(double, double)

    // Add uint8_t instantiations for color conversion checking
    NPPDX_DETAIL_CHECK_ERROR(uint8_t, uint8_t)
    NPPDX_DETAIL_CHECK_ERROR(uint16_t, uint16_t)

    // Detailed error checking instantiations
    NPPDX_DETAIL_CHECK_ERROR_DETAILED(float, float)
    NPPDX_DETAIL_CHECK_ERROR_DETAILED(double, double)
    NPPDX_DETAIL_CHECK_ERROR_DETAILED(uint8_t, uint8_t)
    NPPDX_DETAIL_CHECK_ERROR_DETAILED(uint16_t, uint16_t)

#undef NPPDX_DETAIL_CHECK_ERROR
#undef NPPDX_DETAIL_CHECK_ERROR_DETAILED

} // namespace common
