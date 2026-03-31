// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace bench {

struct Options {
    std::uint64_t loops = 1000;
    std::uint64_t warmups = 5;
    std::uint64_t values = 20;
    std::uint64_t runs = 20;
    std::string output_path;
    std::string benchmark_name;
};

// A single run result: warmup values and timed values (seconds per loop)
struct RunResult {
    std::string date;
    double duration_sec;
    std::vector<double> warmup_values;  // seconds per loop
    std::vector<double> values;         // seconds per loop
};

inline Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--loops" && i + 1 < argc) {
            options.loops = std::strtoull(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--warmups" && i + 1 < argc) {
            options.warmups = std::strtoull(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--values" && i + 1 < argc) {
            options.values = std::strtoull(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--runs" && i + 1 < argc) {
            options.runs = std::strtoull(argv[++i], nullptr, 10);
            continue;
        }
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            options.output_path = argv[++i];
            continue;
        }
        if (arg == "--name" && i + 1 < argc) {
            options.benchmark_name = argv[++i];
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark [options]\n"
                      << "  --loops N       Loop iterations per value (default: 1000)\n"
                      << "  --warmups N     Warmup values per run (default: 5)\n"
                      << "  --values N      Timed values per run (default: 20)\n"
                      << "  --runs N        Number of runs (default: 20)\n"
                      << "  -o, --output F  Write pyperf-compatible JSON to file\n"
                      << "  --name S        Benchmark name (overrides default)\n";
            std::exit(0);
        }

        std::cerr << "Unknown argument: " << arg << '\n';
        std::exit(2);
    }
    return options;
}

inline std::string iso_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buf);
}

// Run a benchmark function. The function signature is: void fn() — one call = one operation.
// The harness calls fn() in a tight loop `loops` times per value.
template <typename Fn>
std::vector<RunResult> run_benchmark(const Options& options, Fn&& fn) {
    std::vector<RunResult> results;
    results.reserve(options.runs);

    for (std::uint64_t r = 0; r < options.runs; ++r) {
        RunResult run;
        run.date = iso_now();
        const auto run_start = std::chrono::steady_clock::now();

        // Warmups
        for (std::uint64_t w = 0; w < options.warmups; ++w) {
            const auto t0 = std::chrono::steady_clock::now();
            for (std::uint64_t i = 0; i < options.loops; ++i) {
                fn();
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(t1 - t0).count();
            run.warmup_values.push_back(elapsed / static_cast<double>(options.loops));
        }

        // Timed values
        for (std::uint64_t v = 0; v < options.values; ++v) {
            const auto t0 = std::chrono::steady_clock::now();
            for (std::uint64_t i = 0; i < options.loops; ++i) {
                fn();
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(t1 - t0).count();
            run.values.push_back(elapsed / static_cast<double>(options.loops));
        }

        const auto run_end = std::chrono::steady_clock::now();
        run.duration_sec = std::chrono::duration<double>(run_end - run_start).count();
        results.push_back(std::move(run));
    }

    return results;
}

inline void print_summary(const std::string& name, const std::vector<RunResult>& results) {
    // Collect all timed values
    std::vector<double> all_values;
    for (const auto& run : results) {
        for (double v : run.values) {
            all_values.push_back(v);
        }
    }
    if (all_values.empty()) return;

    double sum = 0;
    for (double v : all_values) sum += v;
    double mean = sum / static_cast<double>(all_values.size());

    double sq_sum = 0;
    for (double v : all_values) {
        double diff = v - mean;
        sq_sum += diff * diff;
    }
    double stdev = std::sqrt(sq_sum / static_cast<double>(all_values.size()));

    std::cout << name << ": Mean +- std dev: "
              << std::fixed << std::setprecision(0)
              << (mean * 1e9) << " ns +- "
              << (stdev * 1e9) << " ns\n";
}

// Escape a JSON string (minimal — no control chars expected)
inline std::string json_str(const std::string& s) {
    return "\"" + s + "\"";
}

inline void write_pyperf_json(
    const std::string& output_path,
    const std::string& name,
    std::uint64_t loops,
    const std::vector<RunResult>& results
) {
    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_path << '\n';
        std::exit(3);
    }

    out << std::setprecision(17);

    out << "{\"version\": \"1.0\", ";
    out << "\"metadata\": {";
    out << "\"name\": " << json_str(name) << ", ";
    out << "\"loops\": " << loops << ", ";
    out << "\"unit\": \"second\"";
    out << "}, ";

    out << "\"benchmarks\": [{\"runs\": [";

    for (std::size_t r = 0; r < results.size(); ++r) {
        const auto& run = results[r];
        if (r > 0) out << ", ";

        out << "{\"metadata\": {";
        out << "\"date\": " << json_str(run.date) << ", ";
        out << "\"duration\": " << run.duration_sec;
        out << "}, ";

        // Warmups: array of [loops, value] pairs
        out << "\"warmups\": [";
        for (std::size_t w = 0; w < run.warmup_values.size(); ++w) {
            if (w > 0) out << ", ";
            out << "[" << loops << ", " << run.warmup_values[w] << "]";
        }
        out << "], ";

        // Values
        out << "\"values\": [";
        for (std::size_t v = 0; v < run.values.size(); ++v) {
            if (v > 0) out << ", ";
            out << run.values[v];
        }
        out << "]}";
    }

    out << "]}]}\n";
}

}  // namespace bench
