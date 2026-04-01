// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include "bench_support.hpp"

#include <cstdlib>
#include <iostream>


static void check_cu(CUresult status, const char* message) {
    if (status != CUDA_SUCCESS) {
        const char* error_name = nullptr;
        cuGetErrorName(status, &error_name);
        std::cerr << message << ": " << (error_name ? error_name : "unknown") << '\n';
        std::exit(1);
    }
}


int main(int argc, char** argv) {
    bench::Options options = bench::parse_args(argc, argv);
    if (options.benchmark_name.empty()) {
        options.benchmark_name = "cpp.ctx_device.ctx_get_current";
    }

    // Setup: init CUDA and create a context
    check_cu(cuInit(0), "cuInit failed");

    CUdevice device;
    check_cu(cuDeviceGet(&device, 0), "cuDeviceGet failed");

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    check_cu(cuCtxCreate(&ctx, &ctxParams, 0, device), "cuCtxCreate failed");

    CUcontext current_ctx = nullptr;

    // Run benchmark
    auto results = bench::run_benchmark(options, [&]() {
        check_cu(
            cuCtxGetCurrent(&current_ctx),
            "cuCtxGetCurrent failed"
        );
    });

    // Sanity check: the call actually returned our context
    if (current_ctx != ctx) {
        std::cerr << "unexpected: cuCtxGetCurrent returned a different context\n";
    }

    // Cleanup
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    // Output
    bench::print_summary(options.benchmark_name, results);

    if (!options.output_path.empty()) {
        bench::write_pyperf_json(options.output_path, options.benchmark_name, options.loops, results);
    }

    return 0;
}
