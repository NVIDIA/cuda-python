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
        options.benchmark_name = "cpp.pointer_attributes.pointer_get_attribute";
    }

    // Setup: init CUDA, allocate memory
    check_cu(cuInit(0), "cuInit failed");

    CUdevice device;
    check_cu(cuDeviceGet(&device, 0), "cuDeviceGet failed");

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    check_cu(cuCtxCreate(&ctx, &ctxParams, 0, device), "cuCtxCreate failed");

    CUdeviceptr ptr;
    check_cu(cuMemAlloc(&ptr, 1 << 18), "cuMemAlloc failed");

    unsigned int memory_type = 0;

    // Run benchmark
    auto results = bench::run_benchmark(options, [&]() {
        check_cu(
            cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr),
            "cuPointerGetAttribute failed"
        );
    });

    // Sanity check: the call actually did something
    if (memory_type == 0) {
        std::cerr << "unexpected memory_type=0\n";
    }

    // Cleanup
    check_cu(cuMemFree(ptr), "cuMemFree failed");
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    // Output
    bench::print_summary(options.benchmark_name, results);

    if (!options.output_path.empty()) {
        bench::write_pyperf_json(options.output_path, options.benchmark_name, options.loops, results);
    }

    return 0;
}
