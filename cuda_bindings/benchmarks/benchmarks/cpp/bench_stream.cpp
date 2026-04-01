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

    // Setup
    check_cu(cuInit(0), "cuInit failed");

    CUdevice device;
    check_cu(cuDeviceGet(&device, 0), "cuDeviceGet failed");

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    check_cu(cuCtxCreate(&ctx, &ctxParams, 0, device), "cuCtxCreate failed");

    // Persistent stream for query/synchronize benchmarks
    CUstream stream;
    check_cu(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate failed");

    bench::BenchmarkSuite suite(options);

    // --- stream_create_destroy ---
    {
        CUstream s;
        suite.run("stream.stream_create_destroy", [&]() {
            check_cu(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING), "cuStreamCreate failed");
            check_cu(cuStreamDestroy(s), "cuStreamDestroy failed");
        });
    }

    // --- stream_query ---
    {
        suite.run("stream.stream_query", [&]() {
            // cuStreamQuery returns CUDA_SUCCESS if stream is idle,
            // CUDA_ERROR_NOT_READY if busy — both are valid here.
            cuStreamQuery(stream);
        });
    }

    // --- stream_synchronize ---
    {
        suite.run("stream.stream_synchronize", [&]() {
            check_cu(cuStreamSynchronize(stream), "cuStreamSynchronize failed");
        });
    }

    // Cleanup
    check_cu(cuStreamDestroy(stream), "cuStreamDestroy failed");
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    suite.write();

    return 0;
}
