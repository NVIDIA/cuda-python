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

    CUstream stream;
    check_cu(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate failed");

    // Persistent event for query/synchronize/record benchmarks
    CUevent event;
    check_cu(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "cuEventCreate failed");

    // Record and sync so the event starts in a completed state
    check_cu(cuEventRecord(event, stream), "cuEventRecord failed");
    check_cu(cuStreamSynchronize(stream), "cuStreamSynchronize failed");

    bench::BenchmarkSuite suite(options);

    // --- event_create_destroy ---
    {
        CUevent e;
        suite.run("event.event_create_destroy", [&]() {
            check_cu(cuEventCreate(&e, CU_EVENT_DISABLE_TIMING), "cuEventCreate failed");
            check_cu(cuEventDestroy(e), "cuEventDestroy failed");
        });
    }

    // --- event_record ---
    {
        suite.run("event.event_record", [&]() {
            check_cu(cuEventRecord(event, stream), "cuEventRecord failed");
        });
    }

    // Re-sync so event is in a known completed state after the record benchmark
    check_cu(cuStreamSynchronize(stream), "cuStreamSynchronize failed");

    {
        suite.run("event.event_query", [&]() {
            // Returns CUDA_SUCCESS if complete, CUDA_ERROR_NOT_READY if not
            cuEventQuery(event);
        });
    }

    // --- event_synchronize ---
    {
        suite.run("event.event_synchronize", [&]() {
            check_cu(cuEventSynchronize(event), "cuEventSynchronize failed");
        });
    }

    // Cleanup
    check_cu(cuEventDestroy(event), "cuEventDestroy failed");
    check_cu(cuStreamDestroy(stream), "cuStreamDestroy failed");
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    suite.write();

    return 0;
}
