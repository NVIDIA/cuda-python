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

    // Setup: init CUDA and create a context
    check_cu(cuInit(0), "cuInit failed");

    CUdevice device;
    check_cu(cuDeviceGet(&device, 0), "cuDeviceGet failed");

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    check_cu(cuCtxCreate(&ctx, &ctxParams, 0, device), "cuCtxCreate failed");

    bench::BenchmarkSuite suite(options);

    // --- ctx_get_current ---
    {
        CUcontext current_ctx = nullptr;
        suite.run("ctx_device.ctx_get_current", [&]() {
            check_cu(cuCtxGetCurrent(&current_ctx), "cuCtxGetCurrent failed");
        });
    }

    // --- ctx_set_current ---
    {
        suite.run("ctx_device.ctx_set_current", [&]() {
            check_cu(cuCtxSetCurrent(ctx), "cuCtxSetCurrent failed");
        });
    }

    // --- ctx_get_device ---
    {
        CUdevice dev;
        suite.run("ctx_device.ctx_get_device", [&]() {
            check_cu(cuCtxGetDevice(&dev), "cuCtxGetDevice failed");
        });
    }

    // --- device_get ---
    {
        CUdevice dev;
        suite.run("ctx_device.device_get", [&]() {
            check_cu(cuDeviceGet(&dev, 0), "cuDeviceGet failed");
        });
    }

    // --- device_get_attribute ---
    {
        int value = 0;
        suite.run("ctx_device.device_get_attribute", [&]() {
            check_cu(
                cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
                "cuDeviceGetAttribute failed"
            );
        });
    }

    // Cleanup
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    // Write all results
    suite.write();

    return 0;
}
