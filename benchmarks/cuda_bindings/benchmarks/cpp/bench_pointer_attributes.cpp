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

    // Setup: init CUDA, allocate memory
    check_cu(cuInit(0), "cuInit failed");

    CUdevice device;
    check_cu(cuDeviceGet(&device, 0), "cuDeviceGet failed");

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    check_cu(cuCtxCreate(&ctx, &ctxParams, 0, device), "cuCtxCreate failed");

    CUdeviceptr ptr;
    check_cu(cuMemAlloc(&ptr, 1 << 18), "cuMemAlloc failed");

    bench::BenchmarkSuite suite(options);

    // --- pointer_get_attribute ---
    {
        unsigned int memory_type = 0;
        suite.run("pointer_attributes.pointer_get_attribute", [&]() {
            check_cu(
                cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptr),
                "cuPointerGetAttribute failed"
            );
        });
    }

    // --- pointer_get_attributes ---
    {
        unsigned int memory_type = 0;
        CUdeviceptr dev_ptr_out = 0;
        void* host_ptr_out = nullptr;
        unsigned long long buffer_id = 0;

        CUpointer_attribute attrs[4] = {
            CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
            CU_POINTER_ATTRIBUTE_HOST_POINTER,
            CU_POINTER_ATTRIBUTE_BUFFER_ID,
        };
        void* data[4] = {&memory_type, &dev_ptr_out, &host_ptr_out, &buffer_id};

        suite.run("pointer_attributes.pointer_get_attributes", [&]() {
            check_cu(
                cuPointerGetAttributes(4, attrs, data, ptr),
                "cuPointerGetAttributes failed"
            );
        });
    }

    // Cleanup
    check_cu(cuMemFree(ptr), "cuMemFree failed");
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    suite.write();

    return 0;
}
