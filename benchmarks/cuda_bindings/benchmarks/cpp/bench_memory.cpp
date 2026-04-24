// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include "bench_support.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>


static void check_cu(CUresult status, const char* message) {
    if (status != CUDA_SUCCESS) {
        const char* error_name = nullptr;
        cuGetErrorName(status, &error_name);
        std::cerr << message << ": " << (error_name ? error_name : "unknown") << '\n';
        std::exit(1);
    }
}


static constexpr size_t ALLOC_SIZE = 1024;
static constexpr size_t COPY_SIZE = 8;


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

    // Pre-allocate device memory for memcpy benchmarks
    CUdeviceptr dst_dptr, src_dptr;
    check_cu(cuMemAlloc(&dst_dptr, COPY_SIZE), "cuMemAlloc failed");
    check_cu(cuMemAlloc(&src_dptr, COPY_SIZE), "cuMemAlloc failed");

    // Host buffers for memcpy
    uint8_t host_src[COPY_SIZE] = {};
    uint8_t host_dst[COPY_SIZE] = {};

    bench::BenchmarkSuite suite(options);

    // --- mem_alloc_free ---
    {
        CUdeviceptr ptr;
        suite.run("memory.mem_alloc_free", [&]() {
            check_cu(cuMemAlloc(&ptr, ALLOC_SIZE), "cuMemAlloc failed");
            check_cu(cuMemFree(ptr), "cuMemFree failed");
        });
    }

    // --- mem_alloc_async_free_async ---
    {
        CUdeviceptr ptr;
        suite.run("memory.mem_alloc_async_free_async", [&]() {
            check_cu(cuMemAllocAsync(&ptr, ALLOC_SIZE, stream), "cuMemAllocAsync failed");
            check_cu(cuMemFreeAsync(ptr, stream), "cuMemFreeAsync failed");
        });
    }

    check_cu(cuStreamSynchronize(stream), "cuStreamSynchronize failed");

    // --- memcpy_htod ---
    {
        suite.run("memory.memcpy_htod", [&]() {
            check_cu(cuMemcpyHtoD(dst_dptr, host_src, COPY_SIZE), "cuMemcpyHtoD failed");
        });
    }

    // --- memcpy_dtoh ---
    {
        suite.run("memory.memcpy_dtoh", [&]() {
            check_cu(cuMemcpyDtoH(host_dst, src_dptr, COPY_SIZE), "cuMemcpyDtoH failed");
        });
    }

    // --- memcpy_dtod ---
    {
        suite.run("memory.memcpy_dtod", [&]() {
            check_cu(cuMemcpyDtoD(dst_dptr, src_dptr, COPY_SIZE), "cuMemcpyDtoD failed");
        });
    }

    // Cleanup
    check_cu(cuMemFree(dst_dptr), "cuMemFree failed");
    check_cu(cuMemFree(src_dptr), "cuMemFree failed");
    check_cu(cuStreamDestroy(stream), "cuStreamDestroy failed");
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    suite.write();

    return 0;
}
