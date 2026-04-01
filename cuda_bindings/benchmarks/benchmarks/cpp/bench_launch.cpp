// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <nvrtc.h>

#include "bench_support.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>


static void check_cu(CUresult status, const char* message) {
    if (status != CUDA_SUCCESS) {
        const char* error_name = nullptr;
        cuGetErrorName(status, &error_name);
        std::cerr << message << ": " << (error_name ? error_name : "unknown") << '\n';
        std::exit(1);
    }
}

static void check_nvrtc(nvrtcResult status, const char* message) {
    if (status != NVRTC_SUCCESS) {
        std::cerr << message << ": " << nvrtcGetErrorString(status) << '\n';
        std::exit(1);
    }
}

static CUmodule compile_and_load(const char* source, CUdevice device) {
    int major = 0, minor = 0;
    check_cu(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
             "cuDeviceGetAttribute failed");
    check_cu(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
             "cuDeviceGetAttribute failed");

    nvrtcProgram prog;
    check_nvrtc(nvrtcCreateProgram(&prog, source, "benchmark_kernel.cu", 0, nullptr, nullptr),
                "nvrtcCreateProgram failed");

    std::string arch = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
    const char* opts[] = {"--fmad=false", arch.c_str()};
    nvrtcResult compile_result = nvrtcCompileProgram(prog, 2, opts);

    // Print log on failure
    if (compile_result != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size);
        nvrtcGetProgramLog(prog, log.data());
        std::cerr << "NVRTC compile failed:\n" << log.data() << '\n';
        std::exit(1);
    }

    size_t cubin_size = 0;
    check_nvrtc(nvrtcGetCUBINSize(prog, &cubin_size), "nvrtcGetCUBINSize failed");
    std::vector<char> cubin(cubin_size);
    check_nvrtc(nvrtcGetCUBIN(prog, cubin.data()), "nvrtcGetCUBIN failed");
    nvrtcDestroyProgram(&prog);

    CUmodule module;
    check_cu(cuModuleLoadData(&module, cubin.data()), "cuModuleLoadData failed");
    return module;
}


static const char* KERNEL_SOURCE = R"(
extern "C" __global__ void empty_kernel() { return; }
extern "C" __global__ void small_kernel(float *f) { *f = 0.0f; }

#define ITEM_PARAM(x, T) T x
#define REP1(x, T)   , ITEM_PARAM(x, T)
#define REP2(x, T)   REP1(x##0, T)   REP1(x##1, T)
#define REP4(x, T)   REP2(x##0, T)   REP2(x##1, T)
#define REP8(x, T)   REP4(x##0, T)   REP4(x##1, T)
#define REP16(x, T)  REP8(x##0, T)   REP8(x##1, T)

extern "C" __global__
void small_kernel_16_args(
    ITEM_PARAM(F, int*)
    REP1(A, int*)
    REP2(A, int*)
    REP4(A, int*)
    REP8(A, int*))
{ *F = 0; }
)";


int main(int argc, char** argv) {
    bench::Options options = bench::parse_args(argc, argv);

    // Setup
    check_cu(cuInit(0), "cuInit failed");

    CUdevice device;
    check_cu(cuDeviceGet(&device, 0), "cuDeviceGet failed");

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    check_cu(cuCtxCreate(&ctx, &ctxParams, 0, device), "cuCtxCreate failed");

    CUmodule module = compile_and_load(KERNEL_SOURCE, device);

    CUfunction empty_kernel, small_kernel, kernel_16_args;
    check_cu(cuModuleGetFunction(&empty_kernel, module, "empty_kernel"), "GetFunction failed");
    check_cu(cuModuleGetFunction(&small_kernel, module, "small_kernel"), "GetFunction failed");
    check_cu(cuModuleGetFunction(&kernel_16_args, module, "small_kernel_16_args"), "GetFunction failed");

    CUstream stream;
    check_cu(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate failed");

    // Allocate device memory for arguments
    CUdeviceptr float_ptr;
    check_cu(cuMemAlloc(&float_ptr, sizeof(float)), "cuMemAlloc failed");

    CUdeviceptr int_ptrs[16];
    for (int i = 0; i < 16; ++i) {
        check_cu(cuMemAlloc(&int_ptrs[i], sizeof(int)), "cuMemAlloc failed");
    }

    // Pre-pack kernel params for the pre-packed benchmark
    void* packed_16[16];
    for (int i = 0; i < 16; ++i) {
        packed_16[i] = &int_ptrs[i];
    }

    bench::BenchmarkSuite suite(options);

    // --- launch_empty_kernel ---
    {
        suite.run("launch.launch_empty_kernel", [&]() {
            check_cu(
                cuLaunchKernel(empty_kernel, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr),
                "cuLaunchKernel failed"
            );
        });
    }

    // --- launch_small_kernel ---
    {
        void* params[] = {&float_ptr};
        suite.run("launch.launch_small_kernel", [&]() {
            check_cu(
                cuLaunchKernel(small_kernel, 1, 1, 1, 1, 1, 1, 0, stream, params, nullptr),
                "cuLaunchKernel failed"
            );
        });
    }

    // --- launch_16_args ---
    {
        suite.run("launch.launch_16_args", [&]() {
            check_cu(
                cuLaunchKernel(kernel_16_args, 1, 1, 1, 1, 1, 1, 0, stream, packed_16, nullptr),
                "cuLaunchKernel failed"
            );
        });
    }

    // --- launch_16_args_pre_packed (same as above for C++ — no packing overhead) ---
    // In C++ the params are always pre-packed, so this is identical to launch_16_args.
    // We include it for naming parity with the Python benchmark.
    {
        suite.run("launch.launch_16_args_pre_packed", [&]() {
            check_cu(
                cuLaunchKernel(kernel_16_args, 1, 1, 1, 1, 1, 1, 0, stream, packed_16, nullptr),
                "cuLaunchKernel failed"
            );
        });
    }

    // Cleanup
    for (int i = 0; i < 16; ++i) {
        check_cu(cuMemFree(int_ptrs[i]), "cuMemFree failed");
    }
    check_cu(cuMemFree(float_ptr), "cuMemFree failed");
    check_cu(cuStreamDestroy(stream), "cuStreamDestroy failed");
    check_cu(cuModuleUnload(module), "cuModuleUnload failed");
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    suite.write();

    return 0;
}
