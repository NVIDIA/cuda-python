// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <nvrtc.h>

#include "bench_support.hpp"

#include <cstdlib>
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

static std::vector<char> compile_cubin(const char* source, CUdevice device) {
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

    return cubin;
}


static const char* KERNEL_SOURCE = R"(
extern "C" __global__ void empty_kernel() { return; }
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

    std::vector<char> cubin = compile_cubin(KERNEL_SOURCE, device);

    // Load a persistent module + function for get_function / get_attribute benchmarks
    CUmodule persistent_module;
    check_cu(cuModuleLoadData(&persistent_module, cubin.data()), "cuModuleLoadData failed");

    CUfunction function;
    check_cu(cuModuleGetFunction(&function, persistent_module, "empty_kernel"),
             "cuModuleGetFunction failed");

    bench::BenchmarkSuite suite(options);

    // --- module_load_unload ---
    {
        CUmodule m;
        suite.run("module.module_load_unload", [&]() {
            check_cu(cuModuleLoadData(&m, cubin.data()), "cuModuleLoadData failed");
            check_cu(cuModuleUnload(m), "cuModuleUnload failed");
        });
    }

    // --- module_get_function ---
    {
        CUfunction f;
        suite.run("module.module_get_function", [&]() {
            check_cu(cuModuleGetFunction(&f, persistent_module, "empty_kernel"),
                     "cuModuleGetFunction failed");
        });
    }

    // --- func_get_attribute ---
    {
        int value = 0;
        suite.run("module.func_get_attribute", [&]() {
            check_cu(cuFuncGetAttribute(&value, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function),
                     "cuFuncGetAttribute failed");
        });
    }

    // Cleanup
    check_cu(cuModuleUnload(persistent_module), "cuModuleUnload failed");
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    suite.write();

    return 0;
}
