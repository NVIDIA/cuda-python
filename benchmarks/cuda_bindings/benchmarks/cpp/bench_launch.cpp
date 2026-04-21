// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <nvrtc.h>

#include "bench_support.hpp"

#include <cstdint>
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
#define REP32(x, T)  REP16(x##0, T)  REP16(x##1, T)
#define REP64(x, T)  REP32(x##0, T)  REP32(x##1, T)
#define REP128(x, T) REP64(x##0, T)  REP64(x##1, T)
#define REP256(x, T) REP128(x##0, T) REP128(x##1, T)

template<size_t maxBytes>
struct KernelFunctionParam {
   unsigned char p[maxBytes];
};

extern "C" __global__
void small_kernel_16_args(
    ITEM_PARAM(F, int*)
    REP1(A, int*) REP2(A, int*) REP4(A, int*) REP8(A, int*))
{ *F = 0; }

extern "C" __global__
void small_kernel_256_args(
    ITEM_PARAM(F, int*)
    REP1(A, int*) REP2(A, int*) REP4(A, int*) REP8(A, int*)
    REP16(A, int*) REP32(A, int*) REP64(A, int*) REP128(A, int*))
{ *F = 0; }

extern "C" __global__
void small_kernel_512_args(
    ITEM_PARAM(F, int*)
    REP1(A, int*) REP2(A, int*) REP4(A, int*) REP8(A, int*)
    REP16(A, int*) REP32(A, int*) REP64(A, int*) REP128(A, int*)
    REP256(A, int*))
{ *F = 0; }

extern "C" __global__
void small_kernel_512_bools(
    ITEM_PARAM(F, bool)
    REP1(A, bool) REP2(A, bool) REP4(A, bool) REP8(A, bool)
    REP16(A, bool) REP32(A, bool) REP64(A, bool) REP128(A, bool)
    REP256(A, bool))
{ return; }

extern "C" __global__
void small_kernel_512_ints(
    ITEM_PARAM(F, int)
    REP1(A, int) REP2(A, int) REP4(A, int) REP8(A, int)
    REP16(A, int) REP32(A, int) REP64(A, int) REP128(A, int)
    REP256(A, int))
{ return; }

extern "C" __global__
void small_kernel_512_doubles(
    ITEM_PARAM(F, double)
    REP1(A, double) REP2(A, double) REP4(A, double) REP8(A, double)
    REP16(A, double) REP32(A, double) REP64(A, double) REP128(A, double)
    REP256(A, double))
{ return; }

extern "C" __global__
void small_kernel_512_chars(
    ITEM_PARAM(F, char)
    REP1(A, char) REP2(A, char) REP4(A, char) REP8(A, char)
    REP16(A, char) REP32(A, char) REP64(A, char) REP128(A, char)
    REP256(A, char))
{ return; }

extern "C" __global__
void small_kernel_512_longlongs(
    ITEM_PARAM(F, long long)
    REP1(A, long long) REP2(A, long long) REP4(A, long long) REP8(A, long long)
    REP16(A, long long) REP32(A, long long) REP64(A, long long) REP128(A, long long)
    REP256(A, long long))
{ return; }

extern "C" __global__
void small_kernel_2048B(KernelFunctionParam<2048> param) {}
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

    // Get all kernel handles
    auto get_func = [&](const char* name) {
        CUfunction f;
        check_cu(cuModuleGetFunction(&f, module, name), "GetFunction failed");
        return f;
    };

    CUfunction empty_kernel      = get_func("empty_kernel");
    CUfunction small_kernel      = get_func("small_kernel");
    CUfunction kernel_16_args    = get_func("small_kernel_16_args");
    CUfunction kernel_256_args   = get_func("small_kernel_256_args");
    CUfunction kernel_512_args   = get_func("small_kernel_512_args");
    CUfunction kernel_512_bools  = get_func("small_kernel_512_bools");
    CUfunction kernel_512_ints   = get_func("small_kernel_512_ints");
    CUfunction kernel_512_doubles = get_func("small_kernel_512_doubles");
    CUfunction kernel_512_chars  = get_func("small_kernel_512_chars");
    CUfunction kernel_512_longlongs = get_func("small_kernel_512_longlongs");
    CUfunction kernel_2048B      = get_func("small_kernel_2048B");

    CUstream stream;
    check_cu(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate failed");

    // Allocate device memory
    CUdeviceptr float_ptr;
    check_cu(cuMemAlloc(&float_ptr, sizeof(float)), "cuMemAlloc failed");

    CUdeviceptr int_ptrs[512];
    for (int i = 0; i < 512; ++i) {
        check_cu(cuMemAlloc(&int_ptrs[i], sizeof(int)), "cuMemAlloc failed");
    }

    // Pre-pack pointer params
    void* packed_16[16];
    for (int i = 0; i < 16; ++i)
        packed_16[i] = &int_ptrs[i];

    void* packed_256[256];
    for (int i = 0; i < 256; ++i)
        packed_256[i] = &int_ptrs[i];

    void* packed_512[512];
    for (int i = 0; i < 512; ++i)
        packed_512[i] = &int_ptrs[i];

    // Typed args for 512-arg benchmarks
    bool bool_args[512];
    void* bool_params[512];
    for (int i = 0; i < 512; ++i) { bool_args[i] = true; bool_params[i] = &bool_args[i]; }

    int int_args[512];
    void* int_params[512];
    for (int i = 0; i < 512; ++i) { int_args[i] = 123; int_params[i] = &int_args[i]; }

    double double_args[512];
    void* double_params[512];
    for (int i = 0; i < 512; ++i) { double_args[i] = 1.2345; double_params[i] = &double_args[i]; }

    char char_args[512];
    void* char_params[512];
    for (int i = 0; i < 512; ++i) { char_args[i] = 127; char_params[i] = &char_args[i]; }

    long long ll_args[512];
    void* ll_params[512];
    for (int i = 0; i < 512; ++i) { ll_args[i] = 9223372036854775806LL; ll_params[i] = &ll_args[i]; }

    // 2048-byte struct
    struct alignas(8) Struct2048B { unsigned char p[2048]; } struct_2048B = {};
    void* struct_params[] = {&struct_2048B};

    bench::BenchmarkSuite suite(options);

    suite.run("launch.launch_empty_kernel", [&]() {
        check_cu(cuLaunchKernel(empty_kernel, 1, 1, 1, 1, 1, 1, 0, stream, nullptr, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    {
        void* params[] = {&float_ptr};
        suite.run("launch.launch_small_kernel", [&]() {
            check_cu(cuLaunchKernel(small_kernel, 1, 1, 1, 1, 1, 1, 0, stream, params, nullptr),
                     "cuLaunchKernel failed");
        });
    }
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_16_args", [&]() {
        check_cu(cuLaunchKernel(kernel_16_args, 1, 1, 1, 1, 1, 1, 0, stream, packed_16, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_16_args_pre_packed", [&]() {
        check_cu(cuLaunchKernel(kernel_16_args, 1, 1, 1, 1, 1, 1, 0, stream, packed_16, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_256_args", [&]() {
        check_cu(cuLaunchKernel(kernel_256_args, 1, 1, 1, 1, 1, 1, 0, stream, packed_256, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_512_args", [&]() {
        check_cu(cuLaunchKernel(kernel_512_args, 1, 1, 1, 1, 1, 1, 0, stream, packed_512, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_512_args_pre_packed", [&]() {
        check_cu(cuLaunchKernel(kernel_512_args, 1, 1, 1, 1, 1, 1, 0, stream, packed_512, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_512_bools", [&]() {
        check_cu(cuLaunchKernel(kernel_512_bools, 1, 1, 1, 1, 1, 1, 0, stream, bool_params, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_512_ints", [&]() {
        check_cu(cuLaunchKernel(kernel_512_ints, 1, 1, 1, 1, 1, 1, 0, stream, int_params, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_512_doubles", [&]() {
        check_cu(cuLaunchKernel(kernel_512_doubles, 1, 1, 1, 1, 1, 1, 0, stream, double_params, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_512_bytes", [&]() {
        check_cu(cuLaunchKernel(kernel_512_chars, 1, 1, 1, 1, 1, 1, 0, stream, char_params, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_512_longlongs", [&]() {
        check_cu(cuLaunchKernel(kernel_512_longlongs, 1, 1, 1, 1, 1, 1, 0, stream, ll_params, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    suite.run("launch.launch_2048b", [&]() {
        check_cu(cuLaunchKernel(kernel_2048B, 1, 1, 1, 1, 1, 1, 0, stream, struct_params, nullptr),
                 "cuLaunchKernel failed");
    });
    check_cu(cuStreamSynchronize(stream), "sync");

    // Cleanup
    for (int i = 0; i < 512; ++i) {
        check_cu(cuMemFree(int_ptrs[i]), "cuMemFree failed");
    }
    check_cu(cuMemFree(float_ptr), "cuMemFree failed");
    check_cu(cuStreamDestroy(stream), "cuStreamDestroy failed");
    check_cu(cuModuleUnload(module), "cuModuleUnload failed");
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    suite.write();

    return 0;
}
