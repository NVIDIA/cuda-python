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


static const char* KERNEL_SOURCE = R"(extern "C" __global__ void empty_kernel() { return; })";
static const char* PROGRAM_NAME = "benchmark_kernel.cu";

static constexpr int NUM_HEADERS = 100;


int main(int argc, char** argv) {
    bench::Options options = bench::parse_args(argc, argv);

    // Setup: need CUDA init to query compute capability for compile options
    check_cu(cuInit(0), "cuInit failed");

    CUdevice device;
    check_cu(cuDeviceGet(&device, 0), "cuDeviceGet failed");

    int major = 0, minor = 0;
    check_cu(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
             "cuDeviceGetAttribute failed");
    check_cu(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
             "cuDeviceGetAttribute failed");

    std::string arch = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
    const char* compile_opts[] = {"--fmad=false", arch.c_str()};

    // Pre-build 100 empty headers
    std::vector<std::string> header_name_strs(NUM_HEADERS);
    std::vector<const char*> header_names(NUM_HEADERS);
    std::vector<const char*> header_sources(NUM_HEADERS);
    static const char* empty_header = "// empty";
    for (int i = 0; i < NUM_HEADERS; ++i) {
        header_name_strs[i] = "header_" + std::to_string(i) + ".cuh";
        header_names[i] = header_name_strs[i].c_str();
        header_sources[i] = empty_header;
    }

    bench::BenchmarkSuite suite(options);

    // --- nvrtc_create_program (no headers) ---
    {
        nvrtcProgram prog;
        suite.run("nvrtc.nvrtc_create_program", [&]() {
            check_nvrtc(
                nvrtcCreateProgram(&prog, KERNEL_SOURCE, PROGRAM_NAME, 0, nullptr, nullptr),
                "nvrtcCreateProgram failed"
            );
            check_nvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram failed");
        });
    }

    // --- nvrtc_create_program_100_headers ---
    {
        nvrtcProgram prog;
        suite.run("nvrtc.nvrtc_create_program_100_headers", [&]() {
            check_nvrtc(
                nvrtcCreateProgram(&prog, KERNEL_SOURCE, PROGRAM_NAME,
                                   NUM_HEADERS, header_sources.data(), header_names.data()),
                "nvrtcCreateProgram failed"
            );
            check_nvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram failed");
        });
    }

    // --- nvrtc_compile_program ---
    {
        nvrtcProgram prog;
        std::uint64_t compile_loops = std::min(options.loops, static_cast<std::uint64_t>(10));
        suite.run("nvrtc.nvrtc_compile_program", compile_loops, [&]() {
            check_nvrtc(
                nvrtcCreateProgram(&prog, KERNEL_SOURCE, PROGRAM_NAME, 0, nullptr, nullptr),
                "nvrtcCreateProgram failed"
            );
            check_nvrtc(
                nvrtcCompileProgram(prog, 2, compile_opts),
                "nvrtcCompileProgram failed"
            );
            check_nvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram failed");
        });
    }

    suite.write();

    return 0;
}
