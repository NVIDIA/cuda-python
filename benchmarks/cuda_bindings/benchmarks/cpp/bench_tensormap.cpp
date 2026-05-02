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

    // Setup: init CUDA, allocate device memory
    check_cu(cuInit(0), "cuInit failed");

    CUdevice device;
    check_cu(cuDeviceGet(&device, 0), "cuDeviceGet failed");

    CUcontext ctx;
    CUctxCreateParams ctxParams = {};
    check_cu(cuCtxCreate(&ctx, &ctxParams, 0, device), "cuCtxCreate failed");

    CUdeviceptr ptr;
    check_cu(cuMemAlloc(&ptr, 1 << 20), "cuMemAlloc failed");

    // CUtensorMap must be 64-byte aligned.
    alignas(64) CUtensorMap tensor_map{};

    bench::BenchmarkSuite suite(options);

    // --- tensor_map_encode_tiled ---
    {
        const cuuint64_t global_dim[2] = {128ull, 128ull};
        const cuuint64_t global_strides[1] = {128ull * 4ull};
        const cuuint32_t box_dim[2] = {64u, 64u};
        const cuuint32_t element_strides[2] = {1u, 1u};

        CUresult probe = cuTensorMapEncodeTiled(
            &tensor_map,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2u,
            reinterpret_cast<void*>(ptr),
            global_dim,
            global_strides,
            box_dim,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (probe == CUDA_SUCCESS) {
            suite.run("tensormap.tensor_map_encode_tiled", [&]() {
                check_cu(
                    cuTensorMapEncodeTiled(
                        &tensor_map,
                        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                        2u,
                        reinterpret_cast<void*>(ptr),
                        global_dim,
                        global_strides,
                        box_dim,
                        element_strides,
                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                        CU_TENSOR_MAP_SWIZZLE_NONE,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                    ),
                    "cuTensorMapEncodeTiled failed"
                );
            });
        } else {
            const char* err_name = nullptr;
            cuGetErrorName(probe, &err_name);
            std::cerr << "Skipping tensormap.tensor_map_encode_tiled: "
                      << (err_name ? err_name : "unknown") << '\n';
        }
    }

    // --- tensor_map_encode_im2col ---
    {
        const cuuint64_t global_dim[3] = {32ull, 64ull, 64ull};
        const cuuint64_t global_strides[2] = {32ull * 2ull, 32ull * 64ull * 2ull};
        const int pixel_box_lower[1] = {0};
        const int pixel_box_upper[1] = {0};
        const cuuint32_t element_strides[3] = {1u, 1u, 1u};

        CUresult probe = cuTensorMapEncodeIm2col(
            &tensor_map,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            3u,
            reinterpret_cast<void*>(ptr),
            global_dim,
            global_strides,
            pixel_box_lower,
            pixel_box_upper,
            32u,
            32u,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (probe == CUDA_SUCCESS) {
            suite.run("tensormap.tensor_map_encode_im2col", [&]() {
                check_cu(
                    cuTensorMapEncodeIm2col(
                        &tensor_map,
                        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                        3u,
                        reinterpret_cast<void*>(ptr),
                        global_dim,
                        global_strides,
                        pixel_box_lower,
                        pixel_box_upper,
                        32u,
                        32u,
                        element_strides,
                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                        CU_TENSOR_MAP_SWIZZLE_NONE,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                    ),
                    "cuTensorMapEncodeIm2col failed"
                );
            });
        } else {
            const char* err_name = nullptr;
            cuGetErrorName(probe, &err_name);
            std::cerr << "Skipping tensormap.tensor_map_encode_im2col: "
                      << (err_name ? err_name : "unknown") << '\n';
        }
    }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
    // --- tensor_map_encode_im2col_wide ---
    {
        const cuuint64_t global_dim[3] = {32ull, 64ull, 64ull};
        const cuuint64_t global_strides[2] = {32ull * 2ull, 32ull * 64ull * 2ull};
        const cuuint32_t element_strides[3] = {1u, 1u, 1u};

        CUresult probe = cuTensorMapEncodeIm2colWide(
            &tensor_map,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            3u,
            reinterpret_cast<void*>(ptr),
            global_dim,
            global_strides,
            0,
            0,
            32u,
            32u,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_IM2COL_WIDE_MODE_W,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );

        if (probe == CUDA_SUCCESS) {
            suite.run("tensormap.tensor_map_encode_im2col_wide", [&]() {
                check_cu(
                    cuTensorMapEncodeIm2colWide(
                        &tensor_map,
                        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                        3u,
                        reinterpret_cast<void*>(ptr),
                        global_dim,
                        global_strides,
                        0,
                        0,
                        32u,
                        32u,
                        element_strides,
                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                        CU_TENSOR_MAP_IM2COL_WIDE_MODE_W,
                        CU_TENSOR_MAP_SWIZZLE_128B,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
                    ),
                    "cuTensorMapEncodeIm2colWide failed"
                );
            });
        } else {
            const char* err_name = nullptr;
            cuGetErrorName(probe, &err_name);
            std::cerr << "Skipping tensormap.tensor_map_encode_im2col_wide: "
                      << (err_name ? err_name : "unknown") << '\n';
        }
    }
#else
    std::cerr << "Skipping tensormap.tensor_map_encode_im2col_wide: "
                 "built against CUDA_VERSION < 12080.\n";
#endif

    // Cleanup
    check_cu(cuMemFree(ptr), "cuMemFree failed");
    check_cu(cuCtxDestroy(ctx), "cuCtxDestroy failed");

    suite.write();

    return 0;
}
