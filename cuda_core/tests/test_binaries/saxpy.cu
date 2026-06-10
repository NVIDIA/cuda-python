// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>

template<typename T>
__global__ void saxpy(const T a, const T* x, const T* y, T* out, size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
        out[tid] = a * x[tid] + y[tid];
    }
}
