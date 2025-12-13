// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

#ifndef CUDA_CORE_STRIDED_COPY_ARGS_H
#define CUDA_CORE_STRIDED_COPY_ARGS_H

#include "impl/type_utils.h"

#if defined(_MSC_VER)
// For Visual Studio, use __restrict
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
// For GCC and Clang, use __restrict__
#define RESTRICT __restrict__
#else
// Fallback for other compilers, or if restrict is not supported
#define RESTRICT
#endif

namespace cuda_core
{
template <int N>
struct KernelArgs
{
    void *RESTRICT dst_ptr;
    const void *RESTRICT src_ptr;
    int64_t dst_shape[N];
    int64_t src_shape[N];
    int64_t dst_strides[N];
    int64_t src_strides[N];
    int64_t grid_arg;
};
} // namespace cuda_core

#endif // CUDA_CORE_STRIDED_COPY_ARGS_H
