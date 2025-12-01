// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

#ifndef CUDA_CORE_STRIDED_COPY_IMPL_TYPE_UTILS_H_
#define CUDA_CORE_STRIDED_COPY_IMPL_TYPE_UTILS_H_

namespace cuda_core
{
using int32_t = int;
using uint32_t = unsigned int;
using int64_t = long long int;
using uint64_t = unsigned long long int;
static_assert(sizeof(int32_t) == 4, "int32_t must be 4 bytes");
static_assert(sizeof(uint32_t) == 4, "uint32_t must be 4 bytes");
static_assert(sizeof(int64_t) == 8, "int64_t must be 8 bytes");
static_assert(sizeof(uint64_t) == 8, "uint64_t must be 8 bytes");

// Use a struct to represent type of element so that we don't rely
// on actual representation of the type, available arithmetic etc.
template <int n_bytes>
struct alignas(n_bytes) opaque_t
{
    char data[n_bytes];
};

static_assert(sizeof(opaque_t<1>) == 1, "opaque_t<1> must be 1 byte");
static_assert(sizeof(opaque_t<2>) == 2, "opaque_t<2> must be 2 bytes");
static_assert(sizeof(opaque_t<4>) == 4, "opaque_t<4> must be 4 bytes");
static_assert(sizeof(opaque_t<8>) == 8, "opaque_t<8> must be 8 bytes");
static_assert(sizeof(opaque_t<16>) == 16, "opaque_t<16> must be 16 bytes");

static_assert(alignof(opaque_t<1>) == alignof(unsigned char),
              "opaque_t<1> must be 1 byte");
static_assert(alignof(opaque_t<2>) == alignof(unsigned short),
              "opaque_t<2> must be 2 bytes");
static_assert(alignof(opaque_t<4>) == alignof(unsigned int),
              "opaque_t<4> must be 4 bytes");
static_assert(alignof(opaque_t<8>) == alignof(unsigned long long int),
              "opaque_t<8> must be 8 bytes");
#ifdef __CUDA_ARCH__
static_assert(alignof(opaque_t<16>) == 16, "opaque_t<16> must be 16 bytes");
#endif
} // namespace cuda_core

#endif // CUDA_CORE_STRIDED_COPY_IMPL_TYPE_UTILS_H_
