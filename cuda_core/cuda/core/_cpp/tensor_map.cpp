// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor_map_cccl.h"

#include <string.h>

#include <algorithm>
#include <exception>

#if defined(__has_include)
// Older CTK releases do not ship <cuda/tma>. When it is unavailable we keep
// the CCCL helper compiled out and fall back to the direct driver path.
#  if __has_include(<cuda/tma>)
#    include <cuda/tma>
#    define CUDA_CORE_HAS_CUDA_TMA 1
#  else
#    define CUDA_CORE_HAS_CUDA_TMA 0
#  endif
#  if __has_include("dlpack.h")
#    include "dlpack.h"
#    define CUDA_CORE_HAS_DLPACK_H 1
#  elif __has_include(<dlpack/dlpack.h>)
#    include <dlpack/dlpack.h>
#    define CUDA_CORE_HAS_DLPACK_H 1
#  else
#    define CUDA_CORE_HAS_DLPACK_H 0
#  endif
#else
#  define CUDA_CORE_HAS_CUDA_TMA 0
#  define CUDA_CORE_HAS_DLPACK_H 0
#endif

static inline void cuda_core_write_err(char* err, size_t cap, const char* msg) noexcept
{
  if (!err || cap == 0)
    return;
  if (!msg)
  {
    err[0] = '\0';
    return;
  }
  size_t n = ::strlen(msg);
  if (n >= cap)
    n = cap - 1;
  ::memcpy(err, msg, n);
  err[n] = '\0';
}

int cuda_core_cccl_make_tma_descriptor_tiled(
  void* out_tensor_map,
  void* data,
  int device_type,
  int device_id,
  int ndim,
  const int64_t* shape,
  const int64_t* strides,
  uint8_t dtype_code,
  uint8_t dtype_bits,
  uint16_t dtype_lanes,
  const int* box_sizes,
  const int* elem_strides,
  int interleave_layout,
  int swizzle,
  int l2_fetch_size,
  int oob_fill,
  char* err,
  size_t err_cap) noexcept
{
#if !(CUDA_CORE_HAS_CUDA_TMA && CUDA_CORE_HAS_DLPACK_H)
  (void)out_tensor_map;
  (void)data;
  (void)device_type;
  (void)device_id;
  (void)ndim;
  (void)shape;
  (void)strides;
  (void)dtype_code;
  (void)dtype_bits;
  (void)dtype_lanes;
  (void)box_sizes;
  (void)elem_strides;
  (void)interleave_layout;
  (void)swizzle;
  (void)l2_fetch_size;
  (void)oob_fill;
  cuda_core_write_err(err, err_cap, "CCCL <cuda/tma> and/or <dlpack/dlpack.h> not available at build time");
  return 1;
#else
  try
  {
    if (!out_tensor_map)
    {
      cuda_core_write_err(err, err_cap, "out_tensor_map is NULL");
      return 1;
    }
    if (!data)
    {
      cuda_core_write_err(err, err_cap, "tensor data pointer is NULL");
      return 1;
    }
    if (!shape || !box_sizes || ndim <= 0)
    {
      cuda_core_write_err(err, err_cap, "invalid rank/shape/box_sizes");
      return 1;
    }

    DLTensor t{};
    t.data        = data;
    t.device      = {static_cast<DLDeviceType>(device_type), device_id};
    t.ndim        = ndim;
    t.dtype.code  = dtype_code;
    t.dtype.bits  = dtype_bits;
    t.dtype.lanes = dtype_lanes;
    // CCCL promises not to mutate the arrays, but DLPack uses non-const pointers.
    t.shape       = const_cast<int64_t*>(shape);
    t.strides     = const_cast<int64_t*>(strides);
    t.byte_offset = 0;

    const auto layout = static_cast<cuda::tma_interleave_layout>(interleave_layout);
    const auto swz    = static_cast<cuda::tma_swizzle>(swizzle);
    const auto l2     = static_cast<cuda::tma_l2_fetch_size>(l2_fetch_size);
    const auto oob    = static_cast<cuda::tma_oob_fill>(oob_fill);

    auto box = cuda::std::span<const int>(box_sizes, static_cast<size_t>(ndim));

    CUtensorMap desc{};
    if (elem_strides)
    {
      auto es = cuda::std::span<const int>(elem_strides, static_cast<size_t>(ndim));
      desc    = cuda::make_tma_descriptor(t, box, es, layout, swz, l2, oob);
    }
    else
    {
      desc = cuda::make_tma_descriptor(t, box, layout, swz, l2, oob);
    }

    ::memcpy(out_tensor_map, &desc, sizeof(CUtensorMap));
    cuda_core_write_err(err, err_cap, nullptr);
    return 0;
  }
  catch (const std::exception& e)
  {
    cuda_core_write_err(err, err_cap, e.what());
    return 1;
  }
  catch (...)
  {
    cuda_core_write_err(err, err_cap, "unknown error while building TMA descriptor");
    return 1;
  }
#endif
}
