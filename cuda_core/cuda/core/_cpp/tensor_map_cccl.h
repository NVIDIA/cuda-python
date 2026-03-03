// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDA_CORE_TENSOR_MAP_CCCL_H_
#define CUDA_CORE_TENSOR_MAP_CCCL_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Build a tiled CUtensorMap using CCCL's cuda::make_tma_descriptor (from <cuda/tma>).
//
// Returns 0 on success; on failure returns non-zero and writes a best-effort
// human-readable message into (err, err_cap) if provided.
int cuda_core_cccl_make_tma_descriptor_tiled(
  void* out_tensor_map,
  void* data,
  int device_type,
  int device_id,
  int ndim,
  const int64_t* shape,        // length ndim
  const int64_t* strides,      // length ndim, or NULL for contiguous
  uint8_t dtype_code,
  uint8_t dtype_bits,
  uint16_t dtype_lanes,
  const int* box_sizes,        // length ndim
  const int* elem_strides,     // length ndim, or NULL for all-ones overload
  int interleave_layout,
  int swizzle,
  int l2_fetch_size,
  int oob_fill,
  char* err,
  size_t err_cap) noexcept;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CUDA_CORE_TENSOR_MAP_CCCL_H_
