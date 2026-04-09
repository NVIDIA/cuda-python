/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Vendored subset of PyTorch's AOT Inductor (AOTI) stable C ABI.
 * Original: torch/csrc/inductor/aoti_torch/c/shim.h
 *
 * These are declarations only -- no definitions are provided.  The actual
 * symbols are exported by libtorch (loaded via torch._C with RTLD_GLOBAL)
 * and resolved at runtime by the dynamic linker.  This means PyTorch is
 * NOT required at compile time.
 */

#ifndef CUDA_CORE_AOTI_SHIM_H
#define CUDA_CORE_AOTI_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t AOTITorchError;

/* Opaque tensor handle -- corresponds to at::Tensor on the C++ side. */
struct AtenTensorOpaque;
typedef struct AtenTensorOpaque* AtenTensorHandle;

/* ---- tensor metadata --------------------------------------------------- */

AOTITorchError aoti_torch_get_data_ptr(
    AtenTensorHandle tensor, void** ret_data_ptr);

AOTITorchError aoti_torch_get_dim(
    AtenTensorHandle tensor, int64_t* ret_dim);

AOTITorchError aoti_torch_get_sizes(
    AtenTensorHandle tensor, int64_t** ret_sizes);

AOTITorchError aoti_torch_get_strides(
    AtenTensorHandle tensor, int64_t** ret_strides);

/* ---- dtype ------------------------------------------------------------- */

AOTITorchError aoti_torch_get_dtype(
    AtenTensorHandle tensor, int32_t* ret_dtype);

int32_t aoti_torch_dtype_float16(void);
int32_t aoti_torch_dtype_float32(void);
int32_t aoti_torch_dtype_float64(void);
int32_t aoti_torch_dtype_bfloat16(void);
int32_t aoti_torch_dtype_uint8(void);
int32_t aoti_torch_dtype_int8(void);
int32_t aoti_torch_dtype_int16(void);
int32_t aoti_torch_dtype_int32(void);
int32_t aoti_torch_dtype_int64(void);
int32_t aoti_torch_dtype_bool(void);
int32_t aoti_torch_dtype_complex32(void);
int32_t aoti_torch_dtype_complex64(void);
int32_t aoti_torch_dtype_complex128(void);

/* ---- device ------------------------------------------------------------ */

AOTITorchError aoti_torch_get_device_type(
    AtenTensorHandle tensor, int32_t* ret_device_type);

AOTITorchError aoti_torch_get_device_index(
    AtenTensorHandle tensor, int32_t* ret_device_index);

int32_t aoti_torch_device_type_cpu(void);
int32_t aoti_torch_device_type_cuda(void);

/* ---- stream -------------------------------------------------------------- */

AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index, void** ret_stream);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CUDA_CORE_AOTI_SHIM_H */
