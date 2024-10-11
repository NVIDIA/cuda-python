# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import pytest
from cuda import cuda
import ctypes
import random

from .perf_test_utils import ASSERT_DRV, init_cuda

random.seed(0)

idx = 0
def query_attribute(attribute, ptrs):
    global idx
    ptr = ptrs[idx]
    idx = (idx + 1 ) % len(ptrs)

    cuda.cuPointerGetAttribute(attribute, ptr)

def query_attributes(attributes, ptrs):
    global idx
    ptr = ptrs[idx]
    idx = (idx + 1 ) % len(ptrs)

    cuda.cuPointerGetAttributes(len(attributes), attributes, ptr)

@pytest.mark.benchmark(group="pointer-attributes")
# Measure cuPointerGetAttribute in the same way as C benchmarks
def test_pointer_get_attribute(benchmark, init_cuda):
    _ = init_cuda

    ptrs = []
    for _ in range(500):
        err, ptr = cuda.cuMemAlloc(1 << 18)
        ASSERT_DRV(err)
        ptrs.append(ptr)

    random.shuffle(ptrs)

    benchmark(query_attribute, cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, ptrs)

    for p in ptrs:
        err, = cuda.cuMemFree(p)
        ASSERT_DRV(err)

@pytest.mark.benchmark(group="pointer-attributes")
# Measure cuPointerGetAttributes with all attributes
def test_pointer_get_attributes_all(benchmark, init_cuda):
    _ = init_cuda

    ptrs = []
    for _ in range(500):
        err, ptr = cuda.cuMemAlloc(1 << 18)
        ASSERT_DRV(err)
        ptrs.append(ptr)

    random.shuffle(ptrs)

    attributes = [cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_CONTEXT,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_HOST_POINTER,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_P2P_TOKENS,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_BUFFER_ID,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_MANAGED,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_RANGE_SIZE,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MAPPED,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_ACCESS_FLAGS,
                  cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE]

    benchmark(query_attributes, attributes, ptrs)

    for p in ptrs:
        err, = cuda.cuMemFree(p)
        ASSERT_DRV(err)

@pytest.mark.benchmark(group="pointer-attributes")
# Measure cuPointerGetAttributes with a single attribute
def test_pointer_get_attributes_single(benchmark, init_cuda):
    _ = init_cuda

    ptrs = []
    for _ in range(500):
        err, ptr = cuda.cuMemAlloc(1 << 18)
        ASSERT_DRV(err)
        ptrs.append(ptr)

    random.shuffle(ptrs)

    attributes = [cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,]

    benchmark(query_attributes, attributes, ptrs)

    for p in ptrs:
        err, = cuda.cuMemFree(p)
        ASSERT_DRV(err)
