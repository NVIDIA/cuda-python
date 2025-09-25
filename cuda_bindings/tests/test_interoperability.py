# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cudart
import numpy as np
import pytest


def supportsMemoryPool():
    err, isSupported = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported, 0)
    return err == cudart.cudaError_t.cudaSuccess and isSupported


def test_interop_stream():
    (err_dr,) = cuda.cuInit(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, device = cuda.cuDeviceGet(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # DRV to RT
    err_dr, stream = cuda.cuStreamCreate(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaStreamDestroy(stream)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, stream = cudart.cudaStreamCreate()
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuStreamDestroy(stream)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    (err_dr,) = cuda.cuCtxDestroy(ctx)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_event():
    (err_dr,) = cuda.cuInit(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, device = cuda.cuDeviceGet(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # DRV to RT
    err_dr, event = cuda.cuEventCreate(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaEventDestroy(event)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, event = cudart.cudaEventCreate()
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuEventDestroy(event)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    (err_dr,) = cuda.cuCtxDestroy(ctx)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_graph():
    (err_dr,) = cuda.cuInit(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, device = cuda.cuDeviceGet(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # DRV to RT
    err_dr, graph = cuda.cuGraphCreate(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaGraphDestroy(graph)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, graph = cudart.cudaGraphCreate(0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuGraphDestroy(graph)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    (err_dr,) = cuda.cuCtxDestroy(ctx)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_graphNode():
    (err_dr,) = cuda.cuInit(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, device = cuda.cuDeviceGet(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    err_dr, graph = cuda.cuGraphCreate(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # DRV to RT
    err_dr, node = cuda.cuGraphAddEmptyNode(graph, [], 0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaGraphDestroyNode(node)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, node = cudart.cudaGraphAddEmptyNode(graph, [], 0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuGraphDestroyNode(node)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    (err_rt,) = cudart.cudaGraphDestroy(graph)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuCtxDestroy(ctx)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_userObject():
    (err_dr,) = cuda.cuInit(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, device = cuda.cuDeviceGet(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # cudaUserObject_t
    # TODO

    (err_dr,) = cuda.cuCtxDestroy(ctx)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_function():
    (err_dr,) = cuda.cuInit(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, device = cuda.cuDeviceGet(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # cudaFunction_t
    # TODO

    (err_dr,) = cuda.cuCtxDestroy(ctx)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


@pytest.mark.skipif(not supportsMemoryPool(), reason="Requires mempool operations")
def test_interop_memPool():
    (err_dr,) = cuda.cuInit(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, device = cuda.cuDeviceGet(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # DRV to RT
    err_dr, pool = cuda.cuDeviceGetDefaultMemPool(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaDeviceSetMemPool(0, pool)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, pool = cudart.cudaDeviceGetDefaultMemPool(0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuDeviceSetMemPool(0, pool)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    (err_dr,) = cuda.cuCtxDestroy(ctx)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_graphExec():
    (err_dr,) = cuda.cuInit(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, device = cuda.cuDeviceGet(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, graph = cuda.cuGraphCreate(0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    err_dr, node = cuda.cuGraphAddEmptyNode(graph, [], 0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # DRV to RT
    err_dr, graphExec = cuda.cuGraphInstantiate(graph, 0)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_rt,) = cudart.cudaGraphExecDestroy(graphExec)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, graphExec = cudart.cudaGraphInstantiate(graph, 0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuGraphExecDestroy(graphExec)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    (err_rt,) = cudart.cudaGraphDestroy(graph)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    (err_dr,) = cuda.cuCtxDestroy(ctx)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS


def test_interop_deviceptr():
    # Init CUDA
    (err,) = cuda.cuInit(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Get device
    err, device = cuda.cuDeviceGet(0)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Construct context
    err, ctx = cuda.cuCtxCreate(None, 0, device)
    assert err == cuda.CUresult.CUDA_SUCCESS

    # Allocate dev memory
    size = 1024 * np.uint8().itemsize
    err_dr, dptr = cuda.cuMemAlloc(size)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS

    # Allocate host memory
    h1 = np.full(size, 1).astype(np.uint8)
    h2 = np.full(size, 2).astype(np.uint8)
    assert np.array_equal(h1, h2) is False

    # Initialize device memory
    (err_rt,) = cudart.cudaMemset(dptr, 1, size)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # D to h2
    (err_rt,) = cudart.cudaMemcpy(h2, dptr, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # Validate h1 == h2
    assert np.array_equal(h1, h2)

    # Cleanup
    (err_dr,) = cuda.cuMemFree(dptr)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
    (err_dr,) = cuda.cuCtxDestroy(ctx)
    assert err_dr == cuda.CUresult.CUDA_SUCCESS
