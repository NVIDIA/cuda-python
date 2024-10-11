# Copyright 2021-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import pytest
import cuda.cuda as cuda
import cuda.cudart as cudart
import numpy as np

def supportsMemoryPool():
    err, isSupported = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported, 0)
    return err == cudart.cudaError_t.cudaSuccess and isSupported

def test_interop_stream():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    err_dr, stream = cuda.cuStreamCreate(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_rt, = cudart.cudaStreamDestroy(stream)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)

    # RT to DRV
    err_rt, stream = cudart.cudaStreamCreate()
    assert(err_rt == cudart.cudaError_t.cudaSuccess)
    err_dr, = cuda.cuStreamDestroy(stream)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

def test_interop_event():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    err_dr, event = cuda.cuEventCreate(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_rt, = cudart.cudaEventDestroy(event)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)

    # RT to DRV
    err_rt, event = cudart.cudaEventCreate()
    assert(err_rt == cudart.cudaError_t.cudaSuccess)
    err_dr, = cuda.cuEventDestroy(event)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

def test_interop_graph():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    err_dr, graph = cuda.cuGraphCreate(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_rt, = cudart.cudaGraphDestroy(graph)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)

    # RT to DRV
    err_rt, graph = cudart.cudaGraphCreate(0)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)
    err_dr, = cuda.cuGraphDestroy(graph)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

def test_interop_graphNode():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    err_dr, graph = cuda.cuGraphCreate(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    err_dr, node = cuda.cuGraphAddEmptyNode(graph, [], 0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_rt, = cudart.cudaGraphDestroyNode(node)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)

    # RT to DRV
    err_rt, node = cudart.cudaGraphAddEmptyNode(graph, [], 0)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)
    err_dr, = cuda.cuGraphDestroyNode(node)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    err_rt, = cudart.cudaGraphDestroy(graph)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)
    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

def test_interop_userObject():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # cudaUserObject_t
    # TODO

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

def test_interop_function():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # cudaFunction_t
    # TODO

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

@pytest.mark.skipif(not supportsMemoryPool(), reason='Requires mempool operations')
def test_interop_memPool():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    err_dr, pool = cuda.cuDeviceGetDefaultMemPool(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_rt, = cudart.cudaDeviceSetMemPool(0, pool)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)

    # RT to DRV
    err_rt, pool = cudart.cudaDeviceGetDefaultMemPool(0)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)
    err_dr, = cuda.cuDeviceSetMemPool(0, pool)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

def test_interop_graphExec():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, graph = cuda.cuGraphCreate(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, node = cuda.cuGraphAddEmptyNode(graph, [], 0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    err_dr, graphExec, errorNode = cuda.cuGraphInstantiate(graph, b'', 0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_rt, = cudart.cudaGraphExecDestroy(graphExec)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)

    # RT to DRV
    err_rt, graphExec, errorNode = cudart.cudaGraphInstantiate(graph, b'', 0)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)
    err_dr, = cuda.cuGraphExecDestroy(graphExec)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    err_rt, = cudart.cudaGraphDestroy(graph)
    assert(err_rt == cudart.cudaError_t.cudaSuccess)
    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
