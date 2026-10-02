# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# distutils: language=c++
from libc.stdlib cimport calloc, free
from libc.string cimport memset
import cuda.bindings.driver as cuda
import cuda.bindings.runtime as cudart
import numpy as np
import pytest

cimport cuda.bindings.cydriver as ccuda
cimport cuda.bindings.cyruntime as ccudart

# Cap for the memory pool created by test_interop_memPool. Touching the device's
# default pool (e.g. via cuDeviceGetDefaultMemPool, cudaDeviceGetDefaultMemPool,
# or cuDeviceGetMemPool / cudaDeviceGetMemPool before a pool has been set)
# reserves virtual address space of about twice the device memory, which
# cannot be satisfied in a 39-bit address space (e.g. riscv64 Sv39).
POOL_SIZE = 2 * 1024 * 1024  # 2 MiB


def supportsMemoryPool():
    err, isSupported = cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported, 0)
    return err == cudart.cudaError_t.cudaSuccess and isSupported


def test_interop_stream():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    cdef ccuda.CUstream* stream_dr = <ccuda.CUstream*>calloc(1, sizeof(ccuda.CUstream))
    cerr_dr = ccuda.cuStreamCreate(stream_dr, 0)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    cerr_rt = ccudart.cudaStreamDestroy(stream_dr[0])
    assert(cerr_rt == ccudart.cudaSuccess)
    free(stream_dr)

    # RT to DRV
    cdef ccudart.cudaStream_t* stream_rt = <ccudart.cudaStream_t*>calloc(1, sizeof(ccudart.cudaStream_t))
    cerr_rt = ccudart.cudaStreamCreate(stream_rt)
    assert(cerr_rt == ccudart.cudaSuccess)
    cerr_dr = ccuda.cuStreamDestroy(stream_rt[0])
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    free(stream_rt)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)


def test_interop_event():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    cdef ccuda.CUevent* event_dr = <ccuda.CUevent*>calloc(1, sizeof(ccuda.CUevent))
    cerr_dr = ccuda.cuEventCreate(event_dr, 0)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    cerr_rt = ccudart.cudaEventDestroy(event_dr[0])
    assert(cerr_rt == ccudart.cudaSuccess)
    free(event_dr)

    # RT to DRV
    cdef ccudart.cudaEvent_t* event_rt = <ccudart.cudaEvent_t*>calloc(1, sizeof(ccudart.cudaEvent_t))
    cerr_rt = ccudart.cudaEventCreate(event_rt)
    assert(cerr_rt == ccudart.cudaSuccess)
    cerr_dr = ccuda.cuEventDestroy(event_rt[0])
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    free(event_rt)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)


def test_interop_graph():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    cdef ccuda.CUgraph* graph_dr = <ccuda.CUgraph*>calloc(1, sizeof(ccuda.CUgraph))
    cerr_dr = ccuda.cuGraphCreate(graph_dr, 0)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    cerr_rt = ccudart.cudaGraphDestroy(graph_dr[0])
    assert(cerr_rt == ccudart.cudaSuccess)
    free(graph_dr)

    # RT to DRV
    cdef ccudart.cudaGraph_t* graph_rt = <ccudart.cudaGraph_t*>calloc(1, sizeof(ccudart.cudaGraph_t))
    cerr_rt = ccudart.cudaGraphCreate(graph_rt, 0)
    assert(cerr_rt == ccudart.cudaSuccess)
    cerr_dr = ccuda.cuGraphDestroy(graph_rt[0])
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    free(graph_rt)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)


def test_interop_graphNode():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    cdef ccuda.CUgraph* graph_dr = <ccuda.CUgraph*>calloc(1, sizeof(ccuda.CUgraph))
    cdef ccuda.CUgraphNode* graph_node_dr = <ccuda.CUgraphNode*>calloc(1, sizeof(ccuda.CUgraphNode))
    cdef ccuda.CUgraphNode* dependencies_dr = NULL

    cerr_dr = ccuda.cuGraphCreate(graph_dr, 0)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    cerr_dr = ccuda.cuGraphAddEmptyNode(graph_node_dr, graph_dr[0], dependencies_dr, 0)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    cerr_rt = ccudart.cudaGraphDestroyNode(graph_node_dr[0])
    assert(cerr_rt == ccudart.cudaSuccess)

    # RT to DRV
    cdef ccudart.cudaGraphNode_t* graph_node_rt = <ccudart.cudaGraphNode_t*>calloc(1, sizeof(ccudart.cudaGraphNode_t))
    cerr_rt = ccudart.cudaGraphAddEmptyNode(graph_node_rt, graph_dr[0], dependencies_dr, 0)
    assert(cerr_rt == ccudart.cudaSuccess)
    cerr_dr = ccuda.cuGraphDestroyNode(graph_node_rt[0])
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    cerr_rt = ccudart.cudaGraphDestroy(graph_dr[0])
    assert(cerr_rt == ccudart.cudaSuccess)

    free(graph_dr)
    free(graph_node_dr)
    free(graph_node_rt)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)


@pytest.mark.skipif(not supportsMemoryPool(), reason='Requires mempool operations')
def test_interop_memPool():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # Use a small, capped pool instead of the device's default pool. The
    # pool must be set before it is queried below because the getters create
    # (and reserve address space for) the default pool if none has been set.
    cdef ccuda.CUmemPoolProps props
    memset(&props, 0, sizeof(props))
    props.allocType = ccuda.CU_MEM_ALLOCATION_TYPE_PINNED
    props.handleTypes = ccuda.CU_MEM_HANDLE_TYPE_NONE
    props.location.type = ccuda.CU_MEM_LOCATION_TYPE_DEVICE
    props.location.id = 0
    props.maxSize = POOL_SIZE
    cdef ccuda.CUmemoryPool pool
    cerr_dr = ccuda.cuMemPoolCreate(&pool, &props)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)

    # DRV to RT
    cerr_rt = ccudart.cudaDeviceSetMemPool(0, pool)
    assert(cerr_rt == ccudart.cudaSuccess)

    # RT to DRV
    cdef ccudart.cudaMemPool_t mempool_rt
    cerr_rt = ccudart.cudaDeviceGetMemPool(&mempool_rt, 0)
    assert(cerr_rt == ccudart.cudaSuccess)
    assert(<void*>mempool_rt == <void*>pool)
    cerr_dr = ccuda.cuDeviceSetMemPool(cuda.CUdevice(0), mempool_rt)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)

    cerr_dr = ccuda.cuMemPoolDestroy(pool)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)


def test_interop_graphExec():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(None, 0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    cdef ccuda.CUgraph* graph_dr = <ccuda.CUgraph*>calloc(1, sizeof(ccuda.CUgraph))
    cdef ccuda.CUgraphNode* graph_node_dr = <ccuda.CUgraphNode*>calloc(1, sizeof(ccuda.CUgraphNode))
    cdef ccuda.CUgraphExec* graph_exec_dr = <ccuda.CUgraphExec*>calloc(1, sizeof(ccuda.CUgraphExec))
    cdef ccuda.CUgraphNode* dependencies_dr = NULL

    cerr_dr = ccuda.cuGraphCreate(graph_dr, 0)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    cerr_dr = ccuda.cuGraphAddEmptyNode(graph_node_dr, graph_dr[0], dependencies_dr, 0)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)

    # DRV to RT
    cerr_dr = ccuda.cuGraphInstantiate(graph_exec_dr, graph_dr[0], 0)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    cerr_rt = ccudart.cudaGraphExecDestroy(graph_exec_dr[0])
    assert(cerr_rt == ccudart.cudaSuccess)

    # RT to DRV
    cdef ccudart.cudaGraphExec_t* graph_exec_rt = <ccudart.cudaGraphExec_t*>calloc(1, sizeof(ccudart.cudaGraphExec_t))

    cerr_rt = ccudart.cudaGraphInstantiate(graph_exec_rt, graph_dr[0], 0)
    assert(cerr_rt == ccudart.cudaSuccess)
    cerr_dr = ccuda.cuGraphExecDestroy(graph_exec_rt[0])
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    cerr_rt = ccudart.cudaGraphDestroy(graph_dr[0])
    assert(cerr_rt == ccudart.cudaSuccess)

    free(graph_dr)
    free(graph_node_dr)
    free(graph_exec_dr)
    free(graph_exec_rt)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
