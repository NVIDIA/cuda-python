# Copyright 2021-2024 NVIDIA Corporation.  All rights reserved.
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

cimport cuda.ccuda as ccuda
cimport cuda.ccudart as ccudart
from libc.stdlib cimport calloc, free


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
    err_dr, ctx = cuda.cuCtxCreate(0, device)
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
    err_dr, ctx = cuda.cuCtxCreate(0, device)
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
    err_dr, ctx = cuda.cuCtxCreate(0, device)
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
    err_dr, ctx = cuda.cuCtxCreate(0, device)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)

    # DRV to RT
    cdef ccuda.CUmemoryPool* mempool_dr = <ccuda.CUmemoryPool*>calloc(1, sizeof(ccuda.CUmemoryPool))
    cerr_dr = ccuda.cuDeviceGetDefaultMemPool(mempool_dr, 0)
    assert(cerr_dr == ccuda.CUDA_SUCCESS)
    cerr_rt = ccudart.cudaDeviceSetMemPool(0, mempool_dr[0])
    assert(cerr_rt == ccudart.cudaSuccess)

    # RT to DRV
    cdef ccudart.cudaMemPool_t* mempool_rt = <ccudart.cudaMemPool_t*>calloc(1, sizeof(ccudart.cudaMemPool_t))
    cerr_rt = ccudart.cudaDeviceGetDefaultMemPool(mempool_rt, 0)
    assert(cerr_rt == ccudart.cudaSuccess)
    cerr_dr = ccuda.cuDeviceSetMemPool(cuda.CUdevice(0), mempool_rt[0])
    assert(cerr_dr == ccuda.CUDA_SUCCESS)

    free(mempool_dr)
    free(mempool_rt)

    err_dr, = cuda.cuCtxDestroy(ctx)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)


def test_interop_graphExec():
    err_dr, = cuda.cuInit(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, device = cuda.cuDeviceGet(0)
    assert(err_dr == cuda.CUresult.CUDA_SUCCESS)
    err_dr, ctx = cuda.cuCtxCreate(0, device)
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
