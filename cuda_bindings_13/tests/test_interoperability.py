# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from cuda_python_test_helpers.memory import POOL_SIZE, supports_memory_pools

import cuda.bindings.runtime as cudart
from cuda.bindings._v2 import driver

# NULL dependency array: the node has no dependencies.
NO_DEPS = 0


def test_interop_stream():
    # DRV to RT
    stream = driver.stream_create(0)
    (err_rt,) = cudart.cudaStreamDestroy(stream)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, stream = cudart.cudaStreamCreate()
    assert err_rt == cudart.cudaError_t.cudaSuccess
    driver.stream_destroy_v2(int(stream))


def test_interop_event():
    # DRV to RT
    event = driver.event_create(0)
    (err_rt,) = cudart.cudaEventDestroy(event)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, event = cudart.cudaEventCreate()
    assert err_rt == cudart.cudaError_t.cudaSuccess
    driver.event_destroy_v2(int(event))


def test_interop_graph():
    # DRV to RT
    graph = driver.graph_create(0)
    (err_rt,) = cudart.cudaGraphDestroy(graph)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, graph = cudart.cudaGraphCreate(0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    driver.graph_destroy(int(graph))


def test_interop_graphNode():
    graph = driver.graph_create(0)

    # DRV to RT
    node = driver.graph_add_empty_node(graph, NO_DEPS, 0)
    (err_rt,) = cudart.cudaGraphDestroyNode(node)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, node = cudart.cudaGraphAddEmptyNode(graph, [], 0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    driver.graph_destroy_node(int(node))

    (err_rt,) = cudart.cudaGraphDestroy(graph)
    assert err_rt == cudart.cudaError_t.cudaSuccess


# cudaUserObject_t
# TODO


# cudaFunction_t
# TODO


@pytest.mark.thread_unsafe(reason="changes the device's current memory pool")
@pytest.mark.agent_authored(model="claude-sonnet-5.5")
@pytest.mark.skipif(not supports_memory_pools(), reason="Requires mempool operations")
def test_interop_memPool():
    # Use a small, capped pool instead of the device's default pool. The pool
    # must be set before it is queried below because the getters create (and
    # reserve address space for) the default pool if none has been set.
    props = driver.MemPoolProps_v1()
    props.alloc_type = driver.MemAllocationType.PINNED
    props.handle_types = driver.MemAllocationHandleType.NONE
    props.location.type = driver.MemLocationType.DEVICE
    props.location.id = 0
    props.max_size = POOL_SIZE
    pool = driver.mem_pool_create(props)

    try:
        # DRV to RT
        (err_rt,) = cudart.cudaDeviceSetMemPool(0, pool)
        assert err_rt == cudart.cudaError_t.cudaSuccess

        # RT to DRV
        err_rt, rt_pool = cudart.cudaDeviceGetMemPool(0)
        assert err_rt == cudart.cudaError_t.cudaSuccess
        assert int(rt_pool) == pool
        driver.device_set_mem_pool(0, int(rt_pool))
    finally:
        driver.mem_pool_destroy(pool)


def test_interop_graphExec():
    graph = driver.graph_create(0)
    node = driver.graph_add_empty_node(graph, NO_DEPS, 0)

    # DRV to RT
    graphExec = driver.graph_instantiate(graph, 0)
    (err_rt,) = cudart.cudaGraphExecDestroy(graphExec)
    assert err_rt == cudart.cudaError_t.cudaSuccess

    # RT to DRV
    err_rt, graphExec = cudart.cudaGraphInstantiate(graph, 0)
    assert err_rt == cudart.cudaError_t.cudaSuccess
    driver.graph_exec_destroy(int(graphExec))

    (err_rt,) = cudart.cudaGraphDestroy(graph)
    assert err_rt == cudart.cudaError_t.cudaSuccess


def test_interop_deviceptr():
    # Allocate dev memory
    size = 1024 * np.uint8().itemsize
    dptr = driver.mem_alloc_v2(size)

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
    driver.mem_free_v2(dptr)
