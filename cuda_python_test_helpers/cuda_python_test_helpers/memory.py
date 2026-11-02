# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-pool helpers shared by the cuda.bindings and cuda.core test suites."""

import pytest

from cuda.bindings import runtime as cudart

__all__ = [
    "POOL_SIZE",
    "supports_managed_memory",
    "supports_memory_pools",
    "xfail_graph_mempool_oom",
]

# Cap for memory pools created by tests. A pool created without an explicit
# maximum size instead reserves a system-dependent window that scales with
# installed device memory: about twice the device memory, which is hundreds of
# GiB on large-memory GPUs. That reservation can fail where the virtual address
# space is limited (e.g. riscv64 Sv39, a 39-bit address space). The per-process
# virtual address budget is also bounded (~1 TB on Windows MCDM), and a
# reservation is not returned until the pool is torn down and its
# stream-ordered frees retire, so oversized windows accumulate across a session
# and eventually starve later pool creations with CUDA_ERROR_OUT_OF_MEMORY
# (issue #2381).
POOL_SIZE = 2 * 1024 * 1024  # 2 MiB


def _device_attribute_is_set(attribute, device):
    err, value = cudart.cudaDeviceGetAttribute(attribute, device)
    return err == cudart.cudaError_t.cudaSuccess and bool(value)


def supports_memory_pools(device=0):
    """Return True if ``device`` supports stream-ordered memory pools.

    A failing attribute query (e.g. no device) counts as unsupported, so this
    is safe to evaluate in a ``skipif`` condition at collection time.
    """
    return _device_attribute_is_set(cudart.cudaDeviceAttr.cudaDevAttrMemoryPoolsSupported, device)


def supports_managed_memory(device=0):
    """Return True if ``device`` supports managed memory.

    A failing attribute query (e.g. no device) counts as unsupported.
    """
    return _device_attribute_is_set(cudart.cudaDeviceAttr.cudaDevAttrManagedMemory, device)


def xfail_graph_mempool_oom():
    """Mark the calling test as an expected failure; never returns.

    Call this after the graph memory pool failed with an out-of-memory error.
    The graph memory pool reserves virtual address space of about twice the
    device memory, and unlike pools from ``cuMemPoolCreate`` it cannot be
    capped. On systems with a limited virtual address space this reservation
    can fail with CUDA_ERROR_OUT_OF_MEMORY, especially when another uncapped
    pool already exists in the process.

    The caller does its own out-of-memory check and resource cleanup first.
    """
    pytest.xfail("graph memory pool creation failed with CUDA_ERROR_OUT_OF_MEMORY")
