# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport GraphHandle, GraphNodeHandle, OpaqueHandle


cdef class GraphNode:
    cdef:
        GraphNodeHandle _h_node
        object __weakref__

    @staticmethod
    cdef GraphNode _create(GraphHandle h_graph, cydriver.CUgraphNode node)


cdef OpaqueHandle _resolve_memcpy_operand(
    object operand, object owner, str side, cydriver.CUdeviceptr* out_ptr) except *

cdef void _init_memcpy_params(
    cydriver.CUdeviceptr dst, cydriver.CUdeviceptr src, size_t size,
    cydriver.CUDA_MEMCPY3D* params, cydriver.CUmemorytype* dst_type,
    cydriver.CUmemorytype* src_type) except *
