# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport (
    GraphHandle,
    GraphNodeHandle,
    OpaqueHandle,
    as_intptr,
    graph_node_get_graph,
)


cdef class GraphNode:
    cdef:
        GraphNodeHandle _h_node
        bint _is_entry
        object __weakref__

    @staticmethod
    cdef GraphNode _create(GraphHandle h_graph, cydriver.CUgraphNode node)


cdef inline int GN_check_valid(GraphNode self) except -1:
    if as_intptr(graph_node_get_graph(self._h_node)) == 0:
        raise RuntimeError("GraphNode belongs to an invalid GraphDefinition")
    if not self._is_entry and as_intptr(self._h_node) == 0:
        raise RuntimeError("GraphNode has been destroyed")
    return 0


cdef OpaqueHandle _resolve_memcpy_operand(
    object operand, object owner, str side, cydriver.CUdeviceptr* out_ptr) except *

cdef cydriver.CUmemorytype _get_memcpy_memory_type(
    cydriver.CUdeviceptr ptr) except *

cdef void _init_memcpy_params(
    cydriver.CUdeviceptr dst, cydriver.CUdeviceptr src, size_t size,
    cydriver.CUDA_MEMCPY3D* params, cydriver.CUmemorytype* dst_type,
    cydriver.CUmemorytype* src_type) except *
