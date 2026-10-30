# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t

from cuda.bindings cimport cydriver
from cuda.core._memory._buffer cimport Buffer, Buffer_check_open
from cuda.core._rt cimport (
    GraphHandle,
    GraphNodeHandle,
    OpaqueHandle,
    as_cu,
    as_intptr,
    graph_node_get_graph,
    make_opaque_py,
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


cdef inline OpaqueHandle _buffer_attachment_owner(Buffer buf, str label):
    """Copy a Buffer's device-pointer handle into an attachment owner."""
    Buffer_check_open(buf)
    # The local is required: Cython permits the DevicePtrHandle -> OpaqueHandle
    # conversion on assignment, but not directly in a return statement.
    cdef OpaqueHandle attachment_owner = buf._h_ptr
    return attachment_owner


cdef inline OpaqueHandle _resolve_memcpy_operand(
        object operand, object owner, str side,
        cydriver.CUdeviceptr* out_ptr) except *:
    """Resolve an operand to a pointer and optional attachment owner.

    ``operand`` is a :class:`Buffer` or a raw integer address; its device
    pointer is written to ``out_ptr``. For a :class:`Buffer` operand, returns an
    owner that retains the underlying allocation (not the wrapper). For a raw
    pointer, returns an owner built from ``owner`` (or an empty handle when
    ``owner`` is ``None``).

    Raises
    ------
    ValueError
        If ``operand`` is a :class:`Buffer` and ``owner`` is not ``None``.
        If a :class:`Buffer` operand or ``*_owner`` has no active allocation.
    """
    cdef Buffer buf
    cdef OpaqueHandle attachment_owner

    if isinstance(operand, Buffer):
        if owner is not None:
            raise ValueError(
                f"{side}_owner cannot be used when {side} is a Buffer"
            )
        buf = operand
        attachment_owner = _buffer_attachment_owner(buf, side)
        out_ptr[0] = as_cu(buf._h_ptr)
        return attachment_owner
    out_ptr[0] = <cydriver.CUdeviceptr><uintptr_t>operand
    if owner is None:
        return OpaqueHandle()
    if isinstance(owner, Buffer):
        return _buffer_attachment_owner(owner, f"{side}_owner")
    return make_opaque_py(owner)


cdef cydriver.CUmemorytype _get_memcpy_memory_type(
    cydriver.CUdeviceptr ptr) except *

cdef void _init_memcpy_params(
    cydriver.CUdeviceptr dst, cydriver.CUdeviceptr src, size_t size,
    cydriver.CUDA_MEMCPY3D* params, cydriver.CUmemorytype* dst_type,
    cydriver.CUmemorytype* src_type) except *
