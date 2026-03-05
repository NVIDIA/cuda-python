# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport GraphHandle


cdef class GraphDef
cdef class Node
cdef class EmptyNode(Node)
cdef class KernelNode(Node)
cdef class AllocNode(Node)
cdef class FreeNode(Node)
cdef class MemsetNode(Node)
cdef class EventRecordNode(Node)
cdef class EventWaitNode(Node)


cdef class GraphDef:
    cdef:
        GraphHandle _h_graph
        object __weakref__

    @staticmethod
    cdef GraphDef _from_handle(GraphHandle h_graph)


cdef class Node:
    cdef:
        GraphHandle _h_graph
        cydriver.CUgraphNode _node  # NULL for entry node
        tuple _pred_cache
        tuple _succ_cache
        object __weakref__

    @staticmethod
    cdef Node _create(GraphHandle h_graph, cydriver.CUgraphNode node)


cdef class EmptyNode(Node):
    @staticmethod
    cdef EmptyNode _create_impl(GraphHandle h_graph, cydriver.CUgraphNode node)


cdef class KernelNode(Node):
    cdef:
        tuple _grid
        tuple _block
        unsigned int _shmem_size
        cydriver.CUkernel _kern

    @staticmethod
    cdef KernelNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                        tuple grid, tuple block, unsigned int shmem_size,
                                        cydriver.CUkernel kern)

    @staticmethod
    cdef KernelNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node)


cdef class AllocNode(Node):
    cdef:
        cydriver.CUdeviceptr _dptr
        size_t _bytesize
        int _device_id
        str _memory_type
        tuple _peer_access

    @staticmethod
    cdef AllocNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                       cydriver.CUdeviceptr dptr, size_t bytesize,
                                       int device_id, str memory_type, tuple peer_access)

    @staticmethod
    cdef AllocNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node)


cdef class FreeNode(Node):
    cdef:
        cydriver.CUdeviceptr _dptr

    @staticmethod
    cdef FreeNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                      cydriver.CUdeviceptr dptr)

    @staticmethod
    cdef FreeNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node)


cdef class MemsetNode(Node):
    cdef:
        cydriver.CUdeviceptr _dptr
        unsigned int _value
        unsigned int _element_size
        size_t _width
        size_t _height
        size_t _pitch

    @staticmethod
    cdef MemsetNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                        cydriver.CUdeviceptr dptr, unsigned int value,
                                        unsigned int element_size, size_t width,
                                        size_t height, size_t pitch)

    @staticmethod
    cdef MemsetNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node)


cdef class EventRecordNode(Node):
    cdef:
        cydriver.CUevent _event

    @staticmethod
    cdef EventRecordNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                             cydriver.CUevent event)

    @staticmethod
    cdef EventRecordNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node)


cdef class EventWaitNode(Node):
    cdef:
        cydriver.CUevent _event

    @staticmethod
    cdef EventWaitNode _create_with_params(GraphHandle h_graph, cydriver.CUgraphNode node,
                                           cydriver.CUevent event)

    @staticmethod
    cdef EventWaitNode _create_from_driver(GraphHandle h_graph, cydriver.CUgraphNode node)
