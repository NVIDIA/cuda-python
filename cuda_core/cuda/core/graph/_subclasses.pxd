# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t

from cuda.bindings cimport cydriver
from cuda.core.graph._graph_def cimport Condition
from cuda.core.graph._graph_node cimport GraphNode
from cuda.core._resource_handles cimport EventHandle, GraphHandle, GraphNodeHandle, KernelHandle


cdef class EmptyNode(GraphNode):
    @staticmethod
    cdef EmptyNode _create_impl(GraphNodeHandle h_node)


cdef class KernelNode(GraphNode):
    cdef:
        tuple _grid
        tuple _block
        unsigned int _shmem_size
        KernelHandle _h_kernel

    @staticmethod
    cdef KernelNode _create_with_params(GraphNodeHandle h_node,
                                        tuple grid, tuple block, unsigned int shmem_size,
                                        KernelHandle h_kernel)

    @staticmethod
    cdef KernelNode _create_from_driver(GraphNodeHandle h_node)


cdef class AllocNode(GraphNode):
    cdef:
        cydriver.CUdeviceptr _dptr
        size_t _bytesize
        int _device_id
        str _memory_type
        tuple _peer_access

    @staticmethod
    cdef AllocNode _create_with_params(GraphNodeHandle h_node,
                                       cydriver.CUdeviceptr dptr, size_t bytesize,
                                       int device_id, str memory_type, tuple peer_access)

    @staticmethod
    cdef AllocNode _create_from_driver(GraphNodeHandle h_node)


cdef class FreeNode(GraphNode):
    cdef:
        cydriver.CUdeviceptr _dptr

    @staticmethod
    cdef FreeNode _create_with_params(GraphNodeHandle h_node,
                                      cydriver.CUdeviceptr dptr)

    @staticmethod
    cdef FreeNode _create_from_driver(GraphNodeHandle h_node)


cdef class MemsetNode(GraphNode):
    cdef:
        cydriver.CUdeviceptr _dptr
        unsigned int _value
        unsigned int _element_size
        size_t _width
        size_t _height
        size_t _pitch

    @staticmethod
    cdef MemsetNode _create_with_params(GraphNodeHandle h_node,
                                        cydriver.CUdeviceptr dptr, unsigned int value,
                                        unsigned int element_size, size_t width,
                                        size_t height, size_t pitch)

    @staticmethod
    cdef MemsetNode _create_from_driver(GraphNodeHandle h_node)


cdef class MemcpyNode(GraphNode):
    cdef:
        cydriver.CUdeviceptr _dst
        cydriver.CUdeviceptr _src
        size_t _size
        cydriver.CUmemorytype _dst_type
        cydriver.CUmemorytype _src_type

    @staticmethod
    cdef MemcpyNode _create_with_params(GraphNodeHandle h_node,
                                        cydriver.CUdeviceptr dst, cydriver.CUdeviceptr src,
                                        size_t size, cydriver.CUmemorytype dst_type,
                                        cydriver.CUmemorytype src_type)

    @staticmethod
    cdef MemcpyNode _create_from_driver(GraphNodeHandle h_node)


cdef class ChildGraphNode(GraphNode):
    cdef:
        GraphHandle _h_child_graph

    @staticmethod
    cdef ChildGraphNode _create_with_params(GraphNodeHandle h_node,
                                            GraphHandle h_child_graph)

    @staticmethod
    cdef ChildGraphNode _create_from_driver(GraphNodeHandle h_node)


cdef class EventRecordNode(GraphNode):
    cdef:
        EventHandle _h_event

    @staticmethod
    cdef EventRecordNode _create_with_params(GraphNodeHandle h_node,
                                             EventHandle h_event)

    @staticmethod
    cdef EventRecordNode _create_from_driver(GraphNodeHandle h_node)


cdef class EventWaitNode(GraphNode):
    cdef:
        EventHandle _h_event

    @staticmethod
    cdef EventWaitNode _create_with_params(GraphNodeHandle h_node,
                                           EventHandle h_event)

    @staticmethod
    cdef EventWaitNode _create_from_driver(GraphNodeHandle h_node)


cdef class HostCallbackNode(GraphNode):
    cdef:
        object _callable
        cydriver.CUhostFn _fn
        void* _user_data

    @staticmethod
    cdef HostCallbackNode _create_with_params(GraphNodeHandle h_node,
                                              object callable_obj, cydriver.CUhostFn fn,
                                              void* user_data)

    @staticmethod
    cdef HostCallbackNode _create_from_driver(GraphNodeHandle h_node)


cdef class ConditionalNode(GraphNode):
    cdef:
        Condition _condition
        cydriver.CUgraphConditionalNodeType _cond_type
        tuple _branches

    @staticmethod
    cdef ConditionalNode _create_from_driver(GraphNodeHandle h_node)


cdef class IfNode(ConditionalNode):
    pass


cdef class IfElseNode(ConditionalNode):
    pass


cdef class WhileNode(ConditionalNode):
    pass


cdef class SwitchNode(ConditionalNode):
    pass
