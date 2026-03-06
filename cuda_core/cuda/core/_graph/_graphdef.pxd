# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport EventHandle, GraphHandle, GraphNodeHandle, KernelHandle


cdef class Condition
cdef class GraphDef
cdef class Node
cdef class EmptyNode(Node)
cdef class KernelNode(Node)
cdef class AllocNode(Node)
cdef class FreeNode(Node)
cdef class MemsetNode(Node)
cdef class MemcpyNode(Node)
cdef class ChildGraphNode(Node)
cdef class EventRecordNode(Node)
cdef class EventWaitNode(Node)
cdef class HostCallbackNode(Node)
cdef class ConditionalNode(Node)
cdef class IfNode(ConditionalNode)
cdef class IfElseNode(ConditionalNode)
cdef class WhileNode(ConditionalNode)
cdef class SwitchNode(ConditionalNode)


cdef class Condition:
    cdef:
        cydriver.CUgraphConditionalHandle _c_handle
        object __weakref__


cdef class GraphDef:
    cdef:
        GraphHandle _h_graph
        object __weakref__

    @staticmethod
    cdef GraphDef _from_handle(GraphHandle h_graph)


cdef class Node:
    cdef:
        GraphNodeHandle _h_node
        tuple _pred_cache
        tuple _succ_cache
        object __weakref__

    @staticmethod
    cdef Node _create(GraphHandle h_graph, cydriver.CUgraphNode node)


cdef class EmptyNode(Node):
    @staticmethod
    cdef EmptyNode _create_impl(GraphNodeHandle h_node)


cdef class KernelNode(Node):
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


cdef class AllocNode(Node):
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


cdef class FreeNode(Node):
    cdef:
        cydriver.CUdeviceptr _dptr

    @staticmethod
    cdef FreeNode _create_with_params(GraphNodeHandle h_node,
                                      cydriver.CUdeviceptr dptr)

    @staticmethod
    cdef FreeNode _create_from_driver(GraphNodeHandle h_node)


cdef class MemsetNode(Node):
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


cdef class MemcpyNode(Node):
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


cdef class ChildGraphNode(Node):
    cdef:
        GraphHandle _h_child_graph

    @staticmethod
    cdef ChildGraphNode _create_with_params(GraphNodeHandle h_node,
                                            GraphHandle h_child_graph)

    @staticmethod
    cdef ChildGraphNode _create_from_driver(GraphNodeHandle h_node)


cdef class EventRecordNode(Node):
    cdef:
        EventHandle _h_event

    @staticmethod
    cdef EventRecordNode _create_with_params(GraphNodeHandle h_node,
                                             EventHandle h_event)

    @staticmethod
    cdef EventRecordNode _create_from_driver(GraphNodeHandle h_node)


cdef class EventWaitNode(Node):
    cdef:
        EventHandle _h_event

    @staticmethod
    cdef EventWaitNode _create_with_params(GraphNodeHandle h_node,
                                           EventHandle h_event)

    @staticmethod
    cdef EventWaitNode _create_from_driver(GraphNodeHandle h_node)


cdef class HostCallbackNode(Node):
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


cdef class ConditionalNode(Node):
    cdef:
        Condition _condition
        cydriver.CUgraphConditionalNodeType _cond_type
        tuple _branches  # tuple of GraphDef (non-owning wrappers)

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
