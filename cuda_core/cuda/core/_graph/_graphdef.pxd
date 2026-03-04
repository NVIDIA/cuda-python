# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport GraphHandle


cdef class GraphDef
cdef class Node


cdef class GraphDef:
    cdef:
        GraphHandle _h_graph
        object __weakref__

    @staticmethod
    cdef GraphDef _from_handle(GraphHandle h_graph)


cdef class Node:
    cdef:
        GraphHandle _h_graph
        cydriver.CUgraphNode _node  # NULL for root
        cydriver.CUdeviceptr _dptr  # non-zero for alloc nodes
        object __weakref__

    @staticmethod
    cdef Node _create(GraphHandle h_graph, cydriver.CUgraphNode node, cydriver.CUdeviceptr dptr)
