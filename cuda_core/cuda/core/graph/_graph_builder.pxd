# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver

from cuda.core._resource_handles cimport GraphExecHandle, GraphHandle, StreamHandle
from cuda.core._stream cimport Stream


cdef class GraphBuilder:
    cdef:
        GraphHandle _h_graph
        StreamHandle _h_stream
        int _kind
        int _state
        Stream _stream  # cached to avoid reconstruction from _h_stream handle
        object __weakref__

    @staticmethod
    cdef GraphBuilder _init(Stream stream)


cdef class Graph:
    cdef:
        GraphExecHandle _h_graph_exec
        object __weakref__

    @staticmethod
    cdef Graph _init(cydriver.CUgraphExec graph_exec)
