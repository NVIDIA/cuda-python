# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport GraphHandle, GraphNodeHandle


cdef class GraphNode:
    cdef:
        GraphNodeHandle _h_node
        object __weakref__

    @staticmethod
    cdef GraphNode _create(GraphHandle h_graph, cydriver.CUgraphNode node)
