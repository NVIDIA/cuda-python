# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport GraphHandle


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
