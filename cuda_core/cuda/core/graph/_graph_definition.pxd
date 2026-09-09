# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport GraphHandle, as_intptr


cdef class GraphCondition:
    cdef:
        cydriver.CUgraphConditionalHandle _c_handle
        object __weakref__

    @staticmethod
    cdef GraphCondition _from_handle(cydriver.CUgraphConditionalHandle c_handle)


cdef class GraphDefinition:
    cdef:
        GraphHandle _h_graph
        object __weakref__

    @staticmethod
    cdef GraphDefinition _from_handle(GraphHandle h_graph)


cdef inline int GD_check_valid(GraphDefinition self) except -1:
    if as_intptr(self._h_graph) == 0:
        raise RuntimeError("GraphDefinition is no longer valid")
    return 0
