# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver

from cuda.core._resource_handles cimport GraphHandle, OpaqueHandle


cdef bint _is_py_host_trampoline(cydriver.CUhostFn fn) noexcept nogil

cdef void _resolve_host_callback(
    object fn, object user_data,
    cydriver.CUhostFn* out_fn, void** out_user_data,
    OpaqueHandle* out_fn_owner, OpaqueHandle* out_data_owner) except *

cdef int _attach_host_callback_owners(
    const GraphHandle& h_graph, cydriver.CUgraphNode node,
    OpaqueHandle fn_owner, OpaqueHandle data_owner) except -1
