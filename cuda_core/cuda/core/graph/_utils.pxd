# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.bindings cimport cydriver


cdef bint _is_py_host_trampoline(cydriver.CUhostFn fn) noexcept nogil

cdef void _attach_user_object(
    cydriver.CUgraph graph, void* ptr,
    cydriver.CUhostFn destroy) except *

cdef void _attach_host_callback_to_graph(
    cydriver.CUgraph graph, object fn, object user_data,
    cydriver.CUhostFn* out_fn, void** out_user_data) except *
