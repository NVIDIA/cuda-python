# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc
from libc.string cimport memcpy as c_memcpy

from cuda.bindings cimport cydriver

from cuda.core._resource_handles cimport (
    GraphHandle,
    OpaqueHandle,
    graph_set_attachment,
    make_opaque_malloc,
    make_opaque_py,
)
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

import ctypes as ct


cdef void _py_host_trampoline(void* data) noexcept with gil:
    (<object>data)()


cdef bint _is_py_host_trampoline(cydriver.CUhostFn fn) noexcept nogil:
    return fn == <cydriver.CUhostFn>_py_host_trampoline


cdef void _resolve_host_callback(
        object fn, object user_data,
        cydriver.CUhostFn* out_fn, void** out_user_data,
        OpaqueHandle* out_fn_owner, OpaqueHandle* out_data_owner) except *:
    """Resolve a Python callable or ctypes CFuncPtr into a C callback pair and
    the owners that keep it alive.

    On return ``*out_fn`` / ``*out_user_data`` are ready to pass to
    ``cuGraphAddHostNode`` or ``cuLaunchHostFunc``. ``*out_fn_owner`` owns the
    callback object; ``*out_data_owner`` owns a copied ``user_data`` buffer and
    is left null otherwise. The caller attaches both owners to the graph node.
    """
    if isinstance(fn, ct._CFuncPtr):
        out_fn[0] = <cydriver.CUhostFn><uintptr_t>ct.cast(fn, ct.c_void_p).value
        if user_data is None:
            out_user_data[0] = NULL
        elif isinstance(user_data, int):
            out_user_data[0] = <void*><uintptr_t>user_data
        else:
            buf = bytes(user_data)
            if len(buf):
                out_user_data[0] = malloc(len(buf))
                if out_user_data[0] == NULL:
                    raise MemoryError("failed to allocate user_data buffer")
                c_memcpy(out_user_data[0], <const char*>buf, len(buf))
                out_data_owner[0] = make_opaque_malloc(out_user_data[0])
            else:
                out_user_data[0] = NULL
    else:
        if user_data is not None:
            raise ValueError(
                "user_data is only supported with ctypes function pointers")
        out_fn[0] = <cydriver.CUhostFn>_py_host_trampoline
        out_user_data[0] = <void*>fn

    out_fn_owner[0] = make_opaque_py(fn)


cdef int _attach_host_callback_owners(
        const GraphHandle& h_graph, cydriver.CUgraphNode node,
        OpaqueHandle fn_owner, OpaqueHandle data_owner) except -1:
    """Attach a resolved callback and copied user-data owner as one bundle."""
    HANDLE_RETURN(graph_set_attachment(
        h_graph, node, fn_owner, data_owner))
    return 0
