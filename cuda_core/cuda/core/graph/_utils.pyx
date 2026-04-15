# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cpython.ref cimport Py_INCREF

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy as c_memcpy

from cuda.bindings cimport cydriver

from cuda.core._utils.cuda_utils cimport HANDLE_RETURN


cdef extern from "Python.h":
    void _py_decref "Py_DECREF" (void*)


cdef void _py_host_trampoline(void* data) noexcept with gil:
    (<object>data)()


cdef void _py_host_destructor(void* data) noexcept with gil:
    _py_decref(data)


cdef bint _is_py_host_trampoline(cydriver.CUhostFn fn) noexcept nogil:
    return fn == <cydriver.CUhostFn>_py_host_trampoline


cdef void _attach_user_object(
        cydriver.CUgraph graph, void* ptr,
        cydriver.CUhostFn destroy) except *:
    """Create a CUDA user object and transfer ownership to the graph.

    On success the graph owns the resource (via MOVE semantics).
    On failure the destroy callback is invoked to clean up ptr,
    then a CUDAError is raised — callers need no try/except.
    """
    cdef cydriver.CUuserObject user_obj = NULL
    cdef cydriver.CUresult ret
    with nogil:
        ret = cydriver.cuUserObjectCreate(
            &user_obj, ptr, destroy, 1,
            cydriver.CU_USER_OBJECT_NO_DESTRUCTOR_SYNC)
        if ret == cydriver.CUDA_SUCCESS:
            ret = cydriver.cuGraphRetainUserObject(
                graph, user_obj, 1, cydriver.CU_GRAPH_USER_OBJECT_MOVE)
            if ret != cydriver.CUDA_SUCCESS:
                cydriver.cuUserObjectRelease(user_obj, 1)
    if ret != cydriver.CUDA_SUCCESS:
        if user_obj == NULL:
            destroy(ptr)
        HANDLE_RETURN(ret)


cdef void _attach_host_callback_to_graph(
        cydriver.CUgraph graph, object fn, object user_data,
        cydriver.CUhostFn* out_fn, void** out_user_data) except *:
    """Resolve a Python callable or ctypes CFuncPtr into a C callback pair.

    Handles Py_INCREF, user-object attachment for lifetime management,
    and user_data copying.  On return, *out_fn and *out_user_data are
    ready to pass to cuGraphAddHostNode or cuLaunchHostFunc.
    """
    import ctypes as ct

    cdef void* fn_pyobj = NULL

    if isinstance(fn, ct._CFuncPtr):
        Py_INCREF(fn)
        fn_pyobj = <void*>fn
        _attach_user_object(
            graph, fn_pyobj,
            <cydriver.CUhostFn>_py_host_destructor)
        out_fn[0] = <cydriver.CUhostFn><uintptr_t>ct.cast(
            fn, ct.c_void_p).value

        if user_data is not None:
            if isinstance(user_data, int):
                out_user_data[0] = <void*><uintptr_t>user_data
            else:
                buf = bytes(user_data)
                out_user_data[0] = malloc(len(buf))
                if out_user_data[0] == NULL:
                    raise MemoryError(
                        "failed to allocate user_data buffer")
                c_memcpy(out_user_data[0], <const char*>buf, len(buf))
                _attach_user_object(
                    graph, out_user_data[0],
                    <cydriver.CUhostFn>free)
        else:
            out_user_data[0] = NULL
    else:
        if user_data is not None:
            raise ValueError(
                "user_data is only supported with ctypes "
                "function pointers")
        Py_INCREF(fn)
        fn_pyobj = <void*>fn
        out_fn[0] = <cydriver.CUhostFn>_py_host_trampoline
        out_user_data[0] = fn_pyobj
        _attach_user_object(
            graph, fn_pyobj,
            <cydriver.CUhostFn>_py_host_destructor)
