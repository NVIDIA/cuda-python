# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc
from libc.string cimport memcpy as c_memcpy

from cuda.bindings cimport cydriver

from cuda.core._resource_handles cimport (
    OpaqueHandle,
    make_opaque_malloc,
    make_opaque_py,
)

import sys
import ctypes as ct


# CUhostFn is `void (CUDA_CB *)(void*)`. CUDA_CB is __stdcall on Windows and
# empty elsewhere, but ctypes only honors that distinction when it builds a
# callback on 32-bit x86 Windows, which cuda.core does not support: on win-64
# and ARM64 both CFUNCTYPE and WINFUNCTYPE produce a FFI_DEFAULT_ABI thunk. The
# declared result and argument types are all that remain worth checking.
_CUHOSTFN_HINT = (
    "ctypes.CFUNCTYPE(None, ctypes.c_void_p)"
    if sys.platform != "win32"
    else "ctypes.CFUNCTYPE(None, ctypes.c_void_p) or "
    "ctypes.WINFUNCTYPE(None, ctypes.c_void_p)"
)


def _cuhostfn_type_error(detail):
    """Build the rejection message for a non-conforming ctypes callback."""
    return TypeError(
        f"host callback {detail}; CUDA requires a callback matching CUhostFn "
        f"(void (*)(void*)), declared as {_CUHOSTFN_HINT}. "
        "Alternatively, pass a Python callable."
    )


def _validate_ctypes_host_callback(fn):
    """Reject ctypes callbacks whose declared prototype is not CUhostFn.

    ``restype`` and ``argtypes`` are the prototype the caller declared, and are
    what CUDA calls through. A function pointer taken from a shared library
    keeps ctypes' defaults -- a ``c_int`` result and unspecified arguments --
    until the caller declares otherwise, so it must be declared to be accepted.
    """
    restype = fn.restype
    argtypes = fn.argtypes
    if restype is not None or argtypes is None or tuple(argtypes) != (ct.c_void_p,):
        raise _cuhostfn_type_error(
            f"has prototype restype={restype!r}, argtypes={argtypes!r}")


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

    ctypes callbacks are validated against the ``CUhostFn`` ABI before their
    address is passed to CUDA.
    """
    if isinstance(fn, ct._CFuncPtr):
        _validate_ctypes_host_callback(fn)
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
        if not callable(fn):
            raise TypeError(
                f"callback must be callable, got {type(fn).__name__}")
        if user_data is not None:
            raise ValueError(
                "user_data is only supported with ctypes function pointers")
        out_fn[0] = <cydriver.CUhostFn>_py_host_trampoline
        out_user_data[0] = <void*>fn

    out_fn_owner[0] = make_opaque_py(fn)
