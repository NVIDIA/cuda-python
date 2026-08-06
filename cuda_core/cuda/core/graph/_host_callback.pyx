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
import _ctypes
import ctypes as ct


# CUhostFn is `void (CUDA_CB *)(void*)`. CUDA_CB is __stdcall on Windows and
# empty elsewhere, so ctypes.WINFUNCTYPE (stdcall) is the literal match on
# Windows and ctypes.CFUNCTYPE (cdecl) is the match everywhere else. Modern
# Windows (x64 and ARM64) has a single calling convention, where the two are
# interchangeable, and cuda.core callers already pass CFUNCTYPE; Windows
# therefore accepts either.
_FUNCFLAG_CDECL = getattr(ct, "_FUNCFLAG_CDECL", 0x1)
_FUNCFLAG_STDCALL = getattr(_ctypes, "FUNCFLAG_STDCALL", 0x2)
_FUNCFLAG_PYTHONAPI = getattr(ct, "_FUNCFLAG_PYTHONAPI", 0x4)

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

    Only the ctypes type's ``_restype_`` / ``_argtypes_`` / ``_flags_`` are
    checked: that is the ABI the wrapper claims. Instance ``restype`` /
    ``argtypes`` overrides are ignored because CUDA invokes the function
    pointer directly.
    """
    proto = type(fn)
    restype = getattr(proto, "_restype_", None)
    argtypes = getattr(proto, "_argtypes_", None)
    flags = int(getattr(proto, "_flags_", 0))

    if restype is not None or argtypes != (ct.c_void_p,):
        raise _cuhostfn_type_error(
            f"has prototype restype={restype!r}, argtypes={argtypes!r}")

    if flags & _FUNCFLAG_PYTHONAPI:
        raise _cuhostfn_type_error(
            "was declared with ctypes.PYFUNCTYPE, which uses the Python-API "
            "calling convention")

    if sys.platform == "win32":
        if not (flags & (_FUNCFLAG_CDECL | _FUNCFLAG_STDCALL)):
            raise _cuhostfn_type_error("uses an unrecognized calling convention")
    elif flags & _FUNCFLAG_STDCALL:
        raise _cuhostfn_type_error(
            "uses the stdcall calling convention, which applies only to Windows")
    elif not (flags & _FUNCFLAG_CDECL):
        raise _cuhostfn_type_error("uses an unrecognized calling convention")


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
