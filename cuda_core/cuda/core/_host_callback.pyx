# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cpython.buffer cimport (
    PyBUF_SIMPLE,
    PyBUF_WRITABLE,
    PyBuffer_Release,
    PyObject_CheckBuffer,
    PyObject_GetBuffer,
    Py_buffer,
)
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc
from libc.string cimport memcpy as c_memcpy

from cuda.bindings cimport cydriver

from cuda.core._memory._buffer cimport Buffer, Buffer_check_open
from cuda.core._rt cimport (
    OpaqueHandle,
    make_opaque_malloc,
    make_opaque_py,
)

import sys
import ctypes as ct
from _ctypes import CFuncPtr

import numpy

__all__ = ['HOST_COPY_FN', 'pack_host_copy_args']


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


ctypedef struct _HostCopyArgs:
    uintptr_t dst
    uintptr_t src
    size_t nbytes


cdef void _host_copy(void* data) noexcept nogil:
    cdef _HostCopyArgs* args = <_HostCopyArgs*>data
    if args.nbytes:
        c_memcpy(<void*>args.dst, <const void*>args.src, args.nbytes)


cdef object _host_copy_fn_pointer():
    return ct.CFUNCTYPE(None, ct.c_void_p)(<uintptr_t><cydriver.CUhostFn>_host_copy)


HOST_COPY_FN: CFuncPtr = _host_copy_fn_pointer()


cdef uintptr_t _host_copy_address(object obj, size_t nbytes, bint writable, str name) except? 0:
    cdef Buffer buf
    cdef Py_buffer view
    # These export the buffer protocol over their own storage, so a value
    # meant as an address would otherwise pack the address of that storage.
    if isinstance(obj, (bool, numpy.integer, ct._SimpleCData)):
        raise TypeError(f"{name} must not be a {type(obj).__name__}; pass int(addr) for an address")
    if isinstance(obj, int):
        return <uintptr_t>obj
    if isinstance(obj, Buffer):
        buf = <Buffer>obj
        Buffer_check_open(buf)
        # A Buffer without a memory resource could only be checked with a
        # driver query, which is not allowed on a host callback thread.
        if buf._memory_resource is not None and not buf._memory_resource.is_host_accessible:
            raise ValueError(f"{name} is a Buffer that is not host-accessible")
        if nbytes > buf._size:
            raise ValueError(f"nbytes ({nbytes}) exceeds the size of {name} ({buf._size})")
        return <uintptr_t>buf.handle
    if not PyObject_CheckBuffer(obj):
        raise TypeError(
            f"{name} must be an int, a Buffer, or a buffer-protocol object, got {type(obj).__name__}")
    PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE | PyBUF_WRITABLE if writable else PyBUF_SIMPLE)
    try:
        if nbytes > <size_t>view.len:
            raise ValueError(f"nbytes ({nbytes}) exceeds the size of {name} ({view.len})")
        return <uintptr_t>view.buf
    finally:
        PyBuffer_Release(&view)


def pack_host_copy_args(dst, src, nbytes: int) -> bytes:
    """Pack the arguments of :data:`HOST_COPY_FN` into ``user_data`` bytes.

    The result is a record of three native-width fields in this order:
    ``uintptr_t dst``, ``uintptr_t src``, ``size_t nbytes``, with no padding.
    ``struct.pack("@PPN", dst, src, nbytes)`` builds the same record, and
    ``struct.unpack("@PPN", data)`` reads one back. A caller that keeps such a
    record in memory of its own can pass its address as an ``int``
    ``user_data`` instead.

    ``dst`` and ``src`` are not retained. The caller must keep both alive
    until the copy has run.

    Parameters
    ----------
    dst : int, Buffer, or buffer-protocol object
        The destination. Only an ``int`` is a host address. A buffer-protocol
        object must be contiguous and writable.
    src : int, Buffer, or buffer-protocol object
        The source. Only an ``int`` is a host address. A buffer-protocol
        object must be contiguous.
    nbytes : int
        The number of bytes to copy. Zero packs a copy that does nothing.

    Returns
    -------
    bytes
        The packed record, to pass as ``user_data``.

    Raises
    ------
    TypeError
        If ``dst`` or ``src`` is not an ``int``, a :class:`Buffer`, or a
        buffer-protocol object, or is a ``bool``, a NumPy integer scalar, or a
        ctypes scalar such as ``ctypes.c_void_p``. A buffer-protocol object
        that cannot export a contiguous view, or a writable one for ``dst``,
        raises its own error.
    ValueError
        If ``nbytes`` exceeds the size of ``dst`` or ``src``, if a
        :class:`Buffer` comes from a memory resource that is not
        host-accessible, if an address is null, or if the ranges overlap.
    RuntimeError
        If ``dst`` or ``src`` is a closed :class:`Buffer`.

    Notes
    -----
    This function makes no CUDA calls, so a Python host callback can use it to
    enqueue the next copy. It checks host accessibility only for a
    :class:`Buffer` with a memory resource. For an ``int`` or a :class:`Buffer`
    without one, the caller must ensure that the memory is host-accessible.
    """
    cdef size_t n = nbytes
    cdef _HostCopyArgs args
    args.dst = _host_copy_address(dst, n, True, "dst")
    args.src = _host_copy_address(src, n, False, "src")
    args.nbytes = n
    if n:
        if args.dst == 0 or args.src == 0:
            raise ValueError("dst and src must not be null")
        if (args.dst - args.src if args.dst > args.src else args.src - args.dst) < n:
            raise ValueError("dst and src overlap")
    return (<char*>&args)[:sizeof(args)]


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
    cdef bint is_host_copy
    if isinstance(fn, ct._CFuncPtr):
        _validate_ctypes_host_callback(fn)
        out_fn[0] = <cydriver.CUhostFn><uintptr_t>ct.cast(fn, ct.c_void_p).value
        is_host_copy = out_fn[0] == <cydriver.CUhostFn>_host_copy
        if user_data is None:
            out_user_data[0] = NULL
        elif isinstance(user_data, int):
            out_user_data[0] = <void*><uintptr_t>user_data
        else:
            buf = bytes(user_data)
            if is_host_copy and len(buf) != sizeof(_HostCopyArgs):
                raise ValueError(
                    f"HOST_COPY_FN user_data must be {sizeof(_HostCopyArgs)} bytes "
                    f"from pack_host_copy_args, got {len(buf)}")
            if len(buf):
                out_user_data[0] = malloc(len(buf))
                if out_user_data[0] == NULL:
                    raise MemoryError("failed to allocate user_data buffer")
                c_memcpy(out_user_data[0], <const char*>buf, len(buf))
                out_data_owner[0] = make_opaque_malloc(out_user_data[0])
            else:
                out_user_data[0] = NULL
        if is_host_copy and out_user_data[0] == NULL:
            raise ValueError("HOST_COPY_FN requires user_data from pack_host_copy_args or a nonzero address")
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
