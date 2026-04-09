# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor bridge: extract PyTorch tensor metadata via the AOTI stable C ABI.

PyTorch is NOT required at build time.  At runtime the AOTI symbols are
resolved from ``torch._C`` (which is loaded with ``RTLD_GLOBAL``).

The ``pyobj_to_aten_handle`` trick exploits the internal layout of
``THPVariable`` (PyTorch's Python tensor wrapper)::

    struct THPVariable {
        PyObject_HEAD
        MaybeOwned<at::Tensor> cdata;   // <-- this IS the AtenTensorHandle
    };

Offsetting past ``PyObject_HEAD`` gives us the ``at::Tensor`` pointer
without any Python attribute access or method calls (~10 ns per tensor).

Credit: Emilio Castillo (ecastillo@nvidia.com) – original tensor-bridge POC.

.. note::

   This module must NOT be imported at ``cuda.core`` load time.  It is
   loaded lazily (by ``_memoryview.pyx``) only when the user actually
   passes a ``torch.Tensor``.  The caller must ensure that
   ``torch._C`` has been re-opened with ``RTLD_GLOBAL`` *before*
   importing this module so that the AOTI symbols are visible.
"""

from libc.stdint cimport intptr_t, int8_t, int16_t, int32_t, int64_t, uint8_t

from cuda.core._memoryview cimport StridedMemoryView
from cuda.core._layout cimport _StridedLayout
from cuda.bindings cimport cydriver
from cuda.core._resource_handles cimport (
    EventHandle,
    create_event_handle_noctx,
    as_cu,
)
from cuda.core._utils.cuda_utils cimport HANDLE_RETURN

cdef extern from "Python.h":
    ctypedef struct PyObject:
        pass

cdef extern from "_include/aoti_shim.h":
    ctypedef int32_t AOTITorchError

    ctypedef struct AtenTensorOpaque:
        pass
    ctypedef AtenTensorOpaque* AtenTensorHandle

    # tensor metadata
    AOTITorchError aoti_torch_get_data_ptr(AtenTensorHandle, void**)
    AOTITorchError aoti_torch_get_dim(AtenTensorHandle, int64_t*)
    AOTITorchError aoti_torch_get_sizes(AtenTensorHandle, int64_t**)
    AOTITorchError aoti_torch_get_strides(AtenTensorHandle, int64_t**)

    # dtype
    AOTITorchError aoti_torch_get_dtype(AtenTensorHandle, int32_t*)
    int32_t aoti_torch_dtype_float16()
    int32_t aoti_torch_dtype_float32()
    int32_t aoti_torch_dtype_float64()
    int32_t aoti_torch_dtype_bfloat16()
    int32_t aoti_torch_dtype_uint8()
    int32_t aoti_torch_dtype_int8()
    int32_t aoti_torch_dtype_int16()
    int32_t aoti_torch_dtype_int32()
    int32_t aoti_torch_dtype_int64()
    int32_t aoti_torch_dtype_bool()
    int32_t aoti_torch_dtype_complex32()
    int32_t aoti_torch_dtype_complex64()
    int32_t aoti_torch_dtype_complex128()

    # device
    AOTITorchError aoti_torch_get_device_type(AtenTensorHandle, int32_t*)
    AOTITorchError aoti_torch_get_device_index(AtenTensorHandle, int32_t*)
    int32_t aoti_torch_device_type_cpu()
    int32_t aoti_torch_device_type_cuda()

    # stream
    AOTITorchError aoti_torch_get_current_cuda_stream(int32_t, void**)

import numpy


# ---------------------------------------------------------------------------
# Module-level state (initialised at import time — AOTI symbols are
# guaranteed visible because _memoryview bootstraps RTLD_GLOBAL before
# importing us)
# ---------------------------------------------------------------------------

cdef int32_t _DEVICE_TYPE_CPU  = aoti_torch_device_type_cpu()
cdef int32_t _DEVICE_TYPE_CUDA = aoti_torch_device_type_cuda()
cdef dict _aoti_dtype_map = None
cdef dict _aoti_itemsize_map = None


# ---------------------------------------------------------------------------
# pointer extraction
# ---------------------------------------------------------------------------

cdef inline AtenTensorHandle pyobj_to_aten_handle(object obj):
    """Extract AtenTensorHandle by offsetting past PyObject_HEAD."""
    return <AtenTensorHandle>(<char*><PyObject*>obj + sizeof(PyObject))


cdef inline void check_aoti(AOTITorchError err, const char* name) except *:
    """Raise RuntimeError if an AOTI call returned a non-zero error code."""
    if err != 0:
        raise RuntimeError(f"{name.decode()} failed")


# ---------------------------------------------------------------------------
# dtype mapping (AOTI int32 -> numpy dtype)
# ---------------------------------------------------------------------------

cdef dict _build_dtype_map():
    try:
        from ml_dtypes import bfloat16 as _bf16
        has_bfloat16 = True
    except ImportError:
        has_bfloat16 = False

    cdef dict m = {
        aoti_torch_dtype_float16():    numpy.dtype(numpy.float16),
        aoti_torch_dtype_float32():    numpy.dtype(numpy.float32),
        aoti_torch_dtype_float64():    numpy.dtype(numpy.float64),
        aoti_torch_dtype_uint8():      numpy.dtype(numpy.uint8),
        aoti_torch_dtype_int8():       numpy.dtype(numpy.int8),
        aoti_torch_dtype_int16():      numpy.dtype(numpy.int16),
        aoti_torch_dtype_int32():      numpy.dtype(numpy.int32),
        aoti_torch_dtype_int64():      numpy.dtype(numpy.int64),
        aoti_torch_dtype_bool():       numpy.dtype(numpy.bool_),
        aoti_torch_dtype_complex64():  numpy.dtype(numpy.complex64),
        aoti_torch_dtype_complex128(): numpy.dtype(numpy.complex128),
    }
    if has_bfloat16:
        m[aoti_torch_dtype_bfloat16()] = numpy.dtype("bfloat16")
    return m


cdef object _get_aoti_dtype(int32_t dtype_code):
    global _aoti_dtype_map
    if _aoti_dtype_map is None:
        _aoti_dtype_map = _build_dtype_map()
    result = _aoti_dtype_map.get(dtype_code)
    if result is None:
        raise TypeError(f"Unsupported AOTI dtype code: {dtype_code}")
    return result


def resolve_aoti_dtype(int32_t dtype_code):
    """Python-callable wrapper around _get_aoti_dtype (for lazy resolution)."""
    return _get_aoti_dtype(dtype_code)


cdef dict _build_itemsize_map():
    return {
        aoti_torch_dtype_bool():       sizeof(uint8_t),
        aoti_torch_dtype_uint8():      sizeof(uint8_t),
        aoti_torch_dtype_int8():       sizeof(int8_t),
        aoti_torch_dtype_float16():    sizeof(int16_t),    # no C float16
        aoti_torch_dtype_bfloat16():   sizeof(int16_t),    # no C bfloat16
        aoti_torch_dtype_int16():      sizeof(int16_t),
        aoti_torch_dtype_complex32():  2 * sizeof(int16_t),  # no C complex32
        aoti_torch_dtype_float32():    sizeof(float),
        aoti_torch_dtype_int32():      sizeof(int32_t),
        aoti_torch_dtype_complex64():  2 * sizeof(float),
        aoti_torch_dtype_float64():    sizeof(double),
        aoti_torch_dtype_int64():      sizeof(int64_t),
        aoti_torch_dtype_complex128(): 2 * sizeof(double),
    }


cdef int _get_aoti_itemsize(int32_t dtype_code) except -1:
    global _aoti_itemsize_map
    if _aoti_itemsize_map is None:
        _aoti_itemsize_map = _build_itemsize_map()
    result = _aoti_itemsize_map.get(dtype_code)
    if result is None:
        raise TypeError(f"Unsupported AOTI dtype code: {dtype_code}")
    return <int>result


# ---------------------------------------------------------------------------
# Stream ordering helper
# ---------------------------------------------------------------------------

cpdef void sync_torch_stream(int32_t device_index,
                             intptr_t consumer_s) except *:
    """Establish stream ordering between PyTorch's current CUDA stream
    and the given consumer stream.

    Records an event on PyTorch's current stream (the producer) and makes
    the consumer stream wait on it.  This is a no-op if both streams are
    the same.
    """
    cdef void* producer_s
    cdef EventHandle h_event

    check_aoti(aoti_torch_get_current_cuda_stream(device_index, &producer_s),
               b"aoti_torch_get_current_cuda_stream")
    if <intptr_t>producer_s != consumer_s:
        with nogil:
            h_event = create_event_handle_noctx(
                cydriver.CUevent_flags.CU_EVENT_DISABLE_TIMING)
            HANDLE_RETURN(cydriver.cuEventRecord(
                as_cu(h_event), <cydriver.CUstream>producer_s))
            HANDLE_RETURN(cydriver.cuStreamWaitEvent(
                <cydriver.CUstream>consumer_s, as_cu(h_event), 0))


# ---------------------------------------------------------------------------
# Public API: construct StridedMemoryView from a torch.Tensor
# ---------------------------------------------------------------------------

def view_as_torch_tensor(object obj, object stream_ptr, view=None):
    """Create/populate a :class:`StridedMemoryView` from a ``torch.Tensor``.

    This is a fast path that avoids DLPack/CAI protocol overhead by
    reading tensor metadata directly through the AOTI stable C ABI.

    Parameters
    ----------
    obj : torch.Tensor
        The source tensor.
    stream_ptr : int or None
        Consumer stream pointer.  When not ``-1``, stream ordering is
        established between PyTorch's current CUDA stream (the producer)
        and the consumer stream, matching the DLPack contract.
    view : StridedMemoryView, optional
        If provided, populate this existing view in-place.  Otherwise a
        new instance is created.
    """
    cdef AtenTensorHandle handle = pyobj_to_aten_handle(obj)
    cdef void* data_ptr
    cdef int64_t ndim
    cdef int64_t* sizes_ptr
    cdef int64_t* strides_ptr
    cdef int32_t dtype_code
    cdef int32_t device_type, device_index
    cdef StridedMemoryView buf
    cdef int itemsize
    cdef intptr_t _stream_ptr_int
    cdef _StridedLayout layout

    check_aoti(aoti_torch_get_data_ptr(handle, &data_ptr),
               b"aoti_torch_get_data_ptr")
    check_aoti(aoti_torch_get_dim(handle, &ndim),
               b"aoti_torch_get_dim")
    check_aoti(aoti_torch_get_sizes(handle, &sizes_ptr),
               b"aoti_torch_get_sizes")
    check_aoti(aoti_torch_get_strides(handle, &strides_ptr),
               b"aoti_torch_get_strides")
    check_aoti(aoti_torch_get_dtype(handle, &dtype_code),
               b"aoti_torch_get_dtype")
    check_aoti(aoti_torch_get_device_type(handle, &device_type),
               b"aoti_torch_get_device_type")
    check_aoti(aoti_torch_get_device_index(handle, &device_index),
               b"aoti_torch_get_device_index")

    # -- populate StridedMemoryView --
    if view is not None:
        buf = <StridedMemoryView>view
    else:
        buf = StridedMemoryView.__new__(StridedMemoryView)

    buf.ptr = <intptr_t>data_ptr
    buf.readonly = False
    buf.exporting_obj = obj
    buf.dl_tensor = NULL
    buf.metadata = None
    buf._buffer = None

    if device_type == _DEVICE_TYPE_CPU:
        buf.device_id = -1
        buf.is_device_accessible = False
    elif device_type == _DEVICE_TYPE_CUDA:
        buf.device_id = <int>device_index
        buf.is_device_accessible = True

        # -- stream ordering (matches the DLPack contract) --
        if stream_ptr is not None:
            _stream_ptr_int = int(stream_ptr)
            if _stream_ptr_int != -1:
                sync_torch_stream(device_index, _stream_ptr_int)
    else:
        raise BufferError(
            f"Unsupported device type from torch tensor "
            f"(AOTI device type id: {device_type})")

    # Defer full numpy dtype resolution until first .dtype access.
    # Store the raw AOTI dtype code in metadata for lazy lookup.
    buf.metadata = <int>dtype_code

    # Build _StridedLayout.  init_from_ptr copies shape/strides so we are
    # safe even though they are borrowed pointers.
    itemsize = _get_aoti_itemsize(dtype_code)
    layout = _StridedLayout.__new__(_StridedLayout)
    layout.init_from_ptr(
        <int>ndim,
        sizes_ptr,
        strides_ptr,
        itemsize,
    )
    buf._layout = layout

    return buf
