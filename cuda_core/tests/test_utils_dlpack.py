# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DLPack-focused tests for ``StridedMemoryView``.

Split out of ``test_utils.py`` (which had grown large): export/import
round-trips, capsule + deleter paths, ``from_dlpack`` error handling, and the
``__dlpack_c_exchange_api__`` C exchange-API helpers driven through ctypes.
CAI-only behavior stays in ``test_utils.py``.
"""

import ctypes

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None
import numpy as np
import pytest

from cuda.core._dlpack import DLDeviceType
from cuda.core.utils import StridedMemoryView

_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.argtypes = (ctypes.py_object, ctypes.c_char_p)
_PyCapsule_IsValid.restype = ctypes.c_int


_NUMPY_NATIVE_DLPACK_DTYPES = (
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    np.bool_,
)
if ml_dtypes is not None:
    # Supported on NumPy 2.5 and ml_dtypes (probably) 0.5.5+. On older stacks the
    # per-test probe skips it, since NumPy's __dlpack__ doesn't reliably export
    # ml_dtypes-extended dtypes (covered separately via jax/torch).
    _NUMPY_NATIVE_DLPACK_DTYPES += (ml_dtypes.bfloat16,)


def _assert_dlpack_export_roundtrip(src):
    # Skip only if NumPy itself can't round-trip this dtype/shape; past the
    # probe, a failure on our view is a regression, not an env limitation.
    try:
        np.from_dlpack(src)
    except (BufferError, TypeError, RuntimeError) as e:
        pytest.skip(f"NumPy does not support DLPack for {src.dtype} {src.shape}: {e}")
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    out = np.from_dlpack(view)
    assert out.dtype == src.dtype
    assert out.shape == src.shape
    assert np.array_equal(out, src)


@pytest.mark.parametrize("dtype", _NUMPY_NATIVE_DLPACK_DTYPES)
def test_dlpack_export_roundtrip_dtypes(dtype):
    """Export every NumPy-native DLPack dtype through ``StridedMemoryView.__dlpack__``."""
    _assert_dlpack_export_roundtrip(np.zeros((2, 3), dtype=dtype))


@pytest.mark.parametrize(
    "shape",
    [pytest.param((), id="scalar"), pytest.param((0, 3), id="empty")],
)
def test_dlpack_export_roundtrip_special_shapes(shape):
    """Export scalar and zero-volume shapes through ``StridedMemoryView.__dlpack__``."""
    _assert_dlpack_export_roundtrip(np.zeros(shape, dtype=np.complex128))


def test_dlpack_export_unversioned_capsule_and_deleter():
    """``__dlpack__()`` with no ``max_version`` yields an *unversioned* unused
    DLPack capsule; dropping it unconsumed runs ``_smv_pycapsule_deleter`` on
    the non-versioned branch (freeing the managed tensor)."""
    src = np.arange(6, dtype=np.int32)
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    capsule = view.__dlpack__()
    assert _PyCapsule_IsValid(capsule, b"dltensor") == 1
    assert _PyCapsule_IsValid(capsule, b"dltensor_versioned") == 0
    del capsule  # unconsumed -> deleter frees dlm_tensor


def test_dlpack_export_versioned_capsule_and_deleter():
    """``__dlpack__(max_version=(1, 0))`` yields a *versioned* unused capsule;
    dropping it unconsumed runs the versioned ``_smv_pycapsule_deleter`` branch."""
    src = np.arange(6, dtype=np.int32)
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    capsule = view.__dlpack__(max_version=(1, 0))
    assert _PyCapsule_IsValid(capsule, b"dltensor_versioned") == 1
    assert _PyCapsule_IsValid(capsule, b"dltensor") == 0
    del capsule  # unconsumed -> versioned deleter frees dlm_tensor_ver


def test_from_dlpack_cpu_stream_none_ambiguous():
    """A CPU DLPack source with ``stream_ptr=None`` is rejected as ambiguous."""
    src = np.arange(4, dtype=np.float32)
    with pytest.raises(BufferError, match="stream=None is ambiguous"):
        StridedMemoryView.from_dlpack(src, stream_ptr=None)


def test_from_dlpack_unsupported_device_type():
    """``view_as_dlpack`` rejects a DLPack device that is neither CPU, CUDA,
    CUDA-pinned, nor CUDA-managed before ever calling ``__dlpack__``."""

    class _FakeUnsupportedDevice:
        def __dlpack_device__(self):
            return (7, 0)  # e.g. kDLVulkan -- unsupported by cuda.core

        def __dlpack__(self, **kwargs):
            raise AssertionError("__dlpack__ must not be reached")

    with pytest.raises(BufferError, match="device not supported"):
        StridedMemoryView.from_dlpack(_FakeUnsupportedDevice(), stream_ptr=0)


class _DLPackNoMaxVersion:
    """Wraps a StridedMemoryView but rejects the ``max_version`` kwarg, forcing the
    TypeError fallback in ``view_as_dlpack`` and an *unversioned* capsule import.

    Backed by a StridedMemoryView (not NumPy directly) so the test stays valid
    even if NumPy eventually stops exporting unversioned (0.x) DLPack capsules."""

    def __init__(self, arr):
        self._arr = StridedMemoryView.from_any_interface(arr, stream_ptr=-1)
        self.max_versions = []  # max_version seen on each __dlpack__ call, in order

    def __dlpack_device__(self):
        return self._arr.__dlpack_device__()

    def __dlpack__(self, *, stream=None, max_version=None, **kwargs):
        self.max_versions.append(max_version)
        if max_version is not None:
            raise TypeError("max_version is not supported")
        return self._arr.__dlpack__(stream=stream)


def test_from_dlpack_typeerror_fallback_unversioned_import():
    """When ``__dlpack__(max_version=...)`` raises TypeError, view_as_dlpack
    retries without it and imports the resulting unversioned capsule; the view
    then owns that capsule and frees it on ``__dealloc__``."""
    src = np.arange(6, dtype=np.int32)
    wrapper = _DLPackNoMaxVersion(src)
    view = StridedMemoryView.from_dlpack(wrapper, stream_ptr=-1)
    # Guard the TypeError fallback path: versioned attempt, then legacy retry.
    assert len(wrapper.max_versions) == 2, f"expected versioned attempt + retry, got {wrapper.max_versions}"
    assert isinstance(wrapper.max_versions[0], tuple)  # versioned attempt was made
    assert wrapper.max_versions[1] is None  # fallback retried without max_version
    assert view.ptr == src.ctypes.data
    out = np.from_dlpack(view)
    assert np.array_equal(out, src)
    del view  # exercise __dealloc__ on the imported (used) unversioned capsule


# ---------------------------------------------------------------------------
# DLPack C exchange API (`__dlpack_c_exchange_api__`)
#
# Drive the C function pointers exposed by the capsule the way a native
# consumer would, exercising the StridedMemoryView exchange-API implementation.
#
# The C functions report failure by setting a Python error and returning -1.
# Defining the pointers with PYFUNCTYPE (Python calling convention) lets ctypes
# propagate that real exception (TypeError/RuntimeError/NotImplementedError)
# instead of wrapping it in a SystemError, so the tests assert the meaningful
# type directly.
# ---------------------------------------------------------------------------

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
_PyCapsule_GetPointer.restype = ctypes.c_void_p


class _DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class _DLPackExchangeAPIHeader(ctypes.Structure):
    _fields_ = [("version", _DLPackVersion), ("prev_api", ctypes.c_void_p)]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_FN_FROM_PY = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
_FN_TO_PY = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
_FN_DLTENSOR_FROM_PY = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
_FN_ALLOCATOR = ctypes.PYFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p
)
_FN_CURRENT_STREAM = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p))


class _DLPackExchangeAPI(ctypes.Structure):
    _fields_ = [
        ("header", _DLPackExchangeAPIHeader),
        ("managed_tensor_allocator", _FN_ALLOCATOR),
        ("managed_tensor_from_py_object_no_sync", _FN_FROM_PY),
        ("managed_tensor_to_py_object_no_sync", _FN_TO_PY),
        ("dltensor_from_py_object_no_sync", _FN_DLTENSOR_FROM_PY),
        ("current_work_stream", _FN_CURRENT_STREAM),
    ]


def _get_exchange_api():
    capsule = StridedMemoryView.__dlpack_c_exchange_api__
    ptr = _PyCapsule_GetPointer(capsule, b"dlpack_exchange_api")
    assert ptr
    return ctypes.cast(ptr, ctypes.POINTER(_DLPackExchangeAPI)).contents


def test_dlpack_c_exchange_api_header_version():
    """The exchange-API header advertises a non-zero DLPack version."""
    api = _get_exchange_api()
    assert (api.header.version.major, api.header.version.minor) >= (1, 0)
    assert not api.header.prev_api


def test_dlpack_c_exchange_api_current_work_stream():
    """``current_work_stream`` reports no current stream (cuda.core has none)."""
    api = _get_exchange_api()
    out = ctypes.c_void_p(123)
    rc = api.current_work_stream(int(DLDeviceType.kDLCPU), 0, ctypes.byref(out))
    assert rc == 0
    assert not out.value  # set back to NULL


def test_dlpack_c_exchange_api_dltensor_from_py_object():
    """``dltensor_from_py_object_no_sync`` fills a borrowed DLTensor from a view."""
    api = _get_exchange_api()
    src = np.arange(12, dtype=np.int32).reshape(3, 4)
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    out = _DLTensor()
    rc = api.dltensor_from_py_object_no_sync(id(view), ctypes.byref(out))
    assert rc == 0
    assert out.ndim == 2
    assert out.device.device_type == int(DLDeviceType.kDLCPU)
    assert out.data == src.ctypes.data
    assert [out.shape[i] for i in range(out.ndim)] == [3, 4]


def test_dlpack_c_exchange_api_dltensor_from_py_object_type_error():
    """A non-StridedMemoryView py_object is rejected (TypeError, rc=-1)."""
    api = _get_exchange_api()
    not_a_view = object()
    out = _DLTensor()
    with pytest.raises(TypeError, match="must be a StridedMemoryView"):
        api.dltensor_from_py_object_no_sync(id(not_a_view), ctypes.byref(out))


def test_dlpack_c_exchange_api_managed_tensor_roundtrip():
    """``managed_tensor_from_py_object_no_sync`` produces a managed tensor that
    ``managed_tensor_to_py_object_no_sync`` turns back into a StridedMemoryView.

    This exercises the versioned export fill and the capsule-import helper.
    The reconstructed view intentionally keeps a reference (the C side holds one
    via Py_INCREF), so the managed tensor is not freed here -- avoiding any
    double-free across the two calls that share the same tensor.
    """
    api = _get_exchange_api()
    src = np.arange(6, dtype=np.float64).reshape(2, 3)
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)

    tensor = ctypes.c_void_p(0)
    rc = api.managed_tensor_from_py_object_no_sync(id(view), ctypes.byref(tensor))
    assert rc == 0
    assert tensor.value  # non-NULL DLManagedTensorVersioned*

    out_obj = ctypes.c_void_p(0)
    rc = api.managed_tensor_to_py_object_no_sync(tensor, ctypes.byref(out_obj))
    assert rc == 0
    assert out_obj.value
    imported = ctypes.cast(ctypes.c_void_p(out_obj.value), ctypes.py_object).value
    assert isinstance(imported, StridedMemoryView)
    assert imported.shape == (2, 3)
    assert imported.ptr == src.ctypes.data


def test_dlpack_c_exchange_api_to_py_object_null_tensor():
    """``managed_tensor_to_py_object_no_sync`` rejects a NULL tensor (RuntimeError)."""
    api = _get_exchange_api()
    out_obj = ctypes.c_void_p(0)
    with pytest.raises(RuntimeError, match="tensor cannot be NULL"):
        api.managed_tensor_to_py_object_no_sync(None, ctypes.byref(out_obj))
    assert not out_obj.value  # set to NULL before the error


def test_dlpack_c_exchange_api_managed_tensor_allocator_not_supported():
    """``managed_tensor_allocator`` is unsupported (NotImplementedError).

    The implementation sets a Python error even when no ``SetError`` callback is
    passed, so with PYFUNCTYPE ctypes surfaces the NotImplementedError directly.
    """
    api = _get_exchange_api()
    out = ctypes.c_void_p(123)
    with pytest.raises(NotImplementedError, match="not supported"):
        api.managed_tensor_allocator(None, ctypes.byref(out), None, None)
    assert not out.value  # set to NULL before the error
