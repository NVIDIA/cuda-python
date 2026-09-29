# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""DLPack tests for ``StridedMemoryView``: export/import round-trips, capsule +
deleter paths, ``from_dlpack`` error handling, and the
``__dlpack_c_exchange_api__`` C exchange-API helpers driven through ctypes.
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

_Py_DecRef = ctypes.pythonapi.Py_DecRef
_Py_DecRef.argtypes = (ctypes.c_void_p,)
_Py_DecRef.restype = None


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


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_dlpack_export_array_interface_reports_cpu(init_cuda):
    """An array-interface view without a DLPack tensor exports as CPU memory."""
    src = np.arange(6, dtype=np.int32)
    view = StridedMemoryView.from_array_interface(src)
    assert view.is_device_accessible is False
    assert view.device_id == -1
    assert view.__dlpack_device__() == (int(DLDeviceType.kDLCPU), 0)
    assert np.array_equal(np.from_dlpack(view), src)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_dlpack_view_of_buffer_reuses_exporting_buffer(init_cuda):
    """Re-viewing a Buffer-imported tensor reuses the original Buffer owner."""
    buffer = init_cuda.memory_resource.allocate(16, stream=init_cuda.default_stream)
    try:
        view = StridedMemoryView.from_dlpack(buffer, stream_ptr=-1)
        adjusted = view.view(dtype=np.uint8)
        assert adjusted.exporting_obj is buffer
        assert adjusted.ptr == int(buffer.handle)
        del adjusted, view
    finally:
        buffer.close()


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


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_from_dlpack_cuda_stream_none_ambiguous():
    """A CUDA DLPack source requires an explicit consumer stream."""

    class _FakeCudaDevice:
        def __dlpack_device__(self):
            return (int(DLDeviceType.kDLCUDA), 0)

        def __dlpack__(self, **kwargs):
            raise AssertionError("__dlpack__ must not be reached")

    with pytest.raises(BufferError, match="stream=None is ambiguous"):
        StridedMemoryView.from_dlpack(_FakeCudaDevice(), stream_ptr=None)


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
# Pointers use PYFUNCTYPE so a failing call raises its real Python exception
# (TypeError/RuntimeError/NotImplementedError).
#
# dlpack.h documents every `*_no_sync` entry point as returning "-1 on failure
# with a Python exception set", so every failure below is asserted with
# `pytest.raises`. A test that settles for `assert rc == -1` would be asserting a
# contract violation, not the contract.
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


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


class _DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [
        ("version", _DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", _DLTensor),
    ]


# DLPACK_FLAG_BITMASK_READ_ONLY in dlpack.h.
_FLAG_READ_ONLY = 1 << 0


class _VersionedCapsuleExport:
    def __init__(self, base, capsule):
        self.base = base
        self.capsule = capsule

    def __dlpack_device__(self):
        return self.base.__dlpack_device__()

    def __dlpack__(self, **kwargs):
        return self.capsule


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_dlpack_versioned_readonly_export_and_import(init_cuda):
    """The versioned readonly flag survives a StridedMemoryView round-trip."""
    src = np.arange(4, dtype=np.int32)
    src.setflags(write=False)
    base = StridedMemoryView.from_array_interface(src)
    capsule = base.__dlpack__(max_version=(1, 0))
    dlm = ctypes.cast(
        _PyCapsule_GetPointer(capsule, b"dltensor_versioned"),
        ctypes.POINTER(_DLManagedTensorVersioned),
    )
    assert dlm.contents.flags & _FLAG_READ_ONLY

    imported = StridedMemoryView.from_dlpack(
        _VersionedCapsuleExport(base, capsule),
        stream_ptr=-1,
    )
    assert imported.readonly is True


@pytest.mark.parametrize(
    ("code", "bits", "lanes", "exception", "match"),
    [
        pytest.param(0, 32, 2, NotImplementedError, "vector dtypes", id="lanes"),
        pytest.param(1, 24, 1, TypeError, "uint24", id="uint-bits"),
        pytest.param(0, 24, 1, TypeError, "int24", id="int-bits"),
        pytest.param(2, 8, 1, TypeError, "float8", id="float-bits"),
        pytest.param(5, 32, 1, TypeError, "complex32", id="complex-bits"),
        pytest.param(6, 1, 1, TypeError, "1-bit bool", id="bool-bits"),
        pytest.param(255, 8, 1, TypeError, "Unsupported dtype", id="code"),
    ],
)
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_from_dlpack_malformed_dtype_rejected_on_access(code, bits, lanes, exception, match):
    """Accessing ``.dtype`` rejects malformed producer dtype metadata."""
    base = StridedMemoryView.from_any_interface(np.arange(4, dtype=np.int32), stream_ptr=-1)
    capsule = base.__dlpack__(max_version=(1, 0))
    dlm = ctypes.cast(
        _PyCapsule_GetPointer(capsule, b"dltensor_versioned"),
        ctypes.POINTER(_DLManagedTensorVersioned),
    )
    dlm.contents.dl_tensor.dtype = _DLDataType(code, bits, lanes)
    imported = StridedMemoryView.from_dlpack(
        _VersionedCapsuleExport(base, capsule),
        stream_ptr=-1,
    )
    with pytest.raises(exception, match=match):
        _ = imported.dtype


@pytest.mark.agent_authored(model="cursor-grok-4.5")
@pytest.mark.parametrize(
    "max_version, capsule_name, managed_cls",
    [
        pytest.param(None, b"dltensor", _DLManagedTensor, id="unversioned"),
        pytest.param((1, 0), b"dltensor_versioned", _DLManagedTensorVersioned, id="versioned"),
    ],
)
def test_from_dlpack_null_deleter_dealloc(max_version, capsule_name, managed_cls):
    """``__dealloc__`` must tolerate a capsule whose deleter is NULL."""
    src = np.arange(6, dtype=np.int32)
    base = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    capsule = base.__dlpack__(max_version=max_version)
    dlm = ctypes.cast(_PyCapsule_GetPointer(capsule, capsule_name), ctypes.POINTER(managed_cls))
    # Steal the producer deleter so __dealloc__ sees NULL, then invoke it ourselves.
    producer_deleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(managed_cls))(dlm.contents.deleter)
    dlm.contents.deleter = None

    class _Export:
        def __dlpack_device__(self):
            return base.__dlpack_device__()

        def __dlpack__(self, stream=None, max_version=None, **kwargs):
            if capsule_name == b"dltensor" and max_version is not None:
                raise TypeError("force unversioned")
            return capsule

    view = StridedMemoryView.from_dlpack(_Export(), stream_ptr=-1)
    del view  # __dealloc__ must not call the NULL deleter
    producer_deleter(dlm)


@pytest.mark.agent_authored(model="claude-opus-5")
@pytest.mark.parametrize(
    "max_version, capsule_name, managed_cls",
    [
        pytest.param(None, b"dltensor", _DLManagedTensor, id="unversioned"),
        pytest.param((1, 0), b"dltensor_versioned", _DLManagedTensorVersioned, id="versioned"),
    ],
)
def test_from_dlpack_honours_byte_offset(max_version, capsule_name, managed_cls):
    """DLPack puts a tensor's first element at ``data + byte_offset``, so a producer
    may report the allocation base in ``data`` and express a slice as an offset.
    ``view.ptr`` must account for it, as the capsule-consuming path already does."""
    src = np.arange(9, dtype=np.int32)
    # View only the first 8 elements, so shifting by one element below stays inside
    # the allocation and a regression fails an assertion instead of reading OOB.
    base = StridedMemoryView.from_any_interface(src[:8], stream_ptr=-1)
    capsule = base.__dlpack__(max_version=max_version)
    dlm = ctypes.cast(_PyCapsule_GetPointer(capsule, capsule_name), ctypes.POINTER(managed_cls))
    assert dlm.contents.dl_tensor.data == src.ctypes.data
    assert dlm.contents.dl_tensor.byte_offset == 0

    # Re-describe the same 8 elements as src[1:9]. byte_offset is the only field
    # written: shape and strides share one producer-owned block, so leave them alone.
    dlm.contents.dl_tensor.byte_offset = src.itemsize

    class _Export:
        def __dlpack_device__(self):
            return base.__dlpack_device__()

        def __dlpack__(self, stream=None, max_version=None, **kwargs):
            if capsule_name == b"dltensor" and max_version is not None:
                raise TypeError("force unversioned")
            return capsule

    view = StridedMemoryView.from_dlpack(_Export(), stream_ptr=-1)
    assert view.ptr == src.ctypes.data + src.itemsize
    assert view.shape == (8,)
    assert np.array_equal(np.from_dlpack(view), src[1:])


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


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_dlpack_c_exchange_api_current_work_stream_null_output():
    """``current_work_stream`` rejects a NULL output pointer."""
    api = _get_exchange_api()
    with pytest.raises(RuntimeError, match="out_current_stream cannot be NULL"):
        api.current_work_stream(int(DLDeviceType.kDLCPU), 0, None)


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


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_dlpack_c_exchange_api_dltensor_from_py_object_null_output():
    """``dltensor_from_py_object_no_sync`` rejects a NULL output pointer."""
    api = _get_exchange_api()
    view = StridedMemoryView.from_any_interface(np.arange(3), stream_ptr=-1)
    with pytest.raises(RuntimeError, match="out cannot be NULL"):
        api.dltensor_from_py_object_no_sync(id(view), None)


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_dlpack_c_exchange_api_dltensor_from_py_object_scalar():
    """A borrowed scalar DLTensor has NULL shape and strides pointers."""
    api = _get_exchange_api()
    view = StridedMemoryView.from_any_interface(np.array(7, dtype=np.int16), stream_ptr=-1)
    out = _DLTensor()
    assert api.dltensor_from_py_object_no_sync(id(view), ctypes.byref(out)) == 0
    assert out.ndim == 0
    assert not out.shape
    assert not out.strides


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


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_dlpack_c_exchange_api_managed_tensor_from_py_object_errors():
    """The managed-tensor producer validates both output and object inputs."""
    api = _get_exchange_api()
    view = StridedMemoryView.from_any_interface(np.arange(3), stream_ptr=-1)
    with pytest.raises(RuntimeError, match="out cannot be NULL"):
        api.managed_tensor_from_py_object_no_sync(id(view), None)

    not_a_view = object()
    out = ctypes.c_void_p()
    with pytest.raises(TypeError, match="must be a StridedMemoryView"):
        api.managed_tensor_from_py_object_no_sync(id(not_a_view), ctypes.byref(out))
    assert not out.value


@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_dlpack_c_exchange_api_to_py_object_null_output():
    """``managed_tensor_to_py_object_no_sync`` rejects a NULL output pointer."""
    api = _get_exchange_api()
    tensor = _DLManagedTensorVersioned()
    with pytest.raises(RuntimeError, match="out_py_object cannot be NULL"):
        api.managed_tensor_to_py_object_no_sync(ctypes.byref(tensor), None)


def test_dlpack_c_exchange_api_to_py_object_null_tensor():
    """``managed_tensor_to_py_object_no_sync`` rejects a NULL tensor (RuntimeError)."""
    api = _get_exchange_api()
    out_obj = ctypes.c_void_p(0)
    with pytest.raises(RuntimeError, match="tensor cannot be NULL"):
        api.managed_tensor_to_py_object_no_sync(None, ctypes.byref(out_obj))
    assert not out_obj.value  # set to NULL before the error


@pytest.mark.parametrize(
    "device_type",
    [
        DLDeviceType.kDLCUDA,
        DLDeviceType.kDLCUDAHost,
        DLDeviceType.kDLCUDAManaged,
    ],
)
@pytest.mark.agent_authored(model="gpt-5.6-sol")
def test_dlpack_c_exchange_api_to_py_object_device_accessible(device_type):
    """Supported CUDA-family devices produce device-accessible views."""
    api = _get_exchange_api()
    tensor = _DLManagedTensorVersioned()
    tensor.version = _DLPackVersion(1, 0)
    tensor.dl_tensor.device = _DLDevice(int(device_type), 0)
    tensor.dl_tensor.dtype = _DLDataType(0, 32, 1)
    out_obj = ctypes.c_void_p()
    assert api.managed_tensor_to_py_object_no_sync(ctypes.byref(tensor), ctypes.byref(out_obj)) == 0
    assert out_obj.value
    try:
        imported = ctypes.cast(out_obj, ctypes.py_object).value
        assert imported.is_device_accessible is True
        assert imported.device_id == 0
        del imported
    finally:
        # The C API returned a new reference. Release it while the synthetic
        # tensor backing the view is still alive -- __dealloc__ dereferences it.
        _Py_DecRef(out_obj)


def test_dlpack_c_exchange_api_managed_tensor_allocator_not_supported():
    """Covers the ``managed_tensor_allocator`` entry point, which is unsupported
    and only ever raises NotImplementedError (StridedMemoryView never allocates)."""
    api = _get_exchange_api()
    out = ctypes.c_void_p(123)
    with pytest.raises(NotImplementedError, match="not supported"):
        # Currently sets a Python error when `SetError` isn't passed.
        api.managed_tensor_allocator(None, ctypes.byref(out), None, None)
    assert not out.value  # set to NULL before the error
