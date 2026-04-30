# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import math

# TODO: replace optional imports with pytest.importorskip
try:
    import cupy as cp
except ImportError:
    cp = None
try:
    from numba import cuda as numba_cuda
except ImportError:
    numba_cuda = None
try:
    import torch
except ImportError:
    torch = None
import cuda.core

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None
import numpy as np
import pytest
from helpers.marks import requires_module

from cuda.core import Device
from cuda.core._dlpack import DLDeviceType
from cuda.core._layout import _StridedLayout
from cuda.core.utils import StridedMemoryView, args_viewable_as_strided_memory

_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.argtypes = (ctypes.py_object, ctypes.c_char_p)
_PyCapsule_IsValid.restype = ctypes.c_int


def _get_cupy_version_major() -> int | None:
    if cp is None:
        return None
    return int(cp.__version__.split(".")[0])


def test_cast_to_3_tuple_success():
    c3t = cuda.core._utils.cuda_utils.cast_to_3_tuple
    assert c3t("", ()) == (1, 1, 1)
    assert c3t("", 2) == (2, 1, 1)
    assert c3t("", (2,)) == (2, 1, 1)
    assert c3t("", (2, 3)) == (2, 3, 1)
    assert c3t("", (2, 3, 4)) == (2, 3, 4)


_cast_to_3_tuple_value_error_test_cases = {
    "not tuple": ([], r"^Lbl must be an int, or a tuple with up to 3 ints \(got .*\)$"),
    "len 4": ((1, 2, 3, 4), r"^Lbl must be an int, or a tuple with up to 3 ints \(got tuple with length 4\)$"),
    "not int": (("bAd",), r"^Lbl must be an int, or a tuple with up to 3 ints \(got \('bAd',\)\)$"),
    "isolated negative": (-9, r"^Lbl value must be >= 1 \(got -9\)$"),
    "tuple negative": ((-9,), r"^Lbl value must be >= 1 \(got \(-9,\)\)$"),
}


@pytest.mark.parametrize(
    ("cfg", "expected"),
    _cast_to_3_tuple_value_error_test_cases.values(),
    ids=_cast_to_3_tuple_value_error_test_cases.keys(),
)
def test_cast_to_3_tuple_value_error(cfg, expected):
    with pytest.raises(ValueError, match=expected):
        cuda.core._utils.cuda_utils.cast_to_3_tuple("Lbl", cfg)


def convert_strides_to_counts(strides, itemsize):
    return tuple(s // itemsize for s in strides)


def _arr_ptr(arr):
    """Return the data pointer of *arr* regardless of its type."""
    if torch is not None and isinstance(arr, torch.Tensor):
        return arr.data_ptr()
    if isinstance(arr, np.ndarray):
        return arr.ctypes.data
    return gpu_array_ptr(arr)


def _arr_strides_in_counts(arr):
    """Return strides in element counts for *arr* regardless of its type."""
    if torch is not None and isinstance(arr, torch.Tensor):
        return tuple(arr.stride())
    return convert_strides_to_counts(arr.strides, arr.dtype.itemsize)


def _arr_size(arr):
    """Return the number of elements in *arr*."""
    if torch is not None and isinstance(arr, torch.Tensor):
        return arr.numel()
    return arr.size


def _arr_is_c_contiguous(arr):
    if torch is not None and isinstance(arr, torch.Tensor):
        return arr.is_contiguous()
    return arr.flags.c_contiguous if hasattr(arr, "flags") else arr.flags["C_CONTIGUOUS"]


def _arr_is_writeable(arr):
    if torch is not None and isinstance(arr, torch.Tensor):
        return True  # torch tensors are writable by default
    return arr.flags.writeable if hasattr(arr.flags, "writeable") else True


def _cpu_array_samples():
    samples = [
        np.empty(3, dtype=np.int32),
        np.empty((6, 6), dtype=np.float64)[::2, ::2],
        np.empty((3, 4), order="F"),
        np.empty((), dtype=np.float16),
        # readonly is fixed recently (numpy/numpy#26501)
        pytest.param(
            np.frombuffer(b""),
            marks=requires_module(np, "2.1"),
        ),
    ]
    if torch is not None:
        samples += [
            pytest.param(torch.arange(12, dtype=torch.float32), id="torch-1d"),
            pytest.param(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4), id="torch-nd"),
            pytest.param(torch.tensor(42.0), id="torch-scalar"),
            pytest.param(torch.empty(0, dtype=torch.float32), id="torch-empty"),
            pytest.param(
                torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
                id="torch-non-contiguous",
            ),
            pytest.param(torch.arange(100, dtype=torch.int64)[10:20], id="torch-sliced"),
            pytest.param(
                torch.arange(60, dtype=torch.float32).reshape(6, 10)[1:4, 2:7],
                id="torch-sliced-2d",
            ),
        ]
    return samples


@pytest.mark.parametrize("in_arr,", _cpu_array_samples())
class TestViewCPU:
    def test_args_viewable_as_strided_memory_cpu(self, in_arr):
        @args_viewable_as_strided_memory((0,))
        def my_func(arr):
            # stream_ptr=-1 means "the consumer does not care"
            view = arr.view(-1)
            self._check_view(view, in_arr)

        my_func(in_arr)

    def test_strided_memory_view_cpu(self, in_arr):
        # stream_ptr=-1 means "the consumer does not care"
        view = StridedMemoryView.from_any_interface(in_arr, stream_ptr=-1)
        self._check_view(view, in_arr)

    def test_strided_memory_view_cpu_init(self, in_arr):
        # stream_ptr=-1 means "the consumer does not care"
        with pytest.deprecated_call(match="deprecated"):
            view = StridedMemoryView(in_arr, stream_ptr=-1)
        self._check_view(view, in_arr)

    def _check_view(self, view, in_arr):
        assert isinstance(view, StridedMemoryView)
        assert view.ptr == _arr_ptr(in_arr)
        expected_shape = tuple(in_arr.shape)
        assert view.shape == expected_shape
        assert view.size == _arr_size(in_arr)
        strides_in_counts = _arr_strides_in_counts(in_arr)
        assert (_arr_is_c_contiguous(in_arr) and view.strides is None) or view.strides == strides_in_counts
        assert view.device_id == -1
        assert view.is_device_accessible is False
        assert view.exporting_obj is in_arr
        assert view.readonly is not _arr_is_writeable(in_arr)


def gpu_array_samples():
    # TODO: this function would initialize the device at test collection time
    samples = []
    if cp is not None:
        samples += [
            pytest.param(cp.empty(3, dtype=cp.complex64), False, id="cupy-complex64"),
            pytest.param(cp.empty((6, 6), dtype=cp.float64)[::2, ::2], True, id="cupy-float64"),
            pytest.param(cp.empty((3, 4), order="F"), True, id="cupy-fortran"),
        ]
    # Numba's device_array is the only known array container that does not
    # support DLPack (so that we get to test the CAI coverage).
    if numba_cuda is not None:
        samples += [
            pytest.param(numba_cuda.device_array((2,), dtype=np.int8), False, id="numba-cuda-int8"),
            pytest.param(numba_cuda.device_array((4, 2), dtype=np.float32), True, id="numba-cuda-float32"),
        ]
    if torch is not None:
        samples += [
            pytest.param(torch.arange(12, dtype=torch.float32, device="cuda"), True, id="torch-1d"),
            pytest.param(
                torch.arange(24, dtype=torch.float32, device="cuda").reshape(2, 3, 4),
                True,
                id="torch-nd",
            ),
            pytest.param(torch.tensor(42.0, dtype=torch.float32, device="cuda"), False, id="torch-scalar"),
            pytest.param(torch.empty(0, dtype=torch.float32, device="cuda"), False, id="torch-empty"),
            pytest.param(
                torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4).t(),
                True,
                id="torch-non-contiguous",
            ),
            pytest.param(
                torch.arange(100, dtype=torch.int64, device="cuda")[10:20],
                True,
                id="torch-sliced",
            ),
            pytest.param(
                torch.arange(60, dtype=torch.float32, device="cuda").reshape(6, 10)[1:4, 2:7],
                True,
                id="torch-sliced-2d",
            ),
        ]
    return samples


def gpu_array_ptr(arr):
    if torch is not None and isinstance(arr, torch.Tensor):
        return arr.data_ptr()
    if cp is not None and isinstance(arr, cp.ndarray):
        return arr.data.ptr
    if numba_cuda is not None and isinstance(arr, numba_cuda.cudadrv.devicearray.DeviceNDArray):
        return arr.device_ctypes_pointer.value
    raise NotImplementedError(f"{arr=}")


@pytest.mark.parametrize(("in_arr", "use_stream"), gpu_array_samples())
class TestViewGPU:
    def test_args_viewable_as_strided_memory_gpu(self, in_arr, use_stream):
        # TODO: use the device fixture?
        dev = Device()
        dev.set_current()
        # This is the consumer stream
        s = dev.create_stream() if use_stream else None

        @args_viewable_as_strided_memory((0,))
        def my_func(arr):
            view = arr.view(s.handle if s else -1)
            self._check_view(view, in_arr, dev)

        my_func(in_arr)

    def test_strided_memory_view_cpu(self, in_arr, use_stream):
        # TODO: use the device fixture?
        dev = Device()
        dev.set_current()
        # This is the consumer stream
        s = dev.create_stream() if use_stream else None

        view = StridedMemoryView.from_any_interface(in_arr, stream_ptr=s.handle if s else -1)
        self._check_view(view, in_arr, dev)

    def test_strided_memory_view_init(self, in_arr, use_stream):
        # TODO: use the device fixture?
        dev = Device()
        dev.set_current()
        # This is the consumer stream
        s = dev.create_stream() if use_stream else None

        with pytest.deprecated_call(match="deprecated"):
            view = StridedMemoryView(in_arr, stream_ptr=s.handle if s else -1)
        self._check_view(view, in_arr, dev)

    def _check_view(self, view, in_arr, dev):
        assert isinstance(view, StridedMemoryView)
        assert view.ptr == gpu_array_ptr(in_arr)
        expected_shape = tuple(in_arr.shape)
        assert view.shape == expected_shape
        assert view.size == _arr_size(in_arr)
        strides_in_counts = _arr_strides_in_counts(in_arr)
        if _arr_is_c_contiguous(in_arr):
            assert view.strides in (None, strides_in_counts)
        else:
            assert view.strides == strides_in_counts
        assert view.device_id == dev.device_id
        assert view.is_device_accessible is True
        assert view.exporting_obj is in_arr
        # can't test view.readonly with CuPy, Numba, or torch...


def test_strided_memory_view_dlpack_export_numpy_roundtrip():
    src = np.arange(24, dtype=np.int32).reshape(4, 6)[:, ::2]
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    out = np.from_dlpack(view)
    assert out.shape == src.shape
    assert out.dtype == src.dtype
    assert np.array_equal(out, src)
    assert view.__dlpack_device__() == (int(DLDeviceType.kDLCPU), 0)


@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
def test_strided_memory_view_dlpack_export_cupy_roundtrip(init_cuda):
    src = cp.arange(24, dtype=cp.float32).reshape(4, 6)[:, ::2]
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    out = cp.from_dlpack(view)
    cp.testing.assert_array_equal(out, src)
    assert view.__dlpack_device__() == (int(DLDeviceType.kDLCUDA), init_cuda.device_id)


def test_strided_memory_view_dlpack_export_requires_dtype(init_cuda):
    buffer = init_cuda.memory_resource.allocate(16)
    view = StridedMemoryView.from_buffer(
        buffer,
        shape=(16,),
        itemsize=1,
        dtype=None,
    )
    with pytest.raises(BufferError, match="dtype"):
        view.__dlpack__()


def test_strided_memory_view_exposes_dlpack_c_exchange_api_capsule():
    capsule = StridedMemoryView.__dlpack_c_exchange_api__
    assert _PyCapsule_IsValid(capsule, b"dlpack_exchange_api") == 1
    # Backward-compatible alias.
    assert StridedMemoryView.__c_dlpack_exchange_api__ is capsule


@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
@pytest.mark.parametrize("in_arr,use_stream", (*gpu_array_samples(),))
class TestViewCudaArrayInterfaceGPU:
    def test_cuda_array_interface_gpu(self, in_arr, use_stream):
        # TODO: use the device fixture?
        dev = Device()
        dev.set_current()
        # This is the consumer stream
        s = dev.create_stream() if use_stream else None

        # The usual path in `StridedMemoryView` prefers the DLPack interface
        # over __cuda_array_interface__, so we call `view_as_cai` directly
        # here so we can test the CAI code path.
        view = StridedMemoryView.from_cuda_array_interface(in_arr, stream_ptr=s.handle if s else -1)
        self._check_view(view, in_arr, dev)

    def _check_view(self, view, in_arr, dev):
        assert isinstance(view, StridedMemoryView)
        assert view.ptr == _arr_ptr(in_arr)
        expected_shape = tuple(in_arr.shape)
        assert view.shape == expected_shape
        assert view.size == _arr_size(in_arr)
        strides_in_counts = _arr_strides_in_counts(in_arr)
        if _arr_is_c_contiguous(in_arr):
            assert view.strides is None
        else:
            assert view.strides == strides_in_counts
        assert view.device_id == dev.device_id
        assert view.is_device_accessible is True
        assert view.exporting_obj is in_arr


def _dense_strides(shape, stride_order):
    ndim = len(shape)
    strides = [None] * ndim
    if ndim > 0:
        if stride_order == "C":
            strides[-1] = 1
            for i in range(ndim - 2, -1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]
        else:
            assert stride_order == "F"
            strides[0] = 1
            for i in range(1, ndim):
                strides[i] = strides[i - 1] * shape[i - 1]
    return tuple(strides)


@pytest.mark.parametrize("shape", [(), (2, 3), (10, 10), (10, 13, 11)], ids=str)
@pytest.mark.parametrize("dtype", [np.dtype(np.int8), np.dtype(np.uint32)], ids=str)
@pytest.mark.parametrize("stride_order", ["C", "F"])
@pytest.mark.parametrize("readonly", [True, False])
def test_from_buffer(shape, dtype, stride_order, readonly):
    dev = Device()
    dev.set_current()
    layout = _StridedLayout.dense(shape=shape, itemsize=dtype.itemsize, stride_order=stride_order)
    required_size = layout.required_size_in_bytes()
    assert required_size == math.prod(shape) * dtype.itemsize
    buffer = dev.memory_resource.allocate(required_size)
    view = StridedMemoryView.from_buffer(buffer, shape=shape, strides=layout.strides, dtype=dtype, is_readonly=readonly)
    assert view.exporting_obj is buffer
    assert view._layout == layout
    assert view.ptr == int(buffer.handle)
    assert view.shape == shape
    assert view.strides == _dense_strides(shape, stride_order)
    assert view.dtype == dtype
    assert view.device_id == dev.device_id
    assert view.is_device_accessible
    assert view.readonly == readonly


@pytest.mark.parametrize(
    ("dtype", "itemsize", "msg"),
    [
        (np.dtype("int16"), 1, "itemsize .+ does not match dtype.itemsize .+"),
        (None, None, "itemsize or dtype must be specified"),
    ],
)
def test_from_buffer_incompatible_dtype_and_itemsize(dtype, itemsize, msg):
    layout = _StridedLayout.dense((5,), 2)
    device = Device()
    device.set_current()
    buffer = device.memory_resource.allocate(layout.required_size_in_bytes())
    with pytest.raises(ValueError, match=msg):
        StridedMemoryView.from_buffer(buffer, (5,), dtype=dtype, itemsize=itemsize)


@pytest.mark.parametrize("stride_order", ["C", "F"])
def test_from_buffer_sliced(stride_order):
    layout = _StridedLayout.dense((5, 7), 2, stride_order=stride_order)
    device = Device()
    device.set_current()
    buffer = device.memory_resource.allocate(layout.required_size_in_bytes())
    view = StridedMemoryView.from_buffer(buffer, (5, 7), dtype=np.dtype(np.int16))
    assert view.shape == (5, 7)
    assert int(buffer.handle) == view.ptr

    sliced_view = view.view(layout[:-2, 3:])
    assert sliced_view.shape == (3, 4)
    expected_offset = 3 if stride_order == "C" else 3 * 5
    assert sliced_view._layout.slice_offset == expected_offset
    assert sliced_view._layout.slice_offset_in_bytes == expected_offset * 2
    assert sliced_view.ptr == view.ptr + expected_offset * 2
    assert int(buffer.handle) + expected_offset * 2 == sliced_view.ptr


def test_from_buffer_too_small():
    layout = _StridedLayout.dense((5, 4), 2)
    d = Device()
    d.set_current()
    buffer = d.memory_resource.allocate(20)
    with pytest.raises(ValueError, match="Expected at least 40 bytes, got 20 bytes."):
        StridedMemoryView.from_buffer(
            buffer,
            shape=layout.shape,
            strides=layout.strides,
            dtype=np.dtype("int16"),
        )


def test_from_buffer_disallowed_negative_offset():
    layout = _StridedLayout((5, 4), (-4, 1), 1)
    d = Device()
    d.set_current()
    buffer = d.memory_resource.allocate(20)
    with pytest.raises(ValueError):
        StridedMemoryView.from_buffer(
            buffer,
            shape=layout.shape,
            strides=layout.strides,
            dtype=np.dtype("uint8"),
        )


class _EnforceCAIView:
    def __init__(self, array):
        self.array = array
        self.__cuda_array_interface__ = array.__cuda_array_interface__


def _get_ptr(array):
    if isinstance(array, np.ndarray):
        return array.ctypes.data
    else:
        assert isinstance(array, cp.ndarray)
        return array.data.ptr


@pytest.mark.parametrize(
    ("shape", "slices", "stride_order", "view_as"),
    [
        (shape, slices, stride_order, view_as)
        for shape, slices in [
            ((5, 6), (2, slice(1, -1))),
            ((10, 13, 11), (slice(None, None, 2), slice(None, None, -1), slice(2, -3))),
        ]
        for stride_order in ["C", "F"]
        for view_as in ["dlpack", "cai"]
    ],
)
def test_view_sliced_external(init_cuda, shape, slices, stride_order, view_as):
    if view_as == "dlpack":
        if np is None:
            pytest.skip("NumPy is not installed")
        a = np.arange(math.prod(shape), dtype=np.int32).reshape(shape, order=stride_order)
        view = StridedMemoryView.from_dlpack(a, -1)
    else:
        if cp is None:
            pytest.skip("CuPy is not installed")
        a = cp.arange(math.prod(shape), dtype=cp.int32).reshape(shape, order=stride_order)
        view = StridedMemoryView.from_cuda_array_interface(_EnforceCAIView(a), -1)
    layout = view._layout
    assert layout.is_dense
    assert layout.required_size_in_bytes() == a.nbytes
    assert view.ptr == _get_ptr(a)

    sliced_layout = layout[slices]
    sliced_view = view.view(sliced_layout)
    a_sliced = a[slices]
    assert sliced_view.ptr == _get_ptr(a_sliced)
    assert sliced_view.ptr != view.ptr

    assert 0 <= sliced_layout.required_size_in_bytes() <= a.nbytes
    assert not sliced_layout.is_dense
    assert sliced_view._layout is sliced_layout
    assert view.dtype == sliced_view.dtype
    assert sliced_view._layout.itemsize == a_sliced.itemsize == layout.itemsize
    assert sliced_view.shape == a_sliced.shape
    assert sliced_view._layout.strides_in_bytes == a_sliced.strides


@pytest.mark.parametrize(
    ("stride_order", "view_as"),
    [(stride_order, view_as) for stride_order in ["C", "F"] for view_as in ["dlpack", "cai"]],
)
def test_view_sliced_external_negative_offset(init_cuda, stride_order, view_as):
    shape = (5,)
    if view_as == "dlpack":
        if np is None:
            pytest.skip("NumPy is not installed")
        a = np.arange(math.prod(shape), dtype=np.int32).reshape(shape, order=stride_order)
        a = a[::-1]
        view = StridedMemoryView.from_dlpack(a, -1)
    else:
        if cp is None:
            pytest.skip("CuPy is not installed")
        a = cp.arange(math.prod(shape), dtype=cp.int32).reshape(shape, order=stride_order)
        a = a[::-1]
        view = StridedMemoryView.from_cuda_array_interface(_EnforceCAIView(a), -1)
    layout = view._layout
    assert not layout.is_dense
    assert layout.strides == (-1,)
    assert view.ptr == _get_ptr(a)

    sliced_layout = layout[3:]
    sliced_view = view.view(sliced_layout)
    a_sliced = a[3:]
    assert sliced_view.ptr == _get_ptr(a_sliced)
    assert sliced_view.ptr == view.ptr - 3 * a.itemsize

    assert not sliced_layout.is_dense
    assert sliced_view._layout is sliced_layout
    assert view.dtype == sliced_view.dtype
    assert sliced_view._layout.itemsize == a_sliced.itemsize == layout.itemsize
    assert sliced_view.shape == a_sliced.shape
    assert sliced_view._layout.strides_in_bytes == a_sliced.strides


@pytest.mark.parametrize(
    "api",
    [
        StridedMemoryView.from_dlpack,
        StridedMemoryView.from_cuda_array_interface,
    ],
)
@pytest.mark.parametrize("shape", [(0,), (0, 0), (0, 0, 0)])
@pytest.mark.parametrize("dtype", [np.int64, np.uint8, np.float64])
def test_view_zero_size_array(init_cuda, api, shape, dtype):
    cp = pytest.importorskip("cupy")

    x = cp.empty(shape, dtype=dtype)
    smv = api(x, stream_ptr=0)

    assert smv.size == 0
    assert smv.shape == shape
    assert smv.dtype == np.dtype(dtype)


def test_from_buffer_with_non_power_of_two_itemsize():
    dev = Device()
    dev.set_current()
    dtype = np.dtype([("a", "int32"), ("b", "int8")])
    shape = (1,)
    layout = _StridedLayout(shape=shape, strides=None, itemsize=dtype.itemsize)
    required_size = layout.required_size_in_bytes()
    assert required_size == math.prod(shape) * dtype.itemsize
    buffer = dev.memory_resource.allocate(required_size)
    view = StridedMemoryView.from_buffer(buffer, shape=shape, strides=layout.strides, dtype=dtype, is_readonly=True)
    assert view.dtype == dtype


def test_struct_array(init_cuda):
    cp = pytest.importorskip("cupy")

    x = np.array([(1.0, 2), (2.0, 3)], dtype=[("array1", np.float64), ("array2", np.int64)])

    y = cp.empty(2, dtype=x.dtype)
    y.set(x)

    smv = StridedMemoryView.from_cuda_array_interface(y, stream_ptr=0)
    assert smv.size * smv.dtype.itemsize == x.nbytes
    assert smv.size == x.size
    assert smv.shape == x.shape
    # full dtype information doesn't seem to be preserved due to use of type strings,
    # which are lossy, e.g., dtype([("a", "int")]).str == "V8"
    assert smv.dtype == np.dtype(f"V{x.itemsize}")


@pytest.mark.parametrize(
    ("x", "expected_dtype"),
    [
        # 1D arrays with different dtypes
        pytest.param(np.array([1, 2, 3], dtype=np.int32), "int32", id="1d-int32"),
        pytest.param(np.array([1.0, 2.0, 3.0], dtype=np.float64), "float64", id="1d-float64"),
        pytest.param(np.array([1 + 2j, 3 + 4j], dtype=np.complex128), "complex128", id="1d-complex128"),
        pytest.param(np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64), "complex64", id="1d-complex64"),
        pytest.param(np.array([1, 2, 3, 4, 5], dtype=np.uint8), "uint8", id="1d-uint8"),
        pytest.param(np.array([1, 2], dtype=np.int64), "int64", id="1d-int64"),
        pytest.param(np.array([100, 200, 300], dtype=np.int16), "int16", id="1d-int16"),
        pytest.param(np.array([1000, 2000, 3000], dtype=np.uint16), "uint16", id="1d-uint16"),
        pytest.param(np.array([10000, 20000, 30000], dtype=np.uint64), "uint64", id="1d-uint64"),
        # 2D arrays - C-contiguous
        pytest.param(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32), "int32", id="2d-c-int32"),
        pytest.param(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "float32", id="2d-c-float32"),
        # 2D arrays - Fortran-contiguous
        pytest.param(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32, order="F"), "int32", id="2d-f-int32"),
        pytest.param(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="F"), "float64", id="2d-f-float64"),
        # 3D arrays
        pytest.param(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int32), "int32", id="3d-int32"),
        pytest.param(np.ones((2, 3, 4), dtype=np.float64), "float64", id="3d-float64"),
        # Sliced/strided arrays
        pytest.param(np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)[::2], "int32", id="1d-strided-int32"),
        pytest.param(
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float64)[:, ::2], "float64", id="2d-strided-float64"
        ),
        pytest.param(np.arange(20, dtype=np.int32).reshape(4, 5)[::2, ::2], "int32", id="2d-strided-2x2-int32"),
        # Scalar (0-D array)
        pytest.param(np.array(42, dtype=np.int32), "int32", id="scalar-int32"),
        pytest.param(np.array(3.14, dtype=np.float64), "float64", id="scalar-float64"),
        # Empty arrays
        pytest.param(np.array([], dtype=np.int32), "int32", id="empty-1d-int32"),
        pytest.param(np.empty((0, 3), dtype=np.float64), "float64", id="empty-2d-float64"),
        # Single element
        pytest.param(np.array([1], dtype=np.int32), "int32", id="single-element"),
        # Structured dtype
        pytest.param(np.array([(1, 2.0), (3, 4.0)], dtype=[("a", "i4"), ("b", "f8")]), "V12", id="structured-dtype"),
    ],
)
def test_from_array_interface(x, init_cuda, expected_dtype):
    smv = StridedMemoryView.from_array_interface(x)
    assert smv.size == x.size
    assert smv.dtype == np.dtype(expected_dtype)
    assert smv.shape == x.shape
    assert smv.ptr == x.ctypes.data
    assert smv.device_id == init_cuda.device_id
    assert smv.is_device_accessible is False
    assert smv.exporting_obj is x
    assert smv.readonly is not x.flags.writeable
    # Check strides
    strides_in_counts = convert_strides_to_counts(x.strides, x.dtype.itemsize)
    assert (x.flags.c_contiguous and smv.strides is None) or smv.strides == strides_in_counts


def test_from_array_interface_unsupported_strides(init_cuda):
    # Create an array with strides that aren't a multiple of itemsize
    x = np.array([(1, 2.0), (3, 4.0)], dtype=[("a", "i4"), ("b", "f8")])
    b = x["b"]
    with pytest.raises(ValueError, match="strides must be divisible by itemsize"):
        StridedMemoryView.from_array_interface(b)


def _make_cuda_array_interface_obj(*, shape, strides, typestr="<f8", data=(0, False), version=3):
    return type(
        "SyntheticCAI",
        (),
        {
            "__cuda_array_interface__": {
                "shape": shape,
                "strides": strides,
                "typestr": typestr,
                "data": data,
                "version": version,
            }
        },
    )()


def test_from_cuda_array_interface_unsupported_strides(init_cuda):
    cai_obj = _make_cuda_array_interface_obj(shape=(2,), strides=(10,))
    with pytest.raises(ValueError, match="strides must be divisible by itemsize"):
        StridedMemoryView.from_cuda_array_interface(cai_obj, stream_ptr=-1)


def test_from_cuda_array_interface_zero_strides(init_cuda):
    cai_obj = _make_cuda_array_interface_obj(shape=(1, 1), strides=(0, 0))
    smv = StridedMemoryView.from_cuda_array_interface(cai_obj, stream_ptr=-1)
    assert smv.shape == (1, 1)
    assert smv.strides == (0, 0)


@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
def test_from_cuda_array_interface_negative_strides(init_cuda):
    x = cp.arange(4, dtype=cp.float64)[::-1]
    smv = StridedMemoryView.from_cuda_array_interface(_EnforceCAIView(x), stream_ptr=-1)
    assert smv.shape == x.shape
    assert smv.strides == (-1,)


def test_from_cuda_array_interface_empty_array(init_cuda):
    cai_obj = _make_cuda_array_interface_obj(shape=(0, 3), strides=(24, 8))
    smv = StridedMemoryView.from_cuda_array_interface(cai_obj, stream_ptr=-1)
    assert smv.size == 0
    assert smv.shape == (0, 3)
    assert smv.strides == (3, 1)


@pytest.mark.parametrize(
    "slices",
    [
        pytest.param((slice(None), slice(None)), id="contiguous"),
        pytest.param((slice(None, None, 2), slice(1, None, 2)), id="strided"),
    ],
)
@pytest.mark.skipif(ml_dtypes is None, reason="ml_dtypes is not installed")
@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
@pytest.mark.skipif(cp is not None and _get_cupy_version_major() < 14, reason="CuPy version is less than 14.0.0")
def test_ml_dtypes_bfloat16_dlpack(init_cuda, slices):
    a = cp.array([1, 2, 3, 4, 5, 6], dtype=ml_dtypes.bfloat16).reshape(2, 3)[slices]
    smv = StridedMemoryView.from_dlpack(a, stream_ptr=0)

    assert smv.size == a.size
    assert smv.dtype == np.dtype("bfloat16")
    assert smv.dtype == np.dtype(ml_dtypes.bfloat16)
    assert smv.shape == a.shape
    assert smv.ptr == a.data.ptr
    assert smv.device_id == init_cuda.device_id
    assert smv.is_device_accessible is True
    assert smv.exporting_obj is a
    assert smv.readonly is a.__cuda_array_interface__["data"][1]

    strides_in_counts = convert_strides_to_counts(a.strides, a.dtype.itemsize)
    if a.flags["C_CONTIGUOUS"]:
        assert smv.strides in (None, strides_in_counts)
    else:
        assert smv.strides == strides_in_counts


@pytest.mark.parametrize(
    "slices",
    [
        pytest.param((slice(None), slice(None)), id="contiguous"),
        pytest.param((slice(None, None, 2), slice(1, None, 2)), id="strided"),
    ],
)
@pytest.mark.skipif(ml_dtypes is None, reason="ml_dtypes is not installed")
@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_ml_dtypes_bfloat16_torch_dlpack(init_cuda, slices):
    a = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.bfloat16, device="cuda").reshape(2, 3)[slices]
    smv = StridedMemoryView.from_dlpack(a, stream_ptr=0)

    assert smv.size == a.numel()
    assert smv.dtype == np.dtype("bfloat16")
    assert smv.dtype == np.dtype(ml_dtypes.bfloat16)
    assert smv.shape == tuple(a.shape)
    assert smv.ptr == a.data_ptr()
    assert smv.device_id == init_cuda.device_id
    assert smv.is_device_accessible is True
    assert smv.exporting_obj is a

    # PyTorch stride() returns strides in elements, convert to bytes first
    strides_in_bytes = tuple(s * a.element_size() for s in a.stride())
    strides_in_counts = convert_strides_to_counts(strides_in_bytes, a.element_size())
    if a.is_contiguous():
        assert smv.strides in (None, strides_in_counts)
    else:
        assert smv.strides == strides_in_counts


@pytest.fixture
def no_ml_dtypes(monkeypatch):
    monkeypatch.setattr("cuda.core._memoryview.bfloat16", None)
    return


@pytest.mark.parametrize(
    "api",
    [
        pytest.param(StridedMemoryView.from_dlpack, id="from_dlpack"),
        pytest.param(StridedMemoryView.from_any_interface, id="from_any_interface"),
    ],
)
@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
@pytest.mark.skipif(cp is not None and _get_cupy_version_major() < 14, reason="CuPy version is less than 14.0.0")
def test_ml_dtypes_bfloat16_dlpack_requires_ml_dtypes(init_cuda, no_ml_dtypes, api):
    a = cp.array([1, 2, 3], dtype="bfloat16")
    smv = api(a, stream_ptr=0)
    with pytest.raises(NotImplementedError, match=r"requires `ml_dtypes`"):
        smv.dtype  # noqa: B018


def test_strided_memory_view_repr():
    """__repr__ returns a descriptive string."""
    src = np.arange(6, dtype=np.int32).reshape(2, 3)
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    r = repr(view)
    assert r.startswith("StridedMemoryView(ptr=")


def test_strided_memory_view_copy_to_raises():
    """copy_to raises NotImplementedError."""
    src = np.zeros(5, dtype=np.float32)
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    with pytest.raises(NotImplementedError, match="copy_to"):
        view.copy_to(view, stream=None)


def test_strided_memory_view_get_layout_error():
    """get_layout raises ValueError for an empty (uninitialized) StridedMemoryView."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        view = StridedMemoryView()
    with pytest.raises(ValueError, match="Cannot infer layout"):
        _ = view._layout


@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
def test_strided_memory_view_deprecated_cai_init(init_cuda):
    """Deprecated StridedMemoryView(cai_obj) init path for CAI-only objects."""
    src = cp.zeros(5, dtype=cp.float32)
    dev = Device()
    stream = dev.create_stream()
    cai_only = _EnforceCAIView(src)
    with pytest.deprecated_call():
        view = StridedMemoryView(cai_only, stream_ptr=stream.handle)
    assert view.is_device_accessible is True
    assert view.ptr == src.data.ptr


@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
def test_from_any_interface_cai_fallback(init_cuda):
    """from_any_interface falls back to CAI when an object has no __dlpack__."""
    src = cp.zeros(5, dtype=cp.float32)
    dev = Device()
    stream = dev.create_stream()
    cai_only = _EnforceCAIView(src)
    view = StridedMemoryView.from_any_interface(cai_only, stream_ptr=stream.handle)
    assert view.is_device_accessible is True
    assert view.ptr == src.data.ptr


def test_strided_memory_view_copy_from_raises():
    """copy_from raises NotImplementedError."""
    src = np.zeros(5, dtype=np.float32)
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    with pytest.raises(NotImplementedError, match="copy_from"):
        view.copy_from(view, stream=None)


def test_strided_memory_view_view_no_args_returns_self():
    """view() with layout=None and dtype=None returns self."""
    src = np.arange(6, dtype=np.int32).reshape(2, 3)
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    same = view.view(layout=None, dtype=None)
    assert same is view


def test_strided_memory_view_view_with_dtype_only():
    """view() with only dtype re-interprets using current layout."""
    src = np.arange(4, dtype=np.float32)
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    viewed = view.view(dtype=np.dtype("int32"))
    assert viewed.dtype == np.dtype("int32")
    assert viewed._layout == view._layout


def test_dlpack_export_structured_dtype_raises():
    """Structured dtypes are rejected for DLPack export."""
    dt = np.dtype([("x", np.float32), ("y", np.int32)])  # itemsize=8
    # Create a valid view first, then re-view with the structured dtype to
    # bypass numpy's own __dlpack__ rejection during import.
    src = np.zeros(3, dtype=np.float64)  # itemsize=8 to match
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    bad_view = view.view(dtype=dt)
    with pytest.raises(BufferError, match="Structured dtypes"):
        bad_view.__dlpack__()


def test_dlpack_export_unsupported_dtype_raises():
    """Unsupported dtype kind is rejected for DLPack export."""
    # numpy void dtype (kind='V', typestr='|V4') hits the else branch
    # in _smv_dtype_numpy_to_dlpack at _memoryview.pyx:577
    src = np.zeros(3, dtype=np.float32)  # itemsize=4 to match V4
    view = StridedMemoryView.from_any_interface(src, stream_ptr=-1)
    bad_view = view.view(dtype=np.dtype("V4"))
    with pytest.raises(BufferError, match="Unsupported dtype for DLPack export"):
        bad_view.__dlpack__()


class _FakeCAIv2:
    """Object with CUDA Array Interface v2 (unsupported)."""

    def __init__(self):
        self.__cuda_array_interface__ = {
            "version": 2,
            "shape": (5,),
            "typestr": "<f4",
            "data": (0, False),
        }


class _FakeCAIWithMask:
    """Object with CUDA Array Interface that has a mask."""

    def __init__(self):
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": (5,),
            "typestr": "<f4",
            "data": (0, False),
            "mask": np.ones(5, dtype=bool),
        }


class _FakeArrayInterfacev2:
    """Object with NumPy Array Interface v2 (unsupported)."""

    def __init__(self, arr):
        iface = dict(arr.__array_interface__)
        iface["version"] = 2
        self.__array_interface__ = iface


class _FakeArrayInterfaceWithMask:
    """Object with NumPy Array Interface that has a mask."""

    def __init__(self, arr):
        iface = dict(arr.__array_interface__)
        iface["mask"] = np.ones(arr.shape, dtype=bool)
        self.__array_interface__ = iface


def test_cai_v2_rejected():
    """CUDA Array Interface v2 raises BufferError."""
    from cuda.core._memoryview import view_as_cai

    obj = _FakeCAIv2()
    with pytest.raises(BufferError, match="v3 or above"):
        view_as_cai(obj, stream_ptr=-1)


def test_cai_mask_rejected():
    """CUDA Array Interface with mask raises BufferError."""
    from cuda.core._memoryview import view_as_cai

    obj = _FakeCAIWithMask()
    with pytest.raises(BufferError, match="mask is not supported"):
        view_as_cai(obj, stream_ptr=-1)


class _FakeCAIv3:
    """Valid CUDA Array Interface v3 object (for stream=None test)."""

    def __init__(self):
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": (5,),
            "typestr": "<f4",
            "data": (0, False),
        }


def test_cai_stream_none_rejected():
    """CUDA Array Interface with stream=None raises BufferError."""
    from cuda.core._memoryview import view_as_cai

    obj = _FakeCAIv3()
    with pytest.raises(BufferError, match="stream=None is ambiguous"):
        view_as_cai(obj, stream_ptr=None)


def test_array_interface_v2_rejected():
    """NumPy Array Interface v2 raises BufferError."""
    from cuda.core._memoryview import view_as_array_interface

    arr = np.zeros(5, dtype=np.float32)
    obj = _FakeArrayInterfacev2(arr)
    with pytest.raises(BufferError, match="v3 or above"):
        view_as_array_interface(obj)


def test_array_interface_mask_rejected():
    """NumPy Array Interface with mask raises BufferError."""
    from cuda.core._memoryview import view_as_array_interface

    arr = np.zeros(5, dtype=np.float32)
    obj = _FakeArrayInterfaceWithMask(arr)
    with pytest.raises(BufferError, match="mask is not supported"):
        view_as_array_interface(obj)


_torch_skip = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


@_torch_skip
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float16", id="float16"),
        pytest.param("float32", id="float32"),
        pytest.param("float64", id="float64"),
        pytest.param("int8", id="int8"),
        pytest.param("int16", id="int16"),
        pytest.param("int32", id="int32"),
        pytest.param("int64", id="int64"),
        pytest.param("uint8", id="uint8"),
        pytest.param("bool", id="bool"),
        pytest.param("complex64", id="complex64"),
        pytest.param("complex128", id="complex128"),
        pytest.param(
            "bfloat16",
            id="bfloat16",
            marks=pytest.mark.skipif(ml_dtypes is None, reason="ml_dtypes is not installed"),
        ),
    ],
)
def test_torch_tensor_bridge_dtypes(init_cuda, dtype):
    """Verify that dtype mapping via the tensor bridge matches torch's own dtype."""
    torch_dtype = getattr(torch, dtype)
    a = torch.tensor([1, 0, 1], dtype=torch_dtype, device="cuda")
    smv = StridedMemoryView.from_any_interface(a, stream_ptr=0)
    assert smv.dtype.itemsize == a.element_size()
    assert smv.ptr == a.data_ptr()
