# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import math

try:
    import cupy as cp
except ImportError:
    cp = None
try:
    from numba import cuda as numba_cuda
except ImportError:
    numba_cuda = None
import cuda.core.experimental
import numpy as np
import pytest
from cuda.core.experimental import Device
from cuda.core.experimental._memoryview import view_as_cai
from cuda.core.experimental.utils import StridedLayout, StridedMemoryView, args_viewable_as_strided_memory


def test_cast_to_3_tuple_success():
    c3t = cuda.core.experimental._utils.cuda_utils.cast_to_3_tuple
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
        cuda.core.experimental._utils.cuda_utils.cast_to_3_tuple("Lbl", cfg)


def convert_strides_to_counts(strides, itemsize):
    return tuple(s // itemsize for s in strides)


@pytest.mark.parametrize(
    "in_arr,",
    (
        np.empty(3, dtype=np.int32),
        np.empty((6, 6), dtype=np.float64)[::2, ::2],
        np.empty((3, 4), order="F"),
        np.empty((), dtype=np.float16),
        # readonly is fixed recently (numpy/numpy#26501)
        pytest.param(
            np.frombuffer(b""),
            marks=pytest.mark.skipif(
                tuple(int(i) for i in np.__version__.split(".")[:2]) < (2, 1), reason="need numpy 2.1.0+"
            ),
        ),
    ),
)
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
        view = StridedMemoryView(in_arr, stream_ptr=-1)
        self._check_view(view, in_arr)

    def _check_view(self, view, in_arr):
        assert isinstance(view, StridedMemoryView)
        assert view.ptr == in_arr.ctypes.data
        assert view.shape == in_arr.shape
        strides_in_counts = convert_strides_to_counts(in_arr.strides, in_arr.dtype.itemsize)
        if in_arr.flags.c_contiguous:
            assert view.strides is None
        else:
            assert view.strides == strides_in_counts
        assert view.dtype == in_arr.dtype
        assert view.device_id == -1
        assert view.is_device_accessible is False
        assert view.exporting_obj is in_arr
        assert view.readonly is not in_arr.flags.writeable


def gpu_array_samples():
    # TODO: this function would initialize the device at test collection time
    samples = []
    if cp is not None:
        samples += [
            (cp.empty(3, dtype=cp.complex64), False),
            (cp.empty((6, 6), dtype=cp.float64)[::2, ::2], True),
            (cp.empty((3, 4), order="F"), True),
        ]
    # Numba's device_array is the only known array container that does not
    # support DLPack (so that we get to test the CAI coverage).
    if numba_cuda is not None:
        samples += [
            (numba_cuda.device_array((2,), dtype=np.int8), False),
            (numba_cuda.device_array((4, 2), dtype=np.float32), True),
        ]
    return samples


def gpu_array_ptr(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return arr.data.ptr
    if numba_cuda is not None and isinstance(arr, numba_cuda.cudadrv.devicearray.DeviceNDArray):
        return arr.device_ctypes_pointer.value
    raise NotImplementedError(f"{arr=}")


@pytest.mark.parametrize("in_arr,use_stream", (*gpu_array_samples(),))
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

        view = StridedMemoryView(in_arr, stream_ptr=s.handle if s else -1)
        self._check_view(view, in_arr, dev)

    def _check_view(self, view, in_arr, dev):
        assert isinstance(view, StridedMemoryView)
        assert view.ptr == gpu_array_ptr(in_arr)
        assert view.shape == in_arr.shape
        strides_in_counts = convert_strides_to_counts(in_arr.strides, in_arr.dtype.itemsize)
        if in_arr.flags["C_CONTIGUOUS"]:
            assert view.strides in (None, strides_in_counts)
        else:
            assert view.strides == strides_in_counts
        assert view.dtype == in_arr.dtype
        assert view.device_id == dev.device_id
        assert view.is_device_accessible is True
        assert view.exporting_obj is in_arr
        # can't test view.readonly with CuPy or Numba...


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
        view = view_as_cai(in_arr, stream_ptr=s.handle if s else -1)
        self._check_view(view, in_arr, dev)

    def _check_view(self, view, in_arr, dev):
        assert isinstance(view, StridedMemoryView)
        assert view.ptr == gpu_array_ptr(in_arr)
        assert view.shape == in_arr.shape
        strides_in_counts = convert_strides_to_counts(in_arr.strides, in_arr.dtype.itemsize)
        if in_arr.flags["C_CONTIGUOUS"]:
            assert view.strides is None
        else:
            assert view.strides == strides_in_counts
        assert view.dtype == in_arr.dtype
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


@pytest.mark.parametrize("shape", [tuple(), (2, 3), (10, 10), (10, 13, 11)])
@pytest.mark.parametrize("itemsize", [1, 4])
@pytest.mark.parametrize("stride_order", ["C", "F"])
@pytest.mark.parametrize("readonly", [True, False])
def test_from_buffer(shape, itemsize, stride_order, readonly):
    dev = Device()
    dev.set_current()
    layout = StridedLayout.dense(shape=shape, itemsize=itemsize, stride_order=stride_order)
    required_size = layout.required_size_in_bytes()
    assert required_size == math.prod(shape) * itemsize
    buffer = dev.memory_resource.allocate(required_size)
    view = StridedMemoryView.from_buffer(buffer, layout, is_readonly=readonly)
    assert view.exporting_obj is buffer
    assert view.layout is layout
    assert view.ptr == int(buffer.handle)
    assert view.shape == shape
    assert view.strides == _dense_strides(shape, stride_order)
    assert view.dtype is None
    assert view.device_id == dev.device_id
    assert view.is_device_accessible
    assert view.readonly == readonly


@pytest.mark.parametrize("stride_order", ["C", "F"])
def test_from_buffer_sliced(stride_order):
    layout = StridedLayout.dense((5, 7), 2, stride_order=stride_order)
    device = Device()
    device.set_current()
    buffer = device.memory_resource.allocate(layout.required_size_in_bytes())
    view = StridedMemoryView.from_buffer(buffer, layout)
    assert view.shape == (5, 7)

    sliced_view = view.view(layout[:-2, 3:])
    assert sliced_view.shape == (3, 4)
    expected_offset = 3 if stride_order == "C" else 3 * 5
    assert sliced_view.layout.slice_offset == expected_offset
    assert sliced_view.layout.slice_offset_in_bytes == expected_offset * 2
    assert sliced_view.ptr == view.ptr + expected_offset * 2


def test_from_buffer_too_small():
    layout = StridedLayout.dense((5, 4), 2)
    d = Device()
    d.set_current()
    buffer = d.memory_resource.allocate(20)
    with pytest.raises(ValueError, match="Expected at least 40 bytes, got 20 bytes."):
        StridedMemoryView.from_buffer(buffer, layout)


def test_from_buffer_disallowed_negative_offset():
    layout = StridedLayout((5, 4), (-4, 1), 1)
    d = Device()
    d.set_current()
    buffer = d.memory_resource.allocate(20)
    with pytest.raises(ValueError, match="please use StridedLayout.to_dense()."):
        StridedMemoryView.from_buffer(buffer, layout)


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
def test_from_buffer_sliced_external(shape, slices, stride_order, view_as):
    if view_as == "dlpack":
        if np is None:
            pytest.skip("NumPy is not installed")
        a = np.arange(math.prod(shape), dtype=np.int32).reshape(shape, order=stride_order)
        view = StridedMemoryView(a, -1)
    else:
        if cp is None:
            pytest.skip("CuPy is not installed")
        a = cp.arange(math.prod(shape), dtype=cp.int32).reshape(shape, order=stride_order)
        view = StridedMemoryView(_EnforceCAIView(a), -1)
    layout = view.layout
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
    assert sliced_view.layout is sliced_layout
    assert view.dtype == sliced_view.dtype
    assert sliced_view.layout.itemsize == a_sliced.itemsize == layout.itemsize
    assert sliced_view.shape == a_sliced.shape
    assert sliced_view.layout.strides_in_bytes == a_sliced.strides
