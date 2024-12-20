# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

try:
    import cupy as cp
except ImportError:
    cp = None
try:
    from numba import cuda as numba_cuda
except ImportError:
    numba_cuda = None
import numpy as np
import pytest

from cuda.core.experimental import Device
from cuda.core.experimental.utils import StridedMemoryView, args_viewable_as_strided_memory


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
