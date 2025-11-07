# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cuda.core.experimental
import numpy as np
import pytest
from cuda.core.experimental import Device
from cuda.core.experimental.utils import StridedMemoryView, args_viewable_as_strided_memory


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
        view = StridedMemoryView.from_any_interface(in_arr, stream_ptr=-1)
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

        view = StridedMemoryView.from_any_interface(in_arr, stream_ptr=s.handle if s else -1)
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
        view = StridedMemoryView.from_cuda_array_interface(in_arr, stream_ptr=s.handle if s else -1)
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
