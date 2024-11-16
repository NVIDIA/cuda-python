import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
import pytest

from cuda.core.experimental import Device
from cuda.core.experimental.utils import StridedMemoryView, viewable


@pytest.mark.parametrize(
    "in_arr,", (
        np.empty(3, dtype=np.int32),
        np.empty((6, 6), dtype=np.float64)[::2, ::2],
        np.empty((3, 4), order='F'),
    )
)
def test_viewable_cpu(in_arr):

    @viewable((0,))
    def my_func(arr):
        view = arr.view(-1)
        assert view.ptr == in_arr.ctypes.data
        assert view.shape == in_arr.shape
        if in_arr.flags.c_contiguous:
            assert view.strides is None
        else:
            assert view.strides == tuple(s // in_arr.dtype.itemsize for s in in_arr.strides)
        assert view.dtype == in_arr.dtype
        assert view.device_id == 0
        assert view.device_accessible == False
        assert view.exporting_obj is in_arr

    my_func(in_arr)


if cp is not None:

    @pytest.mark.parametrize(
        "in_arr,stream", (
            (cp.empty(3, dtype=cp.complex64), None),
            (cp.empty((6, 6), dtype=cp.float64)[::2, ::2], True),
            (cp.empty((3, 4), order='F'), True),
        )
    )
    def test_viewable_gpu(in_arr, stream):
        # TODO: use the device fixture?
        dev = Device()
        dev.set_current()
        s = dev.create_stream() if stream else None

        @viewable((0,))
        def my_func(arr):
            view = arr.view(s.handle if s else -1)
            assert view.ptr == in_arr.data.ptr
            assert view.shape == in_arr.shape
            strides_in_counts = tuple(s // in_arr.dtype.itemsize for s in in_arr.strides)
            if in_arr.flags.c_contiguous:
                assert view.strides in (None, strides_in_counts)
            else:
                assert view.strides == strides_in_counts
            assert view.dtype == in_arr.dtype
            assert view.device_id == dev.device_id
            assert view.device_accessible == True
            assert view.exporting_obj is in_arr
    
        my_func(in_arr)
