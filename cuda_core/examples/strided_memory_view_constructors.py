# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# ################################################################################
#
# This example demonstrates the explicit StridedMemoryView constructors for
# __array_interface__, DLPack, __cuda_array_interface__, and Buffer objects.
#
# ################################################################################

# /// script
# dependencies = ["cuda_bindings", "cuda_core", "cupy-cuda13x", "numpy>=2.1"]
# ///

import sys

import cupy as cp
import numpy as np

from cuda.core import Device
from cuda.core.utils import StridedMemoryView


def dense_c_strides(shape):
    if not shape:
        return ()

    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return tuple(strides)


def main():
    if np.lib.NumpyVersion(np.__version__) < "2.1.0":
        print("This example requires NumPy 2.1.0 or later", file=sys.stderr)
        sys.exit(1)

    device = Device()
    device.set_current()
    stream = device.create_stream()
    buffer = None

    try:
        host_array = np.arange(12, dtype=np.int16).reshape(3, 4)
        host_view = StridedMemoryView.from_array_interface(host_array)
        host_dlpack_view = StridedMemoryView.from_dlpack(host_array, stream_ptr=-1)

        assert host_view.shape == host_array.shape
        assert host_view.size == host_array.size
        assert not host_view.is_device_accessible
        assert np.array_equal(np.from_dlpack(host_view), host_array)
        assert np.array_equal(np.from_dlpack(host_dlpack_view), host_array)

        gpu_array = cp.arange(12, dtype=cp.float32).reshape(3, 4)
        dlpack_view = StridedMemoryView.from_dlpack(gpu_array, stream_ptr=stream.handle)
        cai_view = StridedMemoryView.from_cuda_array_interface(gpu_array, stream_ptr=stream.handle)

        cp.testing.assert_array_equal(cp.from_dlpack(dlpack_view), gpu_array)
        cp.testing.assert_array_equal(cp.from_dlpack(cai_view), gpu_array)

        buffer = device.memory_resource.allocate(gpu_array.nbytes, stream=stream)
        buffer_array = cp.from_dlpack(buffer).view(dtype=cp.float32).reshape(gpu_array.shape)
        buffer_array[...] = gpu_array
        device.sync()

        buffer_view = StridedMemoryView.from_buffer(
            buffer,
            shape=gpu_array.shape,
            strides=dense_c_strides(gpu_array.shape),
            dtype=np.dtype(np.float32),
        )
        cp.testing.assert_array_equal(cp.from_dlpack(buffer_view), gpu_array)

        print("Constructed StridedMemoryView objects from array, DLPack, CAI, and Buffer inputs.")
    finally:
        if buffer is not None:
            buffer.close(stream)
        stream.close()


if __name__ == "__main__":
    main()
