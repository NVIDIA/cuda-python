# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# /// script
# dependencies = ["cuda-python>=13.0.0", "cuda-core>=1.0.0", "cupy-cuda13x>=14.0.0", "numpy>=2.1"]
# ///

"""
StridedMemoryView constructors with cuda.core

``cuda.core.utils.StridedMemoryView`` is the type a library reaches for when
it wants to accept "any array-like object" and describe its memory layout
without dictating whether the caller uses NumPy, CuPy, PyTorch, DLPack, or a
raw ``Buffer``.

This sample exercises the four explicit ``from_*`` constructors:

  * ``from_array_interface(host_array)``          -- NumPy's ``__array_interface__``
  * ``from_dlpack(any_array, stream_ptr=...)``    -- the DLPack protocol
                                                     (works for host and device arrays)
  * ``from_cuda_array_interface(gpu_array, ...)`` -- ``__cuda_array_interface__``
  * ``from_buffer(buf, shape=..., strides=..., dtype=...)`` -- raw ``cuda.core.Buffer``

For each, the sample constructs a view, prints its shape / dtype /
device-accessibility, and reads the underlying data back through DLPack to
verify the round-trip.
"""

import sys

try:
    import cupy as cp
    import numpy as np

    from cuda.core import Device
    from cuda.core.utils import StridedMemoryView
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def dense_c_strides(shape):
    """Compute contiguous C strides for ``shape`` (in elements, not bytes)."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return tuple(strides)


def _describe(name, view):
    kind = "device-accessible" if view.is_device_accessible else "host-only"
    print(f"  {name}: shape={view.shape}, dtype={view.dtype}, size={view.size} ({kind})")


def main():
    if np.lib.NumpyVersion(np.__version__) < "2.1.0":
        print("This example requires NumPy 2.1.0 or later", file=sys.stderr)
        sys.exit(2)

    device = Device()
    device.set_current()
    stream = device.create_stream()
    buffer = None

    try:
        # ---- 1) __array_interface__ (host NumPy array) ----
        print("[1] from_array_interface(host_numpy_array)")
        host_array = np.arange(12, dtype=np.int16).reshape(3, 4)
        host_view = StridedMemoryView.from_array_interface(host_array)
        _describe("host_view", host_view)
        assert host_view.shape == host_array.shape
        assert host_view.size == host_array.size
        assert not host_view.is_device_accessible
        assert np.array_equal(np.from_dlpack(host_view), host_array)

        # ---- 2) DLPack (host_array) ----
        print("\n[2] from_dlpack(host_array, stream_ptr=-1)")
        host_dlpack_view = StridedMemoryView.from_dlpack(host_array, stream_ptr=-1)
        _describe("host_dlpack_view", host_dlpack_view)
        assert np.array_equal(np.from_dlpack(host_dlpack_view), host_array)

        # ---- 3) DLPack (GPU) and __cuda_array_interface__ (GPU) ----
        print("\n[3] from_dlpack(gpu_array) and from_cuda_array_interface(gpu_array)")
        gpu_array = cp.arange(12, dtype=cp.float32).reshape(3, 4)
        dlpack_view = StridedMemoryView.from_dlpack(gpu_array, stream_ptr=stream.handle)
        cai_view = StridedMemoryView.from_cuda_array_interface(gpu_array, stream_ptr=stream.handle)
        _describe("dlpack_view (gpu)", dlpack_view)
        _describe("cai_view (gpu)", cai_view)
        cp.testing.assert_array_equal(cp.from_dlpack(dlpack_view), gpu_array)
        cp.testing.assert_array_equal(cp.from_dlpack(cai_view), gpu_array)

        # ---- 4) from_buffer (raw cuda.core.Buffer) ----
        print("\n[4] from_buffer(buf, shape=..., strides=..., dtype=...)")
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
        _describe("buffer_view", buffer_view)
        cp.testing.assert_array_equal(cp.from_dlpack(buffer_view), gpu_array)

        print("\nConstructed StridedMemoryView from array_interface, DLPack, CAI, and Buffer inputs.")
        print("Done")
        return 0
    finally:
        if buffer is not None:
            buffer.close(stream)
        stream.close()


if __name__ == "__main__":
    sys.exit(main())
