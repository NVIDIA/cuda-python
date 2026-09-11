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
# dependencies = ["cuda-python>=13.4.0", "numpy>=1.24"]
# ///

"""
Raw NVRTC compilation + driver-API module loading

This is the under-the-hood companion to the high-level
[`samples/cuda_core/jitLtoLinking/`](../../../cuda_core/jitLtoLinking/) sample. Where
``jitLtoLinking`` uses ``cuda.core.Program`` and ``cuda.core.Linker`` to
compile and link device code, this sample walks through the raw calls that
those higher-level classes wrap:

  * ``nvrtc.create_program`` -> ``nvrtc.compile_program`` -> retrieve the log
  * ``nvrtc.get_cubin`` (or ``nvrtc.get_ptx`` on older NVRTC) -> device bytes
  * ``cuModuleLoadData`` -> load the module
  * ``cuModuleGetFunction`` -> get a ``CUfunction`` for a named symbol
  * ``cuLaunchKernel`` -> launch it, then ``cuModuleUnload``

Read this alongside ``jitLtoLinking`` when you want to see what the
``cuda.core`` compile/link pipeline is doing internally.
"""

import ctypes
import sys

try:
    import numpy as np

    from cuda.bindings import driver as cuda
    from cuda.bindings._v2 import nvrtc
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


SAXPY_KERNEL = """\
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a * x[tid] + y[tid];
    }
}
"""


def _assert_drv(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def main():
    # ---- 1) Initialize the driver and get a device + context ----
    (err,) = cuda.cuInit(0)
    _assert_drv(err)

    err, cu_device = cuda.cuDeviceGet(0)
    _assert_drv(err)

    err, context = cuda.cuCtxCreate(None, 0, cu_device)
    _assert_drv(err)

    # ---- 2) Create an NVRTC program from the SAXPY source ----
    prog = nvrtc.create_program(str.encode(SAXPY_KERNEL), b"saxpy.cu")

    # ---- 3) Pick a target architecture and choose CUBIN vs PTX ----
    err, major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device
    )
    _assert_drv(err)
    err, minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device
    )
    _assert_drv(err)
    _nvrtc_major, nvrtc_minor = nvrtc.version()
    use_cubin = nvrtc_minor >= 1
    prefix = "sm" if use_cubin else "compute"
    arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

    # ---- 4) Compile and print the log (may be empty on success) ----
    opts = [b"--fmad=false", arch_arg]
    nvrtc.compile_program(prog, opts)

    log = nvrtc.get_program_log(prog)
    print(log.decode())

    # ---- 5) Retrieve either CUBIN or PTX bytes ----
    if use_cubin:
        data = nvrtc.get_cubin(prog)
    else:
        data = nvrtc.get_ptx(prog)

    # ---- 6) Load the module and get the kernel entry point ----
    data = np.char.array(data)
    err, module = cuda.cuModuleLoadData(data)
    _assert_drv(err)
    err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
    _assert_drv(err)

    # ---- 7) Launch the kernel and verify ----
    num_threads = 128
    num_blocks = 32
    a = np.float32(2.0)
    n = np.array(num_threads * num_blocks, dtype=np.uint32)
    buffer_size = n * a.itemsize

    err, d_x = cuda.cuMemAlloc(buffer_size)
    _assert_drv(err)
    err, d_y = cuda.cuMemAlloc(buffer_size)
    _assert_drv(err)
    err, d_out = cuda.cuMemAlloc(buffer_size)
    _assert_drv(err)

    h_x = np.random.rand(n).astype(dtype=np.float32)
    h_y = np.random.rand(n).astype(dtype=np.float32)
    h_out = np.zeros(n).astype(dtype=np.float32)

    err, stream = cuda.cuStreamCreate(0)
    _assert_drv(err)

    (err,) = cuda.cuMemcpyHtoDAsync(d_x, h_x, buffer_size, stream)
    _assert_drv(err)
    (err,) = cuda.cuMemcpyHtoDAsync(d_y, h_y, buffer_size, stream)
    _assert_drv(err)
    (err,) = cuda.cuStreamSynchronize(stream)
    _assert_drv(err)

    # Sanity: host output is still zeros before the kernel runs.
    h_z = a * h_x + h_y
    if np.allclose(h_out, h_z):
        raise ValueError("Error inside tolerance for host-device vectors")

    arg_values = (a, d_x, d_y, d_out, n)
    arg_types = (ctypes.c_float, None, None, None, ctypes.c_size_t)
    (err,) = cuda.cuLaunchKernel(
        kernel,
        num_blocks,
        1,
        1,  # grid dim
        num_threads,
        1,
        1,  # block dim
        0,
        stream,  # sharedMemBytes, stream
        (arg_values, arg_types),
        0,
    )
    _assert_drv(err)

    (err,) = cuda.cuMemcpyDtoHAsync(h_out, d_out, buffer_size, stream)
    _assert_drv(err)
    (err,) = cuda.cuStreamSynchronize(stream)
    _assert_drv(err)

    h_z = a * h_x + h_y
    if not np.allclose(h_out, h_z):
        raise ValueError("Error outside tolerance for host-device vectors")

    # ---- 8) Tear down ----
    (err,) = cuda.cuStreamDestroy(stream)
    _assert_drv(err)
    (err,) = cuda.cuMemFree(d_x)
    _assert_drv(err)
    (err,) = cuda.cuMemFree(d_y)
    _assert_drv(err)
    (err,) = cuda.cuMemFree(d_out)
    _assert_drv(err)
    (err,) = cuda.cuModuleUnload(module)
    _assert_drv(err)
    (err,) = cuda.cuCtxDestroy(context)
    _assert_drv(err)

    print("SAXPY through raw NVRTC + driver API verified.")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
