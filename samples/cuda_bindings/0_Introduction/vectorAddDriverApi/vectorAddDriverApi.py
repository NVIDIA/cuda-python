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
# dependencies = ["cuda-python>=13.0.0", "numpy>=1.24"]
# ///

"""
Vector addition with the raw CUDA Driver API

The canonical "hello world" for ``cuda.bindings.driver``. This is the same
element-wise ``C = A + B`` computation shown in the
``samples/cuda_core/vectorAdd/`` sample, but implemented at the driver-API layer:

  * ``cuInit`` -> ``cuDeviceGet`` -> ``cuCtxCreate`` (explicit context)
  * ``cuMemAlloc`` / ``cuMemcpyHtoD`` / ``cuMemcpyDtoH`` / ``cuMemFree``
  * ``cuLaunchKernel`` with an explicit ``sharedMemBytes`` and ``stream``

The kernel is compiled at runtime through NVRTC and loaded as a
``CUmodule``. The sample also verifies that the selected device advertises
Unified Virtual Addressing (UVA) -- the driver-API property that lets host
pointers and device pointers share the same address space, which the newer
runtime APIs assume.

Waives with exit code 2 when the device does not support UVA.
"""

import ctypes
import math
import sys
from pathlib import Path

# Add samples/cuda_bindings/Utilities/ to the import path for shared bindings helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Utilities"))

try:
    import numpy as np
    from cuda_bindings_utils import KernelHelper, check_cuda_errors, find_cuda_device_drv, requirement_not_met

    from cuda.bindings import driver as cuda
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


VECTOR_ADD_DRV_KERNEL = """\
/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 */

extern "C" __global__ void VecAdd_kernel(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}
"""


def main():
    n = 50000
    nbytes = n * np.dtype(np.float32).itemsize

    # ---- 1) Initialize the driver + create an explicit context ----
    check_cuda_errors(cuda.cuInit(0))
    cu_device = find_cuda_device_drv()
    cu_context = check_cuda_errors(cuda.cuCtxCreate(None, 0, cu_device))

    # UVA gives host + device pointers a single virtual address space -- most
    # modern driver-API patterns implicitly assume this.
    uva_supported = check_cuda_errors(
        cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cu_device)
    )
    if not uva_supported:
        requirement_not_met("This sample requires Unified Virtual Addressing (UVA) support")

    # ---- 2) Compile and load the kernel ----
    kernel_helper = KernelHelper(VECTOR_ADD_DRV_KERNEL, int(cu_device))
    vec_add_kernel = kernel_helper.get_function(b"VecAdd_kernel")

    # ---- 3) Allocate host + device buffers, copy H2D ----
    h_a = np.random.rand(n).astype(dtype=np.float32)
    h_b = np.random.rand(n).astype(dtype=np.float32)
    h_c = np.zeros(n, dtype=np.float32)

    d_a = check_cuda_errors(cuda.cuMemAlloc(nbytes))
    d_b = check_cuda_errors(cuda.cuMemAlloc(nbytes))
    d_c = check_cuda_errors(cuda.cuMemAlloc(nbytes))

    check_cuda_errors(cuda.cuMemcpyHtoD(d_a, h_a, nbytes))
    check_cuda_errors(cuda.cuMemcpyHtoD(d_b, h_b, nbytes))

    # ---- 4) Launch ----
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    kernel_args = ((d_a, d_b, d_c, n), (None, None, None, ctypes.c_int))
    check_cuda_errors(
        cuda.cuLaunchKernel(
            vec_add_kernel,
            blocks_per_grid,
            1,
            1,  # grid dim
            threads_per_block,
            1,
            1,  # block dim
            0,  # sharedMemBytes
            0,  # stream
            kernel_args,
            0,
        )
    )

    # ---- 5) Copy D2H and verify ----
    check_cuda_errors(cuda.cuMemcpyDtoH(h_c, d_c, nbytes))

    max_err = 0.0
    for i in range(n):
        max_err = max(max_err, math.fabs(h_c[i] - (h_a[i] + h_b[i])))

    # ---- 6) Free device memory and destroy context ----
    check_cuda_errors(cuda.cuMemFree(d_a))
    check_cuda_errors(cuda.cuMemFree(d_b))
    check_cuda_errors(cuda.cuMemFree(d_c))
    check_cuda_errors(cuda.cuCtxDestroy(cu_context))

    if max_err > 1e-5:
        print(f"Result = FAIL (max error {max_err})", file=sys.stderr)
        return 1

    print(f"Result = PASS (max error {max_err:.3e} over {n} elements)")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
