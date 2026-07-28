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
# dependencies = ["cuda-python>=13.0.0", "cuda-core>=1.0.0", "cupy-cuda13x>=14.0.0"]
# ///

"""
Vector Addition using CUDA Core API

This sample demonstrates element-wise vector addition: C = A + B
using cuda.core for runtime compilation and kernel launch.

The kernel is templated on the element type ``T``. The sample compiles
two instantiations of the template in a single ``Program.compile()`` call
via the ``name_expressions`` argument, then launches:

  * ``vectorAdd<float>``  against CuPy-allocated buffers
  * ``vectorAdd<double>`` against a buffer we allocate ourselves through
    ``Device.allocate()`` (wrapped as a CuPy view for verification)

The second phase illustrates how to hand a raw ``Buffer`` to a kernel and
zero-copy view it as a CuPy array to check the result.
"""

import sys
from pathlib import Path

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "Utilities"))
from cuda_samples_utils import verify_array_result

try:
    import cupy as cp

    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


# CUDA kernel source code
VECTOR_ADD_KERNEL = """
/**
 * CUDA Kernel for vector addition
 * Computes the vector addition of A and B into C.
 */
template<typename T>
__global__ void vectorAdd(const T *A, const T *B, T *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
"""


def _launch_add(stream, kernel, size, a_ptr, b_ptr, c_ptr, threads_per_block=256):
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
    print(f"  CUDA kernel launch with {blocks_per_grid} blocks of {threads_per_block} threads")
    config = LaunchConfig(grid=blocks_per_grid, block=threads_per_block)
    launch(stream, config, kernel, a_ptr, b_ptr, c_ptr, cp.int32(size))
    stream.sync()


def _demo_float_cupy(stream, kernel, num_elements, verify):
    """Phase 1: `vectorAdd<float>` with CuPy-managed buffers."""
    print(f"\n[1] vectorAdd<float> on {num_elements} CuPy-allocated elements")
    dtype = cp.float32

    a = cp.random.rand(num_elements).astype(dtype)
    b = cp.random.rand(num_elements).astype(dtype)
    c = cp.empty(num_elements, dtype=dtype)
    _launch_add(stream, kernel, num_elements, a.data.ptr, b.data.ptr, c.data.ptr)

    if verify:
        print("  Verifying result...")
        if not verify_array_result(c, a + b):
            return False
    return True


def _demo_double_owned_buffer(device, stream, kernel, num_elements, verify):
    """Phase 2: `vectorAdd<double>` with an output Buffer we own.

    Instead of letting CuPy allocate the output, we call ``device.allocate()``
    to get a raw ``Buffer``, pass it straight to the kernel, and then wrap it
    as a CuPy array (through ``UnownedMemory``) purely for verification.
    """
    print(f"\n[2] vectorAdd<double> on {num_elements} elements with device.allocate() output")
    dtype = cp.float64

    a = cp.random.rand(num_elements).astype(dtype)
    b = cp.random.rand(num_elements).astype(dtype)

    out_bytes = num_elements * dtype().itemsize
    out_buf = device.allocate(out_bytes, stream=stream)
    device.sync()

    try:
        _launch_add(stream, kernel, num_elements, a.data.ptr, b.data.ptr, out_buf)

        if verify:
            print("  Verifying result...")
            # Wrap the raw Buffer as a CuPy array without copying so we can
            # compare against the reference computed with CuPy.
            c_view = cp.ndarray(
                num_elements,
                dtype=dtype,
                memptr=cp.cuda.MemoryPointer(
                    cp.cuda.UnownedMemory(int(out_buf.handle), out_buf.size, out_buf),
                    0,
                ),
            )
            if not verify_array_result(c_view, a + b):
                return False
        return True
    finally:
        out_buf.close(stream)


def vector_add_cuda_core(num_elements=50000, device_id=0, verify=True):
    """
    Perform vector addition using cuda.core API.

    Parameters
    ----------
    num_elements : int
        Number of elements in each vector
    device_id : int
        CUDA device ID to use
    verify : bool
        Whether to verify the result

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    stream = None
    try:
        # Initialize device
        print("[Vector addition using CUDA Core API]")
        device = Device(device_id)
        device.set_current()

        print(f"Device: {device.name}")
        print(f"Compute Capability: sm_{device.arch}")

        stream = device.create_stream()

        # Compile both template instantiations in a single call: name_expressions
        # tells NVRTC which specializations to emit into the resulting cubin.
        print("Compiling kernel 'vectorAdd<float>' and 'vectorAdd<double>'...")
        program_options = ProgramOptions(std="c++17", arch=f"sm_{device.arch}")
        program = Program(VECTOR_ADD_KERNEL, code_type="c++", options=program_options)
        module = program.compile(
            "cubin",
            name_expressions=("vectorAdd<float>", "vectorAdd<double>"),
        )
        float_kernel = module.get_kernel("vectorAdd<float>")
        double_kernel = module.get_kernel("vectorAdd<double>")
        print("Kernels compiled successfully")

        if not _demo_float_cupy(stream, float_kernel, num_elements, verify):
            return False

        # Half as many elements for the double demo to keep the sample fast.
        if not _demo_double_owned_buffer(device, stream, double_kernel, max(num_elements // 2, 1), verify):
            return False

        print("\nTest PASSED")
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if stream is not None:
            stream.close()


def main():
    """
    Main entry point for the vector addition sample.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Vector Addition using CUDA Core API")
    parser.add_argument(
        "--elements",
        type=int,
        default=50000,
        help="Number of elements in vectors (default: 50000)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID (default: 0)")
    parser.add_argument("--no-verify", action="store_true", help="Skip result verification")

    args = parser.parse_args()

    if args.elements <= 0:
        print("Error: Number of elements must be positive")
        return 1

    success = vector_add_cuda_core(num_elements=args.elements, device_id=args.device, verify=not args.no_verify)

    if success:
        print("\nDone")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
