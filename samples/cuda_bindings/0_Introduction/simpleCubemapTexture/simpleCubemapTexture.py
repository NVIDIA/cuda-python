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
Cubemap textures and 3D memcpy with cuda.bindings

Creates a cubemap ``cudaArray`` (six 2D faces), wraps it as a bindless
``cudaTextureObject_t``, and samples it from a kernel via
``texCubemap<float>(tex, cx, cy, cz)``.

Also demonstrates 3D memory transfers with ``cudaMemcpy3DParms`` and
``cudaMemcpy3D``, and the CUDA texture descriptor knobs
(``cudaTextureDesc``: normalized coords, linear filter, wrap addressing).

This is currently the only sample in ``/samples/cuda_bindings`` that teaches CUDA
texture objects, cubemap arrays, or ``cudaMemcpy3D``.
"""

import ctypes
import sys
import time
from pathlib import Path

# Add samples/cuda_bindings/Utilities/ to the import path for shared bindings helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Utilities"))

try:
    import numpy as np
    from cuda_bindings_utils import KernelHelper, check_cuda_errors, find_cuda_device, requirement_not_met

    from cuda.bindings import driver as cuda
    from cuda.bindings import runtime as cudart
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install from requirements.txt:")
    print("  pip install -r requirements.txt")
    sys.exit(1)


CUBEMAP_KERNEL = """\
extern "C"
__global__ void transformKernel(float *g_odata, int width, cudaTextureObject_t tex)
{
    // calculate this thread's data point
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // 0.5f offset and division are necessary to access the original data points
    // in the texture (such that bilinear interpolation will not be activated).
    // For details, see also CUDA Programming Guide, Appendix D
    float u = ((x+0.5f) / (float) width) * 2.f - 1.f;
    float v = ((y+0.5f) / (float) width) * 2.f - 1.f;

    float cx, cy, cz;
    for (unsigned int face = 0; face < 6; face ++)
    {
        // Direction vectors per cubemap face
        if (face == 0)      { cx =  1; cy = -v; cz = -u; }  // +X
        else if (face == 1) { cx = -1; cy = -v; cz =  u; }  // -X
        else if (face == 2) { cx =  u; cy =  1; cz =  v; }  // +Y
        else if (face == 3) { cx =  u; cy = -1; cz = -v; }  // -Y
        else if (face == 4) { cx =  u; cy = -v; cz =  1; }  // +Z
        else if (face == 5) { cx = -u; cy = -v; cz = -1; }  // -Z

        // Sample the cubemap face, negate, and write to global memory.
        g_odata[face*width*width + y*width + x] = -texCubemap<float>(tex, cx, cy, cz);
    }
}
"""


def main():
    dev_id = find_cuda_device()

    device_props = check_cuda_errors(cudart.cudaGetDeviceProperties(dev_id))
    print(
        f"CUDA device [{device_props.name}] has {device_props.multiProcessorCount} "
        f"Multi-Processors SM {device_props.major}.{device_props.minor}"
    )
    if device_props.major < 2:
        requirement_not_met("Requires SM 2.0 or higher for texture array support")

    # ---- Generate input data for the cubemap texture ----
    width = 64
    num_faces = 6
    num_layers = 1
    cubemap_size = width * width * num_faces
    h_data = np.arange(cubemap_size * num_layers, dtype="float32")
    size = h_data.nbytes

    # Expected: kernel negates the sampled values, so output = layer_index - h_data.
    h_data_ref = np.repeat(np.arange(num_layers, dtype=h_data.dtype), cubemap_size) - h_data

    # ---- Allocate device output ----
    d_data = check_cuda_errors(cudart.cudaMalloc(size))

    # ---- Allocate a cubemap cudaArray and copy the source into it via cudaMemcpy3D ----
    channel_desc = check_cuda_errors(
        cudart.cudaCreateChannelDesc(32, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat)
    )
    cu_3darray = check_cuda_errors(
        cudart.cudaMalloc3DArray(
            channel_desc,
            cudart.make_cudaExtent(width, width, num_faces),
            cudart.cudaArrayCubemap,
        )
    )
    width_nbytes = h_data[:width].nbytes
    memcpy_params = cudart.cudaMemcpy3DParms()
    memcpy_params.srcPos = cudart.make_cudaPos(0, 0, 0)
    memcpy_params.dstPos = cudart.make_cudaPos(0, 0, 0)
    memcpy_params.srcPtr = cudart.make_cudaPitchedPtr(h_data, width_nbytes, width, width)
    memcpy_params.dstArray = cu_3darray
    memcpy_params.extent = cudart.make_cudaExtent(width, width, num_faces)
    memcpy_params.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    check_cuda_errors(cudart.cudaMemcpy3D(memcpy_params))

    # ---- Build the texture object over the cudaArray ----
    tex_res = cudart.cudaResourceDesc()
    tex_res.resType = cudart.cudaResourceType.cudaResourceTypeArray
    tex_res.res.array.array = cu_3darray

    tex_descr = cudart.cudaTextureDesc()
    tex_descr.normalizedCoords = True
    tex_descr.filterMode = cudart.cudaTextureFilterMode.cudaFilterModeLinear
    tex_descr.addressMode[0] = cudart.cudaTextureAddressMode.cudaAddressModeWrap
    tex_descr.addressMode[1] = cudart.cudaTextureAddressMode.cudaAddressModeWrap
    tex_descr.addressMode[2] = cudart.cudaTextureAddressMode.cudaAddressModeWrap
    tex_descr.readMode = cudart.cudaTextureReadMode.cudaReadModeElementType

    tex = check_cuda_errors(cudart.cudaCreateTextureObject(tex_res, tex_descr, None))

    # ---- Launch config: one thread per texel per face ----
    dim_block = cudart.dim3()
    dim_block.x = 8
    dim_block.y = 8
    dim_block.z = 1
    dim_grid = cudart.dim3()
    dim_grid.x = width // dim_block.x
    dim_grid.y = width // dim_block.y
    dim_grid.z = 1

    print(
        f"Covering Cubemap data array of {width}^3 x {num_layers}: "
        f"Grid size is {dim_grid.x} x {dim_grid.y}, each block has 8 x 8 threads"
    )

    kernel_helper = KernelHelper(CUBEMAP_KERNEL, dev_id)
    transform_kernel = kernel_helper.get_function(b"transformKernel")
    kernel_args = ((d_data, width, tex), (ctypes.c_void_p, ctypes.c_int, None))

    # Warm-up launch.
    check_cuda_errors(
        cuda.cuLaunchKernel(
            transform_kernel,
            dim_grid.x,
            dim_grid.y,
            dim_grid.z,
            dim_block.x,
            dim_block.y,
            dim_block.z,
            0,
            0,
            kernel_args,
            0,
        )
    )
    check_cuda_errors(cudart.cudaDeviceSynchronize())

    # Timed launch.
    start = time.time()
    check_cuda_errors(
        cuda.cuLaunchKernel(
            transform_kernel,
            dim_grid.x,
            dim_grid.y,
            dim_grid.z,
            dim_block.x,
            dim_block.y,
            dim_block.z,
            0,
            0,
            kernel_args,
            0,
        )
    )
    check_cuda_errors(cudart.cudaDeviceSynchronize())
    stop = time.time()
    print(f"Processing time: {(stop - start) * 1000:.3f} msec")
    if stop > start:
        print(f"{cubemap_size / (stop - start) / 1e6:.2f} Mtexlookups/sec")

    # ---- Copy result back and verify ----
    h_odata = np.empty_like(h_data)
    check_cuda_errors(cudart.cudaMemcpy(h_odata, d_data, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

    check_cuda_errors(cudart.cudaDestroyTextureObject(tex))
    check_cuda_errors(cudart.cudaFree(d_data))
    check_cuda_errors(cudart.cudaFreeArray(cu_3darray))

    min_epsilon_error = 5.0e-3
    max_err = float(np.max(np.abs(h_odata - h_data_ref)))
    if max_err > min_epsilon_error:
        print(f"Verification FAILED (max error {max_err})", file=sys.stderr)
        return 1

    print(f"Verification PASSED (max error {max_err:.3e})")
    print("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
