# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import sys
import time

import numpy as np
from common import common
from common.helper_cuda import check_cuda_errors, find_cuda_device

from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

simple_cubemap_texture = """\
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
        //Layer 0 is positive X face
        if (face == 0)
        {
            cx = 1;
            cy = -v;
            cz = -u;
        }
        //Layer 1 is negative X face
        else if (face == 1)
        {
            cx = -1;
            cy = -v;
            cz = u;
        }
        //Layer 2 is positive Y face
        else if (face == 2)
        {
            cx = u;
            cy = 1;
            cz = v;
        }
        //Layer 3 is negative Y face
        else if (face == 3)
        {
            cx = u;
            cy = -1;
            cz = -v;
        }
        //Layer 4 is positive Z face
        else if (face == 4)
        {
            cx = u;
            cy = -v;
            cz = 1;
        }
        //Layer 4 is negative Z face
        else if (face == 5)
        {
            cx = -u;
            cy = -v;
            cz = -1;
        }

        // read from texture, do expected transformation and write to global memory
        g_odata[face*width*width + y*width + x] = -texCubemap<float>(tex, cx, cy, cz);
    }
}
"""


def main():
    # Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    dev_id = find_cuda_device()

    # Get number of SMs on this GPU
    device_props = check_cuda_errors(cudart.cudaGetDeviceProperties(dev_id))
    print(
        f"CUDA device [{device_props.name}] has {device_props.multiProcessorCount} Multi-Processors SM {device_props.major}.{device_props.minor}"
    )
    if device_props.major < 2:
        import pytest

        pytest.skip("Test requires SM 2.0 or higher for support of Texture Arrays.")

    # Generate input data for layered texture
    width = 64
    num_faces = 6
    num_layers = 1
    cubemap_size = width * width * num_faces
    h_data = np.arange(cubemap_size * num_layers, dtype="float32")
    size = h_data.nbytes

    # This is the expected transformation of the input data (the expected output)
    h_data_ref = np.repeat(np.arange(num_layers, dtype=h_data.dtype), cubemap_size) - h_data

    # Allocate device memory for result
    d_data = check_cuda_errors(cudart.cudaMalloc(size))

    # Allocate array and copy image data
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
    myparms = cudart.cudaMemcpy3DParms()
    myparms.srcPos = cudart.make_cudaPos(0, 0, 0)
    myparms.dstPos = cudart.make_cudaPos(0, 0, 0)
    myparms.srcPtr = cudart.make_cudaPitchedPtr(h_data, width_nbytes, width, width)
    myparms.dstArray = cu_3darray
    myparms.extent = cudart.make_cudaExtent(width, width, num_faces)
    myparms.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    check_cuda_errors(cudart.cudaMemcpy3D(myparms))

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
    dim_block = cudart.dim3()
    dim_block.x = 8
    dim_block.y = 8
    dim_block.z = 1
    dim_grid = cudart.dim3()
    dim_grid.x = width / dim_block.x
    dim_grid.y = width / dim_block.y
    dim_grid.z = 1

    print(
        f"Covering Cubemap data array of {width}~3 x {num_layers}: Grid size is {dim_grid.x} x {dim_grid.y}, each block has 8 x 8 threads"
    )

    kernel_helper = common.KernelHelper(simple_cubemap_texture, dev_id)
    _transform_kernel = kernel_helper.get_function(b"transformKernel")
    kernel_args = ((d_data, width, tex), (ctypes.c_void_p, ctypes.c_int, None))
    check_cuda_errors(
        cuda.cuLaunchKernel(
            _transform_kernel,
            dim_grid.x,
            dim_grid.y,
            dim_grid.z,  # grid dim
            dim_block.x,
            dim_block.y,
            dim_block.z,  # block dim
            0,
            0,  # shared mem and stream
            kernel_args,
            0,
        )
    )  # arguments

    check_cuda_errors(cudart.cudaDeviceSynchronize())

    start = time.time()

    # Execute the kernel
    check_cuda_errors(
        cuda.cuLaunchKernel(
            _transform_kernel,
            dim_grid.x,
            dim_grid.y,
            dim_grid.z,  # grid dim
            dim_block.x,
            dim_block.y,
            dim_block.z,  # block dim
            0,
            0,  # shared mem and stream
            kernel_args,
            0,
        )
    )  # arguments

    check_cuda_errors(cudart.cudaDeviceSynchronize())
    stop = time.time()
    print(f"Processing time: {stop - start:.3f} msec")
    print(f"{cubemap_size / ((stop - start + 1) / 1000.0) / 1e6:.2f} Mtexlookups/sec")

    # Allocate mem for the result on host side
    h_odata = np.empty_like(h_data)
    # Copy result from device to host
    check_cuda_errors(cudart.cudaMemcpy(h_odata, d_data, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

    check_cuda_errors(cudart.cudaDestroyTextureObject(tex))
    check_cuda_errors(cudart.cudaFree(d_data))
    check_cuda_errors(cudart.cudaFreeArray(cu_3darray))

    min_epsilon_error = 5.0e-3
    if np.max(np.abs(h_odata - h_data_ref)) > min_epsilon_error:
        print("Failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
