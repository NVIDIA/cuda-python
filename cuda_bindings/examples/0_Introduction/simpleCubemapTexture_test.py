# Copyright 2021-2025 NVIDIA Corporation.  All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

import ctypes
import sys
import time

import numpy as np
from common import common
from common.helper_cuda import checkCudaErrors, findCudaDevice
from cuda.bindings import driver as cuda
from cuda.bindings import runtime as cudart

simpleCubemapTexture = """\
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
    devID = findCudaDevice()

    # Get number of SMs on this GPU
    deviceProps = checkCudaErrors(cudart.cudaGetDeviceProperties(devID))
    print(
        f"CUDA device [{deviceProps.name}] has {deviceProps.multiProcessorCount} Multi-Processors SM {deviceProps.major}.{deviceProps.minor}"
    )
    if deviceProps.major < 2:
        print("Test requires SM 2.0 or higher for support of Texture Arrays.  Test will exit...")
        sys.exit()

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
    d_data = checkCudaErrors(cudart.cudaMalloc(size))

    # Allocate array and copy image data
    channelDesc = checkCudaErrors(
        cudart.cudaCreateChannelDesc(32, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat)
    )
    cu_3darray = checkCudaErrors(
        cudart.cudaMalloc3DArray(
            channelDesc,
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
    checkCudaErrors(cudart.cudaMemcpy3D(myparms))

    texRes = cudart.cudaResourceDesc()
    texRes.resType = cudart.cudaResourceType.cudaResourceTypeArray
    texRes.res.array.array = cu_3darray

    texDescr = cudart.cudaTextureDesc()
    texDescr.normalizedCoords = True
    texDescr.filterMode = cudart.cudaTextureFilterMode.cudaFilterModeLinear
    texDescr.addressMode[0] = cudart.cudaTextureAddressMode.cudaAddressModeWrap
    texDescr.addressMode[1] = cudart.cudaTextureAddressMode.cudaAddressModeWrap
    texDescr.addressMode[2] = cudart.cudaTextureAddressMode.cudaAddressModeWrap
    texDescr.readMode = cudart.cudaTextureReadMode.cudaReadModeElementType

    tex = checkCudaErrors(cudart.cudaCreateTextureObject(texRes, texDescr, None))
    dimBlock = cudart.dim3()
    dimBlock.x = 8
    dimBlock.y = 8
    dimBlock.z = 1
    dimGrid = cudart.dim3()
    dimGrid.x = width / dimBlock.x
    dimGrid.y = width / dimBlock.y
    dimGrid.z = 1

    print(
        f"Covering Cubemap data array of {width}~3 x {num_layers}: Grid size is {dimGrid.x} x {dimGrid.y}, each block has 8 x 8 threads"
    )

    kernelHelper = common.KernelHelper(simpleCubemapTexture, devID)
    _transformKernel = kernelHelper.getFunction(b"transformKernel")
    kernelArgs = ((d_data, width, tex), (ctypes.c_void_p, ctypes.c_int, None))
    checkCudaErrors(
        cuda.cuLaunchKernel(
            _transformKernel,
            dimGrid.x,
            dimGrid.y,
            dimGrid.z,  # grid dim
            dimBlock.x,
            dimBlock.y,
            dimBlock.z,  # block dim
            0,
            0,  # shared mem and stream
            kernelArgs,
            0,
        )
    )  # arguments

    checkCudaErrors(cudart.cudaDeviceSynchronize())

    start = time.time()

    # Execute the kernel
    checkCudaErrors(
        cuda.cuLaunchKernel(
            _transformKernel,
            dimGrid.x,
            dimGrid.y,
            dimGrid.z,  # grid dim
            dimBlock.x,
            dimBlock.y,
            dimBlock.z,  # block dim
            0,
            0,  # shared mem and stream
            kernelArgs,
            0,
        )
    )  # arguments

    checkCudaErrors(cudart.cudaDeviceSynchronize())
    stop = time.time()
    print(f"Processing time: {stop - start:.3f} msec")
    print(f"{cubemap_size / ((stop - start + 1) / 1000.0) / 1e6:.2f} Mtexlookups/sec")

    # Allocate mem for the result on host side
    h_odata = np.empty_like(h_data)
    # Copy result from device to host
    checkCudaErrors(cudart.cudaMemcpy(h_odata, d_data, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

    checkCudaErrors(cudart.cudaDestroyTextureObject(tex))
    checkCudaErrors(cudart.cudaFree(d_data))
    checkCudaErrors(cudart.cudaFreeArray(cu_3darray))

    print("Comparing kernel output to expected data")
    MIN_EPSILON_ERROR = 5.0e-3
    if np.max(np.abs(h_odata - h_data_ref)) > MIN_EPSILON_ERROR:
        print("Failed")
        sys.exit(-1)
    print("Passed")


if __name__ == "__main__":
    main()
