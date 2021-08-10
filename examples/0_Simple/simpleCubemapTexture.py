# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import ctypes
import math
import numpy as np
import sys
import time
from cuda import cuda, cudart
from examples.common import common
from examples.common.helper_cuda import checkCudaErrors, findCudaDevice

simpleCubemapTexture = '''\
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
'''

def main():
    # Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    devID = findCudaDevice()

    # Get number of SMs on this GPU
    deviceProps = checkCudaErrors(cudart.cudaGetDeviceProperties(devID));
    print("CUDA device [{}] has {} Multi-Processors SM {}.{}".format(deviceProps.name,
                                                                     deviceProps.multiProcessorCount,
                                                                     deviceProps.major,
                                                                     deviceProps.minor))
    if (deviceProps.major < 2):
        print("{} requires SM 2.0 or higher for support of Texture Arrays.  Test will exit...".format(sSDKname))
        sys.exit()

    # Generate input data for layered texture
    width = 64
    num_faces = 6
    num_layers = 1
    cubemap_size = width * width * num_faces
    size = cubemap_size * num_layers * np.dtype(np.float32).itemsize
    h_data = np.zeros(cubemap_size * num_layers, dtype='float32')

    for i in range(cubemap_size * num_layers):
        h_data[i] = i

    # This is the expected transformation of the input data (the expected output)
    h_data_ref = np.zeros(cubemap_size * num_layers, dtype='float32')

    for layer in range(num_layers):
        for i in range(cubemap_size):
            h_data_ref[layer*cubemap_size + i] = -h_data[layer*cubemap_size + i] + layer

    # Allocate device memory for result
    d_data = checkCudaErrors(cudart.cudaMalloc(size))

    # Allocate array and copy image data
    channelDesc = checkCudaErrors(cudart.cudaCreateChannelDesc(32, 0, 0, 0, cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat))
    cu_3darray = checkCudaErrors(cudart.cudaMalloc3DArray(channelDesc, cudart.make_cudaExtent(width, width, num_faces), cudart.cudaArrayCubemap))
    myparms = cudart.cudaMemcpy3DParms()
    myparms.srcPos = cudart.make_cudaPos(0,0,0)
    myparms.dstPos = cudart.make_cudaPos(0,0,0)
    myparms.srcPtr = cudart.make_cudaPitchedPtr(h_data, width * np.dtype(np.float32).itemsize, width, width)
    myparms.dstArray = cu_3darray
    myparms.extent = cudart.make_cudaExtent(width, width, num_faces)
    myparms.kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    checkCudaErrors(cudart.cudaMemcpy3D(myparms))

    texRes = cudart.cudaResourceDesc()
    texRes.resType            = cudart.cudaResourceType.cudaResourceTypeArray
    texRes.res.array.array    = cu_3darray

    texDescr = cudart.cudaTextureDesc()
    texDescr.normalizedCoords = True
    texDescr.filterMode       = cudart.cudaTextureFilterMode.cudaFilterModeLinear
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

    print("Covering Cubemap data array of {}~3 x {}: Grid size is {} x {}, each block has 8 x 8 threads".format(
           width, num_layers, dimGrid.x, dimGrid.y))

    kernelHelper = common.KernelHelper(simpleCubemapTexture, devID)
    _transformKernel = kernelHelper.getFunction(b'transformKernel')
    kernelArgs = ((d_data, width, tex),(ctypes.c_void_p, ctypes.c_int, None))
    checkCudaErrors(cuda.cuLaunchKernel(_transformKernel,
                                        dimGrid.x, dimGrid.y, dimGrid.z,         # grid dim
                                        dimBlock.x, dimBlock.y, dimBlock.z,      # block dim
                                        0, cuda.CUstream(0),                     # shared mem and stream
                                        kernelArgs, 0))                          # arguments

    checkCudaErrors(cudart.cudaDeviceSynchronize())

    start = time.time()

    # Execute the kernel
    checkCudaErrors(cuda.cuLaunchKernel(_transformKernel,
                                        dimGrid.x, dimGrid.y, dimGrid.z,         # grid dim
                                        dimBlock.x, dimBlock.y, dimBlock.z,      # block dim
                                        0, cuda.CUstream(0),                     # shared mem and stream
                                        kernelArgs, 0))                          # arguments

    checkCudaErrors(cudart.cudaDeviceSynchronize())
    stop = time.time()
    print("Processing time: {:.3f} msec".format(stop - start))
    print("{:.2f} Mtexlookups/sec".format(cubemap_size / ((stop - start + 1) / 1000.0) / 1e6))

    # Allocate mem for the result on host side
    h_odata = np.zeros(cubemap_size * num_layers, dtype='float32')
    # Copy result from device to host
    checkCudaErrors(cudart.cudaMemcpy(h_odata, d_data, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

    print("Comparing kernel output to expected data")
    MIN_EPSILON_ERROR = 5.0e-3
    for i in range(cubemap_size * num_layers):
        d = h_odata[i] - h_data_ref[i]
        if math.fabs(d) > MIN_EPSILON_ERROR:
            print("Failed")
            sys.exit(-1)
    print("Passed")

    checkCudaErrors(cudart.cudaDestroyTextureObject(tex))
    checkCudaErrors(cudart.cudaFree(d_data))
    checkCudaErrors(cudart.cudaFreeArray(cu_3darray))

if __name__=="__main__":
    main()
