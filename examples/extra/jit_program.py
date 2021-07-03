# Copyright 2021 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
import ctypes
import numpy as np
from cudapython import cuda, nvrtc

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Cuda Error: {}'.format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError('Nvrtc Error: {}'.format(err))
    else:
        raise RuntimeError('Unknown error type: {}'.format(err))

saxpy = '''\
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a * x[tid] + y[tid];
    }
}
'''

def main():
    # Init
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)

    # Device
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)

    # Ctx
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)

    # Create program
    err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice)
    ASSERT_DRV(err)
    err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice)
    ASSERT_DRV(err)

    err, prog = nvrtc.nvrtcCreateProgram(str.encode(saxpy), b'saxpy.cu', 0, [], [])
    ASSERT_DRV(err)

    # Compile program
    opts = [b'--fmad=false', bytes('--gpu-architecture=sm_' + str(major) + str(minor), 'ascii')]
    err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)
    ASSERT_DRV(err)

    # Get log from compilation
    err, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
    ASSERT_DRV(err)
    log = b' ' * logSize
    err, = nvrtc.nvrtcGetProgramLog(prog, log)
    ASSERT_DRV(err)
    print(log.decode())

    # Get CUBIN from compilation
    err, cubinSize = nvrtc.nvrtcGetCUBINSize(prog)
    ASSERT_DRV(err)
    cubin = b' ' * cubinSize
    err, = nvrtc.nvrtcGetCUBIN(prog, cubin)
    ASSERT_DRV(err)

    # Load cubin as module data and retrieve function
    cubin = np.char.array(cubin)
    err, module = cuda.cuModuleLoadData(cubin)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(module, b'saxpy')
    ASSERT_DRV(err)

    # Test the kernel
    NUM_THREADS = 128
    NUM_BLOCKS = 32

    a = np.array([2.0], dtype=np.float32)
    n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
    bufferSize = n * a.itemsize

    err, dX = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)
    err, dY = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)
    err, dOut = cuda.cuMemAlloc(bufferSize)
    ASSERT_DRV(err)

    hX = np.random.rand(n).astype(dtype=np.float32)
    hY = np.random.rand(n).astype(dtype=np.float32)
    hOut = np.zeros(n).astype(dtype=np.float32)

    err, stream = cuda.cuStreamCreate(0)
    ASSERT_DRV(err)

    err, = cuda.cuMemcpyHtoDAsync(dX, hX, bufferSize, stream)
    ASSERT_DRV(err)
    err, = cuda.cuMemcpyHtoDAsync(dY, hY, bufferSize, stream)
    ASSERT_DRV(err)

    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    # Assert values are different before running kernel
    hZ = a * hX + hY
    if np.allclose(hOut, hZ):
        raise ValueError('Error inside tolerence for host-device vectors')

    arg_values = (a, dX, dY, dOut, n)
    arg_types = (ctypes.c_float, None, None, None, ctypes.c_size_t)
    err, = cuda.cuLaunchKernel(kernel,
                              NUM_BLOCKS, 1, 1,          # grid dim
                              NUM_THREADS, 1, 1,         # block dim
                              0, stream,                 # shared mem and stream
                              (arg_values, arg_types), 0) # arguments
    ASSERT_DRV(err)

    err, = cuda.cuMemcpyDtoHAsync(hOut, dOut, bufferSize, stream)
    ASSERT_DRV(err)
    err, = cuda.cuStreamSynchronize(stream)
    ASSERT_DRV(err)

    # Assert values are same after running kernel
    hZ = a * hX + hY
    if not np.allclose(hOut, hZ):
        raise ValueError('Error outside tolerence for host-device vectors')

    err, = cuda.cuStreamDestroy(stream)
    ASSERT_DRV(err)

    err, = cuda.cuMemFree(dX)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(dY)
    ASSERT_DRV(err)
    err, = cuda.cuMemFree(dOut)
    ASSERT_DRV(err)

    err, = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
    err, = cuda.cuCtxDestroy(context)
    ASSERT_DRV(err)

if __name__=="__main__":
    main()
